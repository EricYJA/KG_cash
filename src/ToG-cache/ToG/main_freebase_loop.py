from tqdm import tqdm
import argparse
import json
import time
from utils import *
from llm_client import LLMKeyPoolExhaustedError
import random
from client import *
from question_cache import (PersistentQuestionCache, extract_oracle_answer_key,
                            restore_cache_from_answers)

# See main_freebase.py: sparse failures are survivable, this many in a row means
# something is down and the rest of the run would be empty answers.
MAX_CONSECUTIVE_FAILURES = 20

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="cwq", help="choose the dataset.")
    parser.add_argument("--max_length", type=int,
                        default=256, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.4, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0, help="the temperature in reasoning stage.")
    parser.add_argument("--width", type=int,
                        default=3, help="choose the search width of ToG.")
    parser.add_argument("--depth", type=int,
                        default=3, help="choose the search depth of ToG.")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-3.5-turbo", help="base LLM model.")
    parser.add_argument("--opeani_api_keys", type=str,
                        default="",
                        help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    parser.add_argument("--num_retain_entity", type=int,
                        default=5, help="Number of entities retained during entities search.")
    parser.add_argument("--prune_tools", type=str,
                        default="llm", help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.")
    parser.add_argument("--test-limit", type=parse_test_limit,
                        default=None, help="only run the first k dataset samples, or 'all'.")
    parser.add_argument("--output-file", type=str,
                        default=None, help="path to save jsonl results. Defaults to ../output/ToG_<dataset>.jsonl.")
    parser.add_argument("--vendor", type=str,
                        default="tamu",
                        help="LLM vendor: tamu, openai, google. When set to 'tamu', uses the httpx-based client with LLM_API_KEY env var.")
    parser.add_argument("--model", type=str, default="",
                        help="Override the vendor's default model id (empty = use vendor preset default).")
    parser.add_argument("--question-cache-path", type=str,
                        default="../output/question_chain_cache.json",
                        help="Path to persistent per-question chain cache (JSON). On hit, Virtuoso and per-loop LLM calls are skipped; final answer is still generated.")
    parser.add_argument("--question-cache-capacity", type=int,
                        default=4096, help="Max number of cached questions (LRU eviction).")
    parser.add_argument("--no-question-cache", action="store_true",
                        help="Disable the persistent per-question cache.")
    parser.add_argument("--cache-policy", type=str, default="semantic_lru",
                        choices=["exact", "semantic_lru", "semantic_lfu", "semantic_oracle"],
                        help="Cache hit policy. 'exact' = key only. 'semantic_lru' = exact + cosine-similarity fallback (LRU eviction). 'semantic_oracle' = exact + cosine-similarity AND gold-answer-overlap (upper bound for accuracy-preserving semantic caching; requires gold answers in dataset).")
    parser.add_argument("--similarity-threshold", type=float, default=0.95,
                        help="Cosine similarity >= threshold (i.e. cosine distance <= 1-threshold) forces a semantic hit. Forced hits count as hits and LRU-touch the matched entry.")
    parser.add_argument("--embedder-model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model used to embed questions for the cache key.")
    parser.add_argument("--loop", type=int,
                        default=1, help="number of times to loop over the dataset samples.")
    parser.add_argument("--timing-log", type=str,
                        default="../output/cache_timing.jsonl",
                        help="Append a per-run timing record (JSON line) to this file. Set to '' to disable.")
    args = parser.parse_args()

    datas, question_string = prepare_dataset(args.dataset)
    if args.test_limit is not None:
        datas = datas[:min(args.test_limit, len(datas))]

    question_cache = None
    if not args.no_question_cache:
        question_cache = PersistentQuestionCache(
            path=args.question_cache_path,
            capacity=args.question_cache_capacity,
            policy=args.cache_policy,
            similarity_threshold=args.similarity_threshold,
            embedder_model=args.embedder_model,
        )
        print(f"[question_cache] policy={args.cache_policy}, loaded {len(question_cache._store)} entries from {args.question_cache_path}")
        # Before the first question, so the embedder's one-off load is not
        # charged to it as cache overhead and subtracted from its traversal time.
        question_cache.warm_embedder()

    per_loop_stats: list = []

    from collections import Counter
    output_path = args.output_file or os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "output", f"ToG_{args.dataset}.jsonl"
    )
    # Per-question metrics sidecar -> restart-safe timing/summary (see utils).
    metrics_path = metrics_sidecar_path(output_path)
    processed_counts: Counter = Counter()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = rec.get("question")
                if isinstance(q, str):
                    processed_counts[q.strip()] += 1
        print(f"[resume] {len(processed_counts)} unique questions seen in {output_path}; "
              f"each loop will skip questions that already have results for that pass")
        # Those passes are never recomputed, so a cache that did not survive with
        # them would leave the rest of the run measuring an emptier cache than
        # the one the answers were produced with.
        restored = restore_cache_from_answers(
            question_cache, output_path, metrics_path,
            oracle_key_by_question=(
                {d[question_string].strip(): extract_oracle_answer_key(d, args.dataset)
                 for d in datas}
                if (question_cache is not None
                    and args.cache_policy == "semantic_oracle") else None),
        )
        if restored:
            print(f"[resume] restored {restored} cache entries from {output_path} "
                  f"that the cache file no longer held")

    failed = 0               # questions recorded unanswered after an LLM failure
    consecutive_failures = 0  # trips MAX_CONSECUTIVE_FAILURES when the run is broken

    def cache_overhead_since(mark):
        """Seconds the cache itself spent on this question; 0.0 with no cache.

        See main_freebase.py: recorded per question so aggregate_run_metrics can
        take it back out of the miss times before pricing a hit against them.
        """
        if question_cache is None:
            return 0.0
        return round(question_cache.overhead_total_s - mark, 4)

    for loop_idx in range(args.loop):
        if args.loop > 1:
            print(f"\n--- Starting loop {loop_idx + 1}/{args.loop} ---")

        loop_hit_times: list = []
        loop_miss_times: list = []
        loop_t_start = time.perf_counter()

        for data in tqdm(datas, desc=f"Loop {loop_idx + 1}" if args.loop > 1 else None):
            question = data[question_string]
            if processed_counts.get(question.strip(), 0) > loop_idx:
                continue
            t_question_start = time.perf_counter()
            calls_start = llm_call_count()
            overhead_start = (question_cache.overhead_total_s
                              if question_cache is not None else 0.0)
            # Set once the cache serves this question, so a failure in the answer
            # call that follows is still counted as the hit it was.
            cache_hit_kind = None

            record_id = extract_record_id(data, args.dataset)
            ground_truth = extract_ground_truth(data, args.dataset)

            oracle_key = (extract_oracle_answer_key(data, args.dataset)
                          if (question_cache is not None and args.cache_policy == "semantic_oracle")
                          else None)

            try:
                if question_cache is not None:
                    cached_chain = question_cache.get(question, oracle_key=oracle_key)
                    if cached_chain is not None:
                        cache_hit_kind = getattr(question_cache, "last_hit_kind", None)
                        if cached_chain:
                            cached_results = generate_answer(question, cached_chain, args)
                        else:
                            cached_results = generate_without_explored_paths(question, args)
                        save_2_jsonl(question, cached_results, cached_chain,
                                     file_name=args.dataset, output_file=args.output_file,
                                     qid=record_id, ground_truth=ground_truth, loop_idx=loop_idx)
                        elapsed = time.perf_counter() - t_question_start
                        loop_hit_times.append(elapsed)
                        append_question_metrics(metrics_path, {
                            "id": record_id, "question": question, "loop_idx": loop_idx,
                            "cache_hit": True,
                            "cache_hit_type": cache_hit_kind,
                            "elapsed_s": elapsed,
                            "cache_overhead_s": cache_overhead_since(overhead_start),
                            "llm_calls": llm_call_count() - calls_start,
                        })
                        consecutive_failures = 0
                        continue

                topic_entity = data['topic_entity']
                cluster_chain_of_entities = []
                pre_relations = [],
                pre_heads = [-1] * len(topic_entity)
                flag_printed = False
                for depth in range(1, args.depth + 1):
                    current_entity_relations_list = []
                    i = 0
                    for entity in topic_entity:
                        if entity != "[FINISH_ID]":
                            retrieve_relations_with_scores = relation_search_prune(entity, topic_entity[entity],
                                                                                   pre_relations, pre_heads[i], question,
                                                                                   args)  # best entity triplet, entitiy_id
                            current_entity_relations_list.extend(retrieve_relations_with_scores)
                        i += 1
                    total_candidates = []
                    total_scores = []
                    total_relations = []
                    total_entities_id = []
                    total_topic_entities = []
                    total_head = []

                    for entity in current_entity_relations_list:
                        if entity['head']:
                            entity_candidates_id = entity_search(entity['entity'], entity['relation'], True)
                        else:
                            entity_candidates_id = entity_search(entity['entity'], entity['relation'], False)

                        if len(entity_candidates_id) >= 20:
                            entity_candidates_id = random.sample(entity_candidates_id, args.num_retain_entity)

                        if len(entity_candidates_id) == 0:
                            continue

                        scores, entity_candidates, entity_candidates_id = entity_score(question, entity_candidates_id,
                                                                                       entity['score'], entity['relation'],
                                                                                       args)

                        total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(
                            entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores,
                            total_relations, total_entities_id, total_topic_entities, total_head)

                    if len(total_candidates) == 0:
                        half_stop(question, cluster_chain_of_entities, args, qid=record_id, ground_truth=ground_truth, loop_idx=loop_idx)
                        flag_printed = True
                        break

                    flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id,
                                                                                                  total_relations,
                                                                                                  total_candidates,
                                                                                                  total_topic_entities,
                                                                                                  total_head, total_scores,
                                                                                                  args)
                    cluster_chain_of_entities.append(chain_of_entities)
                    if flag:
                        stop, results = reasoning(question, cluster_chain_of_entities, args)
                        if stop:
                            print("ToG stoped at depth %d." % depth)
                            save_2_jsonl(question, results, cluster_chain_of_entities, file_name=args.dataset,
                                         output_file=args.output_file, qid=record_id, ground_truth=ground_truth,
                                         loop_idx=loop_idx)
                            flag_printed = True
                            break
                        else:
                            print("depth %d still not find the answer." % depth)
                            topic_entity = {entity: id2entity_name_or_type(entity) for entity in entities_id}
                            continue
                    else:
                        half_stop(question, cluster_chain_of_entities, args, qid=record_id, ground_truth=ground_truth, loop_idx=loop_idx)
                        flag_printed = True
                        break

                if not flag_printed:
                    results = generate_without_explored_paths(question, args)
                    save_2_jsonl(question, results, [], file_name=args.dataset, output_file=args.output_file, qid=record_id, ground_truth=ground_truth, loop_idx=loop_idx)
                    chain_to_cache = []
                else:
                    chain_to_cache = cluster_chain_of_entities

                if question_cache is not None:
                    question_cache.put(question, chain_to_cache, oracle_key=oracle_key)

                elapsed = time.perf_counter() - t_question_start
                loop_miss_times.append(elapsed)
                append_question_metrics(metrics_path, {
                    "id": record_id, "question": question, "loop_idx": loop_idx,
                    "cache_hit": False,
                    "cache_hit_type": None,
                    "elapsed_s": elapsed,
                    "cache_overhead_s": cache_overhead_since(overhead_start),
                    "llm_calls": llm_call_count() - calls_start,
                })
                consecutive_failures = 0
            except LLMKeyPoolExhaustedError as exc:
                # See main_freebase.py: every key was tried for this one request
                # and every one failed, so nothing is left to answer the next
                # question with. Stop before this question is written.
                raise SystemExit(
                    f"\nEvery API key failed on one request, so no further question "
                    f"can be answered -- stopping rather than filling the output "
                    f"with empty answers. Nothing unanswered was written; fix the "
                    f"keys and re-run to resume.\nCause: {exc}"
                ) from exc
            except RuntimeError as exc:
                # See main_freebase.py: one question's LLM calls failing must not
                # end the run. Recorded unanswered (so it scores as wrong rather
                # than shrinking the split) and never cached.
                failed += 1
                consecutive_failures += 1
                print(f"[warn] question failed, recording it unanswered: {exc}", flush=True)
                save_2_jsonl(question, "", [], file_name=args.dataset,
                             output_file=args.output_file, qid=record_id,
                             ground_truth=ground_truth, loop_idx=loop_idx)
                append_question_metrics(metrics_path, {
                    "id": record_id, "question": question, "loop_idx": loop_idx,
                    # A hit whose answer call failed is still a hit, not a miss.
                    "cache_hit": cache_hit_kind is not None,
                    "cache_hit_type": cache_hit_kind,
                    "elapsed_s": time.perf_counter() - t_question_start,
                    "cache_overhead_s": cache_overhead_since(overhead_start),
                    "llm_calls": llm_call_count() - calls_start,
                    "failed": True, "error": str(exc),
                })
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(
                        f"{consecutive_failures} questions failed in a row -- stopping "
                        f"rather than filling the output with empty answers. "
                        f"Last error: {exc}"
                    ) from exc
                continue

        loop_wall = time.perf_counter() - loop_t_start
        n_lh, n_lm = len(loop_hit_times), len(loop_miss_times)
        avg_lh = (sum(loop_hit_times) / n_lh) if n_lh else 0.0
        avg_lm = (sum(loop_miss_times) / n_lm) if n_lm else 0.0
        per_loop_stats.append({
            "loop": loop_idx + 1,
            "wall_s": round(loop_wall, 3),
            "hits": n_lh,
            "misses": n_lm,
            "avg_hit_s": round(avg_lh, 3),
            "avg_miss_s": round(avg_lm, 3),
        })
        print(f"[question_cache] loop {loop_idx + 1} timing: " + json.dumps(per_loop_stats[-1]))

    if failed:
        print(f"WARNING: {failed} questions were recorded unanswered because their "
              f"LLM calls failed; they score as wrong.")

    if question_cache is not None:
        print("[question_cache] stats: " + json.dumps(question_cache.stats()))

    # Restart-safe: rebuild timing + summary + cache breakdown + per-loop from the
    # full metrics sidecar, so a resumed run reports whole-dataset numbers.
    timing, summary, cache_breakdown, per_loop = aggregate_run_metrics(metrics_path)
    timing["per_loop"] = per_loop
    print("[question_cache] timing: " + json.dumps(timing))
    summary = {**summary, "vendor": args.vendor, "model": args.model or ""}
    print("[question_cache] summary: " + json.dumps(summary))

    cache_stats = None
    if question_cache is not None:
        # Config fields from the live cache; hit/miss counts from the sidecar so
        # they stay correct across restarts.
        cache_stats = {**question_cache.stats(), **cache_breakdown,
                       "hits": timing["hits"], "misses": timing["misses"],
                       "hit_rate": summary["hit_rate"]}

    if args.timing_log:
        from datetime import datetime, timezone
        log_record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "script": "main_freebase_loop.py",
            # Two instances on different SPARQL backends may share this log;
            # without the endpoint their timing records are indistinguishable.
            "kg_endpoint": os.environ.get("SPARQL_ENDPOINT", ""),
            "dataset": args.dataset,
            "test_limit": args.test_limit,
            "loop": args.loop,
            "policy": args.cache_policy if not args.no_question_cache else "off",
            "similarity_threshold": args.similarity_threshold,
            "capacity": args.question_cache_capacity,
            "timing": timing,
            "cache_stats": cache_stats,
            **summary,
        }
        log_dir = os.path.dirname(args.timing_log)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(args.timing_log, "a") as f:
            f.write(json.dumps(log_record) + "\n")
        print(f"[question_cache] appended timing record to {args.timing_log}")
