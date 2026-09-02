from tqdm import tqdm
import argparse
import json
import sys
import time
from utils import *
from llm_client import LLMKeyPoolExhaustedError
import random
from client import *
from question_cache import (PersistentQuestionCache, extract_oracle_answer_key,
                            restore_cache_from_answers)
import trace_instrument
from trace_utils import TraceRecorder, set_active_trace_recorder

# One question failing is the vendor having a bad minute, and the run records it
# unanswered and carries on. This many failures back to back is something being
# down -- the endpoint, every key, or the KG -- and continuing would only fill
# the output file with empty answers.
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
    parser.add_argument("--reasoning_effort", type=str,
                        default="low", help="reasoning effort for GPT-5/o-series models: low, medium, or high.")
    parser.add_argument("--opeani_api_keys", type=str,
                        default="", help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    parser.add_argument("--num_retain_entity", type=int,
                        default=5, help="Number of entities retained during entities search.")
    parser.add_argument("--prune_tools", type=str,
                        default="llm", help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.")
    parser.add_argument("--test-limit", type=parse_test_limit,
                        default=None, help="only run the first k dataset samples, or 'all'.")
    parser.add_argument("--output-file", type=str,
                        default=None, help="path to save jsonl results. Defaults to ../output/ToG_<dataset>.jsonl.")
    parser.add_argument("--vendor", type=str,
                        default="tamu", help="LLM vendor: tamu, openai, google. When set to 'tamu', uses the httpx-based client with LLM_API_KEY env var.")
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
    parser.add_argument("--similarity-threshold", type=float, default=0.90,
                        help="Cosine similarity >= threshold (i.e. cosine distance <= 1-threshold) forces a semantic hit. Forced hits count as hits and LRU-touch the matched entry.")
    parser.add_argument("--embedder-model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model used to embed questions for the cache key.")
    parser.add_argument("--timing-log", type=str,
                        default="../output/cache_timing.jsonl",
                        help="Append a per-run timing record (JSON line) to this file. Set to '' to disable.")
    parser.add_argument("--trace-output", type=str, default="",
                        help="Record a per-question KG/LLM event trace to this .jsonl "
                             "(a pretty .json is written alongside at the end). This is "
                             "the input cache_simulator.py replays; off by default because "
                             "it costs a little overhead per call.")
    parser.add_argument("--shard-count", type=int, default=1,
                        help="Split the dataset across this many independent processes.")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Which shard this process handles (0-based, < --shard-count).")
    args = parser.parse_args()

    if args.shard_count < 1 or not (0 <= args.shard_index < args.shard_count):
        parser.error(f"--shard-index must be in [0, {args.shard_count}), "
                     f"got {args.shard_index} with --shard-count {args.shard_count}")

    trace_recorder = None
    if args.trace_output:
        trace_recorder = TraceRecorder(enabled=True, output_path=args.trace_output)
        set_active_trace_recorder(trace_recorder)
        # main_freebase did `from utils import *`, so its globals hold their own
        # references to the functions being patched; patch those too.
        patched = trace_instrument.install([sys.modules[__name__]])
        print(f"[trace] recording to {args.trace_output} (patched: {', '.join(patched)})")

    datas, question_string = prepare_dataset(args.dataset)
    if args.test_limit is not None:
        datas = datas[:min(args.test_limit, len(datas))]
    if args.shard_count > 1:
        # Strided, not contiguous: question cost varies a lot (a 1-hop lookup vs a
        # 3-depth traversal), so taking every Nth question spreads the expensive
        # ones evenly instead of leaving one shard running hours after the rest.
        total = len(datas)
        datas = datas[args.shard_index::args.shard_count]
        print(f"[shard] {args.shard_index+1}/{args.shard_count}: "
              f"{len(datas)} of {total} questions")

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

    output_path = args.output_file or os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "output", f"ToG_{args.dataset}.jsonl"
    )
    # Per-question metrics land in a sidecar next to the answers, so the timing /
    # cache summary is rebuilt from the full file at the end -- restart-safe.
    metrics_path = metrics_sidecar_path(output_path)
    processed_questions: set = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = rec.get("question")
                if isinstance(q, str):
                    processed_questions.add(q.strip())
        print(f"[resume] {len(processed_questions)} questions already in {output_path}, will skip them")
        # Those questions are never recomputed, so a cache that did not survive
        # with them would leave the rest of the run measuring an emptier cache
        # than the one the answers were produced with.
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

        Recorded per question so aggregate_run_metrics can take it back out of
        the miss times before pricing a hit against them -- the lookup that
        missed and the store that followed are costs an uncached run never pays,
        and leaving them in credits the cache with the cost of running it.
        """
        if question_cache is None:
            return 0.0
        return round(question_cache.overhead_total_s - mark, 4)

    for data in tqdm(datas):
        question = data[question_string]
        if question.strip() in processed_questions:
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

        if trace_recorder is not None:
            trace_recorder.start_question(
                question_id=record_id,
                dataset=args.dataset,
                question=question,
                question_field=question_string,
                initial_topic_entity=data.get('topic_entity'),
            )

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
                                 qid=record_id, ground_truth=ground_truth)
                    append_question_metrics(metrics_path, {
                        "id": record_id, "question": question,
                        "cache_hit": True,
                        "cache_hit_type": cache_hit_kind,
                        "elapsed_s": time.perf_counter() - t_question_start,
                        "cache_overhead_s": cache_overhead_since(overhead_start),
                        "llm_calls": llm_call_count() - calls_start,
                    })
                    if trace_recorder is not None:
                        trace_recorder.finish_question("cache_hit")
                    consecutive_failures = 0
                    continue

            topic_entity = data['topic_entity']
            cluster_chain_of_entities = []
            pre_relations = [],
            pre_heads= [-1] * len(topic_entity)
            flag_printed = False
            for depth in range(1, args.depth+1):
                if trace_recorder is not None:
                    trace_recorder.set_depth(depth)
                current_entity_relations_list = []
                i=0
                for entity in topic_entity:
                    if entity!="[FINISH_ID]":
                        retrieve_relations_with_scores = relation_search_prune(entity, topic_entity[entity], pre_relations, pre_heads[i], question, args)  # best entity triplet, entitiy_id
                        current_entity_relations_list.extend(retrieve_relations_with_scores)
                    i+=1
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
                
                    if len(entity_candidates_id) >=20:
                        entity_candidates_id = random.sample(entity_candidates_id, args.num_retain_entity)

                    if len(entity_candidates_id) ==0:
                        continue

                    scores, entity_candidates, entity_candidates_id = entity_score(question, entity_candidates_id, entity['score'], entity['relation'], args)
                
                    total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head)
            
                if len(total_candidates) ==0:
                    half_stop(question, cluster_chain_of_entities, args, qid=record_id, ground_truth=ground_truth)
                    flag_printed = True
                    break
                
                flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args)
                cluster_chain_of_entities.append(chain_of_entities)
                if flag:
                    stop, results = reasoning(question, cluster_chain_of_entities, args)
                    if stop:
                        print("ToG stoped at depth %d." % depth)
                        save_2_jsonl(question, results, cluster_chain_of_entities, file_name=args.dataset, output_file=args.output_file, qid=record_id, ground_truth=ground_truth)
                        flag_printed = True
                        break
                    else:
                        print("depth %d still not find the answer." % depth)
                        topic_entity = {entity: id2entity_name_or_type(entity) for entity in entities_id}
                        continue
                else:
                    half_stop(question, cluster_chain_of_entities, args, qid=record_id, ground_truth=ground_truth)
                    flag_printed = True
                    break
        
            if not flag_printed:
                results = generate_without_explored_paths(question, args)
                save_2_jsonl(question, results, [], file_name=args.dataset, output_file=args.output_file, qid=record_id, ground_truth=ground_truth)
                chain_to_cache = []
            else:
                chain_to_cache = cluster_chain_of_entities

            if question_cache is not None:
                question_cache.put(question, chain_to_cache, oracle_key=oracle_key)

            append_question_metrics(metrics_path, {
                "id": record_id, "question": question,
                "cache_hit": False,
                "cache_hit_type": None,
                "elapsed_s": time.perf_counter() - t_question_start,
                "cache_overhead_s": cache_overhead_since(overhead_start),
                "llm_calls": llm_call_count() - calls_start,
            })

            if trace_recorder is not None:
                trace_recorder.clear_depth()
                trace_recorder.finish_question("reasoning_stop" if flag_printed else "no_path_found")
            consecutive_failures = 0
        except LLMKeyPoolExhaustedError as exc:
            # Every key in the pool was tried for this one request and every one
            # failed, so there is nothing left to answer the next question with
            # either. Stop here, before this question is written: carrying on
            # would only add unanswered questions that score as wrong. A re-run
            # resumes from the answers already on disk.
            if trace_recorder is not None:
                trace_recorder.clear_depth()
                trace_recorder.finish_question("error")
            raise SystemExit(
                f"\nEvery API key failed on one request, so no further question "
                f"can be answered -- stopping rather than filling the output with "
                f"empty answers. Nothing unanswered was written; fix the keys and "
                f"re-run to resume.\nCause: {exc}"
            ) from exc
        except RuntimeError as exc:
            # The client already retried and rotated keys, so this question is not
            # going to answer. Record it unanswered and move on: a 1600-question
            # run must not die on one of them, and an empty answer scores as wrong
            # -- so the failure shows up in the numbers instead of quietly
            # shrinking the denominator. Nothing goes into the question cache: a
            # failed call is not a reasoning chain, and caching it would serve the
            # failure to every later question that matches this one.
            failed += 1
            consecutive_failures += 1
            print(f"[warn] question failed, recording it unanswered: {exc}", flush=True)
            save_2_jsonl(question, "", [], file_name=args.dataset,
                         output_file=args.output_file, qid=record_id,
                         ground_truth=ground_truth)
            append_question_metrics(metrics_path, {
                "id": record_id, "question": question,
                # The cache can serve a question and the answer call after it
                # still fail; that is a hit whose answer was lost, not a miss.
                "cache_hit": cache_hit_kind is not None,
                "cache_hit_type": cache_hit_kind,
                "elapsed_s": time.perf_counter() - t_question_start,
                "cache_overhead_s": cache_overhead_since(overhead_start),
                "llm_calls": llm_call_count() - calls_start,
                # Counted like any other question so the denominator stays the
                # whole split; flagged so an analysis can tell the two apart --
                # aggregate_run_metrics keeps it in hit_rate and out of the
                # averages, because a question that died in the client's retry
                # loop times the vendor being down, not what a miss costs.
                "failed": True, "error": str(exc),
            })
            if trace_recorder is not None:
                trace_recorder.clear_depth()
                trace_recorder.finish_question("error")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                # Sparse failures are the vendor having a bad minute; this many in
                # a row is the endpoint, the keys, or the KG being down, and the
                # rest of the run would be a file of empty answers.
                raise RuntimeError(
                    f"{consecutive_failures} questions failed in a row -- stopping "
                    f"rather than filling the output with empty answers. "
                    f"Last error: {exc}"
                ) from exc
            continue

    if trace_recorder is not None:
        # Writes the pretty .json array alongside the appended .jsonl.
        trace_recorder.finalize_run()
        print(f"[trace] wrote {args.trace_output} (+ pretty .json)")

    if failed:
        print(f"WARNING: {failed} questions were recorded unanswered because their "
              f"LLM calls failed; they score as wrong.")

    if question_cache is not None:
        print("[question_cache] stats: " + json.dumps(question_cache.stats()))

    # Restart-safe: rebuild timing + summary + cache breakdown from the full
    # metrics sidecar, so a resumed run still reports whole-dataset numbers.
    # llm_calls_saved uses the same estimator as estimated_time_saved_s: a hit is
    # assumed to have otherwise cost the average miss's LLM calls.
    timing, summary, cache_breakdown, _per_loop = aggregate_run_metrics(metrics_path)
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
            "script": "main_freebase.py",
            # Two instances on different SPARQL backends may share this log;
            # without the endpoint their timing records are indistinguishable.
            "kg_endpoint": os.environ.get("SPARQL_ENDPOINT", ""),
            "dataset": args.dataset,
            "test_limit": args.test_limit,
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
