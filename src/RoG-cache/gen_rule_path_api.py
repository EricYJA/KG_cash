"""RoG stage 1 (planner) against a chat LLM API instead of the fine-tuned model.

    question  ->  relation paths ("rules")        <-- THE CACHED STAGE

The drop-in API replacement for RoG's src/qa_prediction/gen_rule_path.py. Same
CLI, same output file layout, same record schema, so stage 2 and RoG's own
evaluate_results.py consume the output unchanged. Adds `--cache-policy` and
friends: a hit skips the LLM call entirely.

NOT a paper reproduction. The fine-tuned rmanluo/RoG emits relation paths from
weights alone; a general chat model has never seen Freebase relation names and
would hallucinate them. So this planner is *grounded*: it shows the model the
relations actually present within --max-hop of the question entity and asks it
to pick a path. That is a strictly easier task than the paper's, and the two
planners' Hits@1 are NOT comparable. Use `--engine local` for the paper's method.

The cache experiment only compares api-vs-api across policies, so the grounding
is a constant and cancels out.

Usage (inside the rog-eval image; see scripts/run_rog_cache_experiment.py):
    python gen_rule_path_api.py --model_name RoG -d RoG-webqsp --split test \
        --n_beam 3 --vendor tamu --no-question-cache
    python gen_rule_path_api.py ... --cache-policy semantic_lru \
        --similarity-threshold 0.85 --question-cache-path /out/cache/x.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import datasets
from datasets import load_dataset

# Before any load_dataset call: the image's datasets is older than the one
# that may have written the shared HF cache. See datasets_compat.
import datasets_compat  # noqa: F401,E402
from tqdm import tqdm

import utils

datasets.disable_progress_bar()

from llm_client import (ChatMessage, LLMChatClient,  # noqa: E402  (ToG's client, reused)
                        LLMKeyPoolExhaustedError)
from llm_config import resolve_llm_config  # noqa: E402
from rog_e2e_metrics import (  # noqa: E402  (ToG's sidecar helpers, reused)
    aggregate_run_metrics,
    append_question_metrics,
    metrics_sidecar_path,
)
from sparql_kg import add_kg_args, kg_from_args  # noqa: E402
from rog_question_cache import (  # noqa: E402
    TracingQuestionCache,
    extract_oracle_answer_key,
)

PATH_RE = r"<PATH>(.*?)</PATH>"

SYSTEM_PROMPT = (
    "You are a knowledge-graph reasoning planner. Given a question and the "
    "relations available near the question entity, you output relation paths "
    "that lead from the question entity to the answer."
)

# The model must not invent relation names: stage 2 grounds each path against the
# real subgraph, and an unknown relation simply yields no paths (a silent miss).
USER_PROMPT = """\
Question: {question}
Question entity: {q_entity}

Relations available within {max_hop} hop(s) of the question entity:
{relations}

Output exactly {n_beam} candidate relation path(s), best first, that would lead
from the question entity to the answer. Rules:
  - Use ONLY relations from the list above, copied exactly.
  - A path is 1 to {max_hop} relations, joined by <SEP>.
  - Wrap each path in <PATH></PATH>, one per line. Output nothing else.

Example (2 relations):
<PATH>people.person.sibling_s<SEP>people.sibling_relationship.sibling</PATH>"""


def get_output_file(path, force=False):
    """Reopen `path` for append, returning ids already done (RoG's own helper)."""
    if not os.path.exists(path) or force:
        return open(path, "w"), []
    with open(path, "r") as f:
        processed = []
        for line in f:
            try:
                processed.append(json.loads(line)["id"])
            except json.JSONDecodeError:
                raise ValueError(f"Error in line: {line}")
    return open(path, "a"), processed


def parse_prediction(text):
    """Pull `<PATH>a<SEP>b</PATH>` paths out of a raw completion.

    Mirrors gen_rule_path.py's parser; it just reads one multi-path completion
    instead of one string per beam. Malformed paths are dropped, matching the
    upstream behaviour of returning [] rather than raising.
    """
    results = []
    for match in re.findall(PATH_RE, text, flags=re.DOTALL):
        rules = [rel.strip() for rel in match.split("<SEP>") if rel.strip()]
        if rules:
            results.append(rules)
    return results


def candidate_relations(graph, q_entities, max_hop):
    """Relation names reachable within `max_hop` undirected hops of `q_entities`.

    RoG's graphs are nx.Graph with a single `relation` attribute per edge, so
    this is a plain BFS collecting edge labels. Sorted for prompt stability:
    the same question must produce the same prompt on every run.
    """
    frontier = {e for e in q_entities if graph.has_node(e)}
    visited = set(frontier)
    relations = set()
    for _ in range(max_hop):
        nxt = set()
        for node in frontier:
            for neighbor in graph.neighbors(node):
                relations.add(graph[node][neighbor]["relation"])
                if neighbor not in visited:
                    visited.add(neighbor)
                    nxt.add(neighbor)
        frontier = nxt
        if not frontier:
            break
    return sorted(relations)


def build_prompt(sample, max_hop, n_beam, kg=None):
    """Return (prompt_text, n_candidate_relations); prompt is None if unusable.

    With `kg` the relation menu comes from the live endpoint instead of the row's
    bundled subgraph. Everything downstream -- prompt wording, the n_beam cap, the
    empty-menu skip -- is identical, so the only thing that moves between the two
    backends is which relations the planner is allowed to choose from.
    """
    q_entities = sample["q_entity"]
    if isinstance(q_entities, str):
        q_entities = [q_entities]
    if kg is not None:
        relations = kg.relations_within(q_entities, max_hop)
    else:
        relations = candidate_relations(utils.build_graph(sample["graph"]),
                                        q_entities, max_hop)
    if not relations:
        # No q_entity in the subgraph => nothing to ground against. Upstream would
        # still call the model; we skip the spend and record an empty prediction.
        return None, 0
    return (
        USER_PROMPT.format(
            question=sample["question"],
            q_entity=", ".join(q_entities),
            max_hop=max_hop,
            n_beam=n_beam,
            relations="\n".join(f"  {r}" for r in relations),
        ),
        len(relations),
    )


def ground_paths_for(sample, kg=None, max_hop=2):
    """Gold relation paths, exactly as gen_rule_path.py computes them.

    With `kg` the shortest q_entity->a_entity connections are searched in the live
    KG (bounded to `max_hop`; see sparql_kg.shortest_rules) rather than inside the
    bundled subgraph. This field is metadata for the runs here -- stage 2 only
    reads it under --use_true, and run_rog_eval.py's planner sanity check scores
    predictions against it -- but it is exactly the field that reveals whether the
    bundled subgraph was doing the retrieving, so it has to follow the backend.
    """
    if kg is not None:
        return kg.shortest_rules(sample["q_entity"], sample["a_entity"], max_hop)
    graph = utils.build_graph(sample["graph"])
    paths = utils.get_truth_paths(sample["q_entity"], sample["a_entity"], graph)
    return [list(t) for t in {tuple(p[1] for p in path) for path in paths}]


def build_cache(args):
    """Return a TracingQuestionCache, or None when caching is disabled."""
    if args.no_question_cache:
        return None
    return TracingQuestionCache(
        path=args.question_cache_path,
        capacity=args.question_cache_capacity,
        policy=args.cache_policy,
        similarity_threshold=args.similarity_threshold,
        embedder_model=args.embedder_model,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data_path", default="rmanluo")
    ap.add_argument("--d", "-d", default="RoG-webqsp")
    ap.add_argument("--split", default="test")
    ap.add_argument("--output_path", default="results/gen_rule_path")
    ap.add_argument("--model_name", default="RoG", help="model_name for save results")
    ap.add_argument("--n_beam", type=int, default=3, help="number of paths to request")
    ap.add_argument("--max-hop", dest="max_hop", type=int, default=2)
    ap.add_argument("--force", "-f", action="store_true")
    ap.add_argument("--debug", action="store_true")
    # LLM
    ap.add_argument("--vendor", default="tamu")
    ap.add_argument("--model", default=None, help="override the vendor's default model")
    ap.add_argument("--timeout-s", type=float, default=120.0)
    # cache
    ap.add_argument("--no-question-cache", action="store_true")
    ap.add_argument("--cache-policy", default="exact")
    ap.add_argument("--similarity-threshold", type=float, default=0.9)
    ap.add_argument("--question-cache-capacity", type=int, default=4096)
    ap.add_argument("--question-cache-path", default="cache/rog_question_cache.json")
    ap.add_argument("--embedder-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--timing-log", default=None, help="append a JSON timing record here")
    # KG
    add_kg_args(ap)
    args = ap.parse_args()

    # Built once and shared by the whole pass: it holds only a name->mid memo, so
    # reusing it costs nothing the run should be paying per question.
    kg = kg_from_args(args)

    config = resolve_llm_config(vendor=args.vendor, model=args.model)
    client = LLMChatClient.from_connection_config(config, timeout_s=args.timeout_s)
    print(f"planner LLM: {config.vendor}/{config.model}")

    dataset = load_dataset(os.path.join(args.data_path, args.d), split=args.split)
    output_dir = os.path.join(args.output_path, args.d, args.model_name, args.split)
    os.makedirs(output_dir, exist_ok=True)
    print("Save results to: ", output_dir)

    cache = build_cache(args)
    if cache is not None:
        os.makedirs(os.path.dirname(args.question_cache_path) or ".", exist_ok=True)

    prediction_file = os.path.join(output_dir, f"predictions_{args.n_beam}_False.jsonl")
    fout, processed = get_output_file(prediction_file, force=args.force)

    # Per-question sidecar, in ToG's format. The aggregate `timing` block below
    # only ever sees stage 1, so it can only ever report a planner-stage
    # speedup; this file is stage 1's half of the whole-question time that
    # rog_e2e_metrics.py joins with stage 2's to get a real full-system number.
    metrics_path = metrics_sidecar_path(prediction_file)
    if args.force and metrics_path and os.path.exists(metrics_path):
        # --force truncates the predictions file, so the sidecar has to go too or
        # the join would pair fresh stage-2 times with a previous run's stage 1.
        os.remove(metrics_path)

    # Timing is split by hit/miss so the summary can report time actually saved
    # rather than a modelled estimate.
    hits = misses = 0
    failed = 0  # questions whose planner call could not be completed
    hit_total_s = miss_total_s = 0.0
    wall_start = time.time()

    for sample in tqdm(dataset):
        qid = sample["id"]
        if qid in processed:
            continue
        question = sample["question"]
        oracle_key = (
            extract_oracle_answer_key(sample, args.d)
            if cache is not None and args.cache_policy == "semantic_oracle"
            else None
        )

        t0 = time.time()
        prompt = None  # stays None on a hit: no prompt was built, none was sent
        cached = cache.get(question, oracle_key=oracle_key) if cache is not None else None
        if cached is not None:
            rel_paths = cached
            raw_output = None
            elapsed = time.time() - t0
            hits += 1
            hit_total_s += elapsed
            cache_info = {
                "hit": True,
                "kind": cache.last_hit["kind"],
                "source_question": cache.last_hit["source"],
                "similarity": cache.last_hit["similarity"],
            }
        else:
            prompt, n_relations = build_prompt(sample, args.max_hop, args.n_beam, kg)
            planner_error = None
            if prompt is None:
                rel_paths, text = [], ""
            else:
                try:
                    text = client.complete_json(
                        [
                            ChatMessage(role="system", content=SYSTEM_PROMPT),
                            ChatMessage(role="user", content=prompt),
                        ],
                        temperature=0.0,
                    )
                except LLMKeyPoolExhaustedError as exc:
                    # Every key in the pool was tried for this one request and
                    # every one failed, so nothing is left to plan the next
                    # question with either. Each remaining question would be
                    # written with no paths and scored as a miss, so a dead key
                    # pool would come back as a plausible-looking accuracy
                    # number. Stop instead.
                    raise SystemExit(
                        f"\nEvery API key failed on one planner request, so no "
                        f"further question can be planned. Stopping.\n"
                        f"Cause: {exc}"
                    ) from exc
                except RuntimeError as exc:
                    # The client already retried and rotated keys, so this call is
                    # not coming back. One question must not end a run of 1600:
                    # record it with no paths, exactly as an unpromptable question
                    # above is recorded. Stage 2 then answers it without rules and
                    # stage 3 scores it as a miss, so the failure lands in the
                    # numbers instead of quietly shrinking the split.
                    print(f"  [warn] planner call failed for {qid}: {exc}", flush=True)
                    planner_error = str(exc)
                    failed += 1
                    rel_paths, text = [], ""
                else:
                    rel_paths = parse_prediction(text)[: args.n_beam]
            raw_output = {
                # No beam search behind an API: there are no sequence scores to
                # report, and inventing them would be a lie. Kept for schema parity
                # with gen_rule_path.py, whose consumers only read `prediction`.
                "paths": [utils.rule_to_string(p) for p in rel_paths],
                "scores": None,
                "norm_scores": None,
                "text": text,
                "n_candidate_relations": n_relations,
                "error": planner_error,
            }
            elapsed = time.time() - t0
            misses += 1
            miss_total_s += elapsed
            # A failed call produced no plan. Caching the empty result would
            # serve it to every later question that matches this one, turning one
            # transient failure into a run-wide accuracy hole.
            if cache is not None and planner_error is None:
                cache.put(question, rel_paths, oracle_key=oracle_key)
            cache_info = {"hit": False, "kind": None, "source_question": None, "similarity": None}

        if args.debug:
            print("ID: ", qid)
            print("Question: ", question)
            print("Prediction: ", rel_paths)

        fout.write(
            json.dumps(
                {
                    "id": qid,
                    "question": question,
                    "prediction": rel_paths,
                    "ground_paths": ground_paths_for(sample, kg, args.max_hop),
                    "input": prompt,
                    "raw_output": raw_output,
                    "cache": cache_info,
                }
            )
            + "\n"
        )
        fout.flush()

        # Appended only after the answer is durably written, so a record here
        # always corresponds to a question present in predictions.jsonl.
        append_question_metrics(
            metrics_path,
            {
                "id": qid,
                "question": question,
                "cache_hit": cache_info["hit"],
                "cache_hit_type": cache_info["kind"],
                "elapsed_s": elapsed,
                "llm_calls": 0 if cache_info["hit"] else 1,
                # Counted as a question (the split must not shrink) but kept out
                # of the miss average: a question that died in the client's retry
                # loop times the vendor being down, not what a planner call costs,
                # and every speedup here is priced against that average.
                "failed": planner_error is not None,
            },
        )
    fout.close()

    process_wall_s = time.time() - wall_start

    # Rebuilt from the per-question sidecar, NOT from the counters above. Those
    # live in one process, so an interrupted policy that was resumed reported only
    # the questions its last invocation happened to handle -- 400 of 1628, with a
    # hit rate and an LLM-call count to match -- and cache_stats.json is what
    # summarize_rog_cache.py and _rog_common.assert_stages_agree read. The sidecar
    # holds every question ever recorded for this output, so this is correct after
    # any number of restarts. It is also ToG's own aggregator, so a planner number
    # and an end-to-end one come out of one function and one set of definitions.
    timing, run_summary, breakdown, _per_loop = aggregate_run_metrics(metrics_path)

    # PLANNER-STAGE ONLY -- renamed on the way out so neither can be read as a
    # system-level result. Stage 2's grounding and reasoner call happen in another
    # process and are absent from every term here; the cache never touches them,
    # and they run on hits and misses alike. The bare names are reserved for the
    # whole-question numbers summarize_rog_cache.py joins in from stage 2, and the
    # planner figure is strictly the more flattering of the two.
    #
    #   planner_speedup_x       per question: a cold planner call over a served one
    #   planner_full_speedup_x  whole run: n*avg_miss over what stage 1 actually cost
    #
    # `estimated_time_saved_s` now comes from the aggregator too, which nets out
    # what serving the hits cost (hits*avg_miss - hit_total_s). The local formula
    # this replaces omitted that subtraction and reported gross avoided time, so
    # the same column name meant two different things depending on which half of
    # summary.csv you read it from.
    timing["planner_speedup_x"] = timing.pop("speedup_x", None)
    timing["planner_full_speedup_x"] = timing.pop("full_speedup_x", None)
    n = run_summary["n_questions"]

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "script": "gen_rule_path_api.py",
        "dataset": args.d,
        "test_limit": args.split,
        "policy": "off" if cache is None else args.cache_policy,
        "similarity_threshold": args.similarity_threshold,
        "capacity": args.question_cache_capacity,
        # Recorded because the run tag cannot be trusted to carry it: every
        # artifacts/rog_cache/*_virtuoso_* run predating this flag was named for a
        # backend it never queried, since RoG had no SPARQL path at all.
        "kg_backend": args.kg_backend,
        "kg_endpoint": kg.endpoint if kg is not None else None,
        "kg_sparql_queries": kg.n_queries if kg is not None else 0,
        "timing": timing,
        # cache.stats() counts what THIS process saw and is left as-is: it is the
        # cache object's own view, and on a resume it describes the tail of the
        # run. The breakdown beside it is rebuilt from the sidecar, so that is the
        # one to read for per-policy hit counts over the whole split.
        "cache_stats": cache.stats() if cache is not None else None,
        "cache_hit_breakdown": breakdown,
    }
    if args.timing_log:
        os.makedirs(os.path.dirname(args.timing_log) or ".", exist_ok=True)
        with open(args.timing_log, "a") as f:
            f.write(json.dumps(record) + "\n")

    stats = dict(record)
    stats.update(
        {
            "split": args.split,
            "n_beam": args.n_beam,
            "n_questions": n,
            "planner_llm_calls": run_summary["llm_calls"],
            # Exactly one planner call per question, so a hit saves exactly one --
            # no estimator needed (ToG has to average, because a ToG miss costs a
            # variable number of calls).
            "planner_llm_calls_saved": timing["hits"],
            "hit_rate": run_summary["hit_rate"],
            # Sum of per-question times, which is well defined across a resume --
            # unlike this process's clock, which also covers work the per-question
            # timer excludes (the ground_paths lookup). The raw clock is kept
            # beside it for operational reference and nothing computes from it.
            "wall_s_total": run_summary["wall_s_total"],
            "process_wall_s": round(process_wall_s, 2),
            "prediction_file": prediction_file,
            "stage1_metrics_file": metrics_path,
            "vendor": config.vendor,
            "model": config.model,
        }
    )
    with open(os.path.join(output_dir, "cache_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(
        f"\nplanner done: {n} questions, {stats['planner_llm_calls']} LLM calls, "
        f"{timing['hits']} saved by cache (hit rate {100 * stats['hit_rate']:.1f}%), "
        f"{stats['wall_s_total']:.1f}s"
    )
    if hits + misses != n:
        # A resume: the line above covers the whole split (from the sidecar), this
        # one covers only what this invocation did.
        print(
            f"  this pass: {hits + misses} questions, {misses} LLM calls, "
            f"{hits} served from cache, {process_wall_s:.1f}s wall"
        )
    if failed:
        print(
            f"WARNING: {failed}/{hits + misses} questions in this pass have no plan "
            f"because the planner call failed; they are recorded with an empty "
            f"prediction, marked failed in the sidecar so they do not price a miss, "
            f"and score as misses."
        )
    return prediction_file


if __name__ == "__main__":
    main()
