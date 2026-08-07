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
from tqdm import tqdm

import utils

datasets.disable_progress_bar()

from llm_client import ChatMessage, LLMChatClient  # noqa: E402  (ToG's client, reused)
from llm_config import resolve_llm_config  # noqa: E402
from rog_e2e_metrics import (  # noqa: E402  (ToG's sidecar helpers, reused)
    append_question_metrics,
    metrics_sidecar_path,
)
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


def build_prompt(sample, max_hop, n_beam):
    """Return (prompt_text, n_candidate_relations); prompt is None if unusable."""
    graph = utils.build_graph(sample["graph"])
    q_entities = sample["q_entity"]
    if isinstance(q_entities, str):
        q_entities = [q_entities]
    relations = candidate_relations(graph, q_entities, max_hop)
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


def ground_paths_for(sample):
    """Gold relation paths, exactly as gen_rule_path.py computes them."""
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
    args = ap.parse_args()

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
            prompt, n_relations = build_prompt(sample, args.max_hop, args.n_beam)
            if prompt is None:
                rel_paths, text = [], ""
            else:
                text = client.complete_json(
                    [
                        ChatMessage(role="system", content=SYSTEM_PROMPT),
                        ChatMessage(role="user", content=prompt),
                    ],
                    temperature=0.0,
                )
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
            }
            elapsed = time.time() - t0
            misses += 1
            miss_total_s += elapsed
            if cache is not None:
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
                    "ground_paths": ground_paths_for(sample),
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
            },
        )
    fout.close()

    wall_s = time.time() - wall_start
    n = hits + misses
    timing = {
        "hits": hits,
        "misses": misses,
        "hit_total_s": round(hit_total_s, 3),
        "miss_total_s": round(miss_total_s, 3),
        "avg_hit_s": round(hit_total_s / hits, 3) if hits else 0.0,
        "avg_miss_s": round(miss_total_s / misses, 3) if misses else 0.0,
        "estimated_time_saved_s": round(hits * (miss_total_s / misses), 3) if (hits and misses) else 0.0,
        "speedup_x": None,
        "planner_full_speedup_x": None,
    }
    if hits and misses:
        would_have_been = timing["estimated_time_saved_s"] + wall_s
        timing["speedup_x"] = round(would_have_been / wall_s, 3) if wall_s else None
        # PLANNER-STAGE ONLY -- not a full-system speedup. Every second measured
        # here is inside this script's per-question timer (cache lookup, or prompt
        # build + one planner call); stage 2's grounding and reasoner call are in
        # another process entirely and are absent from both terms. Without the
        # cache every request would have been a miss, so the amortised no-cache
        # stage-1 time is n*avg_miss against the actual hit+miss stage-1 time.
        # The real end-to-end number comes from rog_e2e_metrics.py, which joins
        # the sidecar below with stage 2's; it is strictly lower than this one.
        served_s = hit_total_s + miss_total_s
        baseline_s = n * (miss_total_s / misses)
        timing["planner_full_speedup_x"] = round(baseline_s / served_s, 3) if served_s else None

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "script": "gen_rule_path_api.py",
        "dataset": args.d,
        "test_limit": args.split,
        "policy": "off" if cache is None else args.cache_policy,
        "similarity_threshold": args.similarity_threshold,
        "capacity": args.question_cache_capacity,
        "timing": timing,
        "cache_stats": cache.stats() if cache is not None else None,
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
            "planner_llm_calls": misses,
            "planner_llm_calls_saved": hits,
            "hit_rate": (hits / n) if n else 0.0,
            "wall_s_total": round(wall_s, 2),
            "prediction_file": prediction_file,
            "stage1_metrics_file": metrics_path,
            "vendor": config.vendor,
            "model": config.model,
        }
    )
    with open(os.path.join(output_dir, "cache_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(
        f"\nplanner done: {n} questions, {misses} LLM calls, {hits} saved by cache "
        f"(hit rate {100 * stats['hit_rate']:.1f}%), {wall_s:.1f}s"
    )
    return prediction_file


if __name__ == "__main__":
    main()
