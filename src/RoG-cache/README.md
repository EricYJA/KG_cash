# RoG-cache

Question caching for [RoG](https://github.com/RManLuo/reasoning-on-graphs)'s
planner, and the ToG-vs-RoG comparison that motivates it.

RoG answers a question in two stages:

```
stage 1  planner    question -> relation paths ("rules")     <-- THE CACHED STAGE
stage 2  reasoner   paths + KG -> answer                     (always runs)
stage 3  scoring    Hits@1 / F1
```

Only stage 1 is cached. Stage 2 always runs, so any Hits@1/F1 delta between
policies is caused by the reused relation paths and nothing else.

This mirrors `src/ToG-cache`, keyed the same way (`question`) with the same
`PersistentQuestionCache`, but the cached value differs:

| | key | cached value |
|---|---|---|
| ToG | question | `cluster_chain_of_entities` (contains concrete entities) |
| RoG | question | relation paths (entity-agnostic schema) |

That difference is the experiment: a RoG relation path is grounded against the
*querying* question's own subgraph in stage 2, so a semantic hit transplants a
reasoning *schema* between questions rather than entities.

## Files

| file | role |
|---|---|
| `rog_question_cache.py` | glue over ToG's `PersistentQuestionCache`; adds `TracingQuestionCache` (reports *why* a get() hit) and the oracle-key helper |
| `gen_rule_path_api.py` | stage 1 via a chat LLM API. **Cached.** |
| `predict_answer_api.py` | stage 2 via a chat LLM API, seeded |
| `simulate_rog_cache.py` | hit rates only, no LLM/GPU — seconds, not hours |

The `--engine local` counterparts (`gen_rule_path_cached.py`,
`predict_answer_seeded.py`) are **missing** — see *Provenance* below.

## Running

Nothing here is run directly; the runners in `scripts/` build the
`kgcash/rog-eval` image and mount this directory into it.

```bash
./scripts/run_rog_cache_sim.py                      # hit rates only, free
./scripts/run_rog_cache_experiment.py               # 50 questions, all policies
./scripts/run_rog_cache_experiment.py --limit 200 --vendor tamu
./scripts/run_rog_eval.py                           # no cache, plain RoG eval
```

Each policy starts from a **cold** cache and makes a single pass over the split,
so there is no pre-warming and no train/test leak. Results land in
`artifacts/rog_cache/`; `summarize_rog_cache.py` joins each policy's hit rate to
its accuracy into `summary.csv`. Read those two together — hit rate alone cannot
tell you whether caching *hurt*.

## The API planner is not a paper reproduction

The fine-tuned `rmanluo/RoG` emits Freebase relation paths from weights alone. A
general chat model has never seen those relation names and would hallucinate
them, so `gen_rule_path_api.py` is **grounded**: it shows the model the relations
actually present within `--max-hop` of the question entity and asks it to pick a
path.

That is a strictly easier task than the paper's. **API and local Hits@1 are not
comparable**, and neither is comparable to the paper's WebQSP figures
(Hits@1 ~85, F1 ~70). The cache experiment only ever compares api-vs-api across
policies, so the grounding is a constant that cancels out.

## Provenance — read before trusting results

This directory was lost (never committed, then deleted from disk) and only
partly restored on 2026-07-16:

- **Recovered verbatim** from PyCharm local history: `rog_question_cache.py`,
  `simulate_rog_cache.py`.
- **Rewritten from scratch** against the interfaces the surviving runners in
  `scripts/` pin: `gen_rule_path_api.py`, `predict_answer_api.py`,
  `summarize_rog_cache.py`, and this README. These satisfy the same CLI and
  emit the same record schemas, but they are **not** the original code and have
  not been run end-to-end against a live LLM.
- **Not recovered:** `gen_rule_path_cached.py` and `predict_answer_seeded.py`
  (the `--engine local` GPU path). `run_rog_cache_experiment.py --engine local`
  is therefore broken until they are rewritten.

`summarize_rog_cache.py` was verified by regenerating `artifacts/rog_cache/summary.csv`
byte-identically from the surviving manifests, so its join logic is known-good.
The rest is only as good as the interfaces it was reconstructed from.

The existing numbers in `artifacts/rog_cache/` were produced on 2026-07-14 by
the **original** local-engine files (`args.txt` records `rmanluo/RoG`, 8-bit).
Nothing here reproduces them. Treat them as historical until re-run.
