# KG Backend Baseline

This repository currently centers on a small uncached knowledge-graph backend in `src/kg_backend` and a thin iterative LLM frontend in `src/llm_frontend`.

The backend is the baseline retrieval layer for later cache experiments. It is not a graph database and it does not expose a free-form query language.

## Current Status

- `src/kg_backend`: self-contained uncached KG backend
- `datasets/WebQSP_KG`: built WebQSP dataset-specific subKG that the backend can load directly
- `src/llm_frontend`: iterative one-hop LLM frontend wired to the current `kg_backend`

By default, local backend startup now prefers `datasets/WebQSP_KG` when that directory exists.

## What "Uncached" Means

In this repo, "uncached" means:

- triples are loaded from local files once at startup
- immutable adjacency indexes are built once at startup
- each query is answered directly from those baseline indexes
- no query-result memoization or path-expansion cache is used

It does not mean rescanning raw files on every request.

## Repository Layout

Main directories:

- `src/kg_backend`
- `src/llm_frontend`
- `datasets/WebQSP`
- `datasets/WebQSP_KG`
- `ref_KG_projects`

`ref_KG_projects/EPR-KGQA/data/dataset/WebQSP` is the reference source used to build `datasets/WebQSP_KG`.

## Setup

Use the shared Conda environment:

```bash
conda env create -f environment.yml
conda activate kg-cache
export PYTHONPATH=src:${PYTHONPATH}
```

If the env already exists:

```bash
conda env update -f environment.yml --prune
conda activate kg-cache
export PYTHONPATH=src:${PYTHONPATH}
```

There is intentionally no editable-install step. Run everything from the repo root with `PYTHONPATH=src`.

## KG Data Format

Supported graph inputs:

- `triples.tsv`
- `triples.parquet`

Required columns:

```text
head  relation  tail
```

Optional metadata files:

- `entities.tsv` or `entities.parquet` with columns `id`, `label`
- `relations.tsv` or `relations.parquet` with columns `id`, `label`

If metadata files are present, they must cover every entity or relation referenced by the triples.

Duplicate triples are accepted and deduplicated when the immutable adjacency index is built.

## Python Backend API

```python
from kg_backend.backend import UncachedKGBackend
from kg_backend.types import GetNeighborsQuery, PathStep

backend = UncachedKGBackend.from_data_path("datasets/WebQSP_KG")

backend.get_out_relations("m.06w2sn5")
backend.get_in_relations("m.0gxnnwq")
backend.get_neighbors("m.06w2sn5", "people.person.sibling_s", direction="out")
backend.search_entity_ids_by_name("Justin Bieber", limit=5)
backend.entity_name_exists("Justin Bieber")
backend.follow_path(
    ["m.06w2sn5"],
    [
        PathStep(relation_id="people.person.sibling_s", direction="out"),
        PathStep(relation_id="people.sibling_relationship.sibling", direction="out"),
    ],
)
backend.execute(
    GetNeighborsQuery(
        entity_id="m.06w2sn5",
        relation_id="people.person.sibling_s",
        direction="out",
    )
)
```

Supported operations:

- `get_out_relations`
- `get_in_relations`
- `get_neighbors`
- `search_entity_ids_by_name`
- `entity_name_exists`
- `follow_path`
- `extract_subgraph`

All public IDs are strings and all returned lists are deterministic.

## WebQSP Dataset-Specific KG

`datasets/WebQSP_KG` is a built subKG for the processed WebQSP corpus in this repo.

It is constructed as the deduplicated union of `subgraph.tuples` across:

- `ref_KG_projects/EPR-KGQA/data/dataset/WebQSP/train_simple.jsonl`
- `ref_KG_projects/EPR-KGQA/data/dataset/WebQSP/dev_simple.jsonl`
- `ref_KG_projects/EPR-KGQA/data/dataset/WebQSP/test_simple.jsonl`

Build or rebuild it with:

```bash
PYTHONPATH=src python datasets/WebQSP_KG/build_webqsp_subkg.py
```

The builder also enriches `datasets/WebQSP_KG/entities.tsv` with human-readable labels when they can be recovered from the original WebQSP JSON and writes `datasets/WebQSP_KG/entity_name_to_ids.tsv` for reverse name-to-id lookup.

## LLM Frontend

`src/llm_frontend` is a thin iterative planner loop that uses the current backend one hop at a time.

The frontend now resolves the initial entity from a human-readable entity name through backend metadata before the first KG hop.
After that initial resolution step, the loop stays in entity-id space: the frontier contains ids, each `KG_QUERY` traverses ids, and later KG hops do not use name-to-id lookup again.

The runner resolves vendor presets from `src/llm_frontend/llm_config.py`.
You can pass `--VENDOR` and `--API_KEY` directly, or set only `LLM_API_KEY`.

```bash
export LLM_API_KEY=...
```

Run one manual question:

```bash
PYTHONPATH=src python -m llm_frontend.run_webqsp_llm \
  --kg-path datasets/WebQSP_KG \
  --question "what is the name of justin bieber brother" \
  --VENDOR tamu \
  --API_KEY ... \
  --model protected.gemini-2.0-flash-lite \
  --max-steps 4 \
  --output artifacts/llm_frontend_single.jsonl
```

Run one WebQSP example:

```bash
PYTHONPATH=src python -m llm_frontend.run_webqsp_llm \
  --kg-path datasets/WebQSP_KG \
  --webqsp datasets/WebQSP \
  --split test \
  --limit 1 \
  --VENDOR tamu \
  --API_KEY ... \
  --model protected.gemini-2.0-flash-lite \
  --max-steps 4 \
  --output artifacts/webqsp_one_llm_trace.jsonl
```

Run a small WebQSP subset:

```bash
PYTHONPATH=src python -m llm_frontend.run_webqsp_llm \
  --kg-path datasets/WebQSP_KG \
  --webqsp datasets/WebQSP \
  --split test \
  --limit 5 \
  --VENDOR tamu \
  --API_KEY ... \
  --model protected.gemini-2.0-flash-lite \
  --max-steps 6 \
  --output artifacts/webqsp_test_llm_traces.jsonl
```

The frontend accepts `--triples` as a backward-compatible alias for `--kg-path`, but `--kg-path` is the current name.

## Verification

There is no in-repo pytest suite in the current worktree.

The actively used checks for the current code are:

```bash
ruff check .
ruff format --check .
PYTHONPATH=src python -m mypy src
```

## Future Cached Variant

The current backend is intentionally kept uncached so a later cached layer can wrap the same interface without changing baseline semantics.
