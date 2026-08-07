# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**KG_cash** is a knowledge graph question-answering (KGQA) system. It answers natural-language questions by iteratively traversing a knowledge graph under LLM guidance. The name reflects the goal of adding caching optimizations on top of the baseline.

Two main components:
- **KG Backend** (`src/kg_backend`) — loads KG triples into immutable in-memory indexes and executes typed queries
- **LLM Frontend** (`src/llm_frontend`) — drives an iterative planner loop where an LLM decides which KG hops to follow until it can produce a final answer

## Environment Setup

```bash
conda env create -f environment.yml
conda activate kg-cache
export PYTHONPATH=src:${PYTHONPATH}
```

Python 3.11. Key deps: `httpx`, `pydantic`, `pyarrow`, `ruff`, `mypy`, `pytest`.

API keys go in `.env` (already gitignored):
```
LLM_API_KEY=...
TAMU_API_KEY=...
GOOGLE_API_KEY=...
```

## Common Commands

**Type checking and linting:**
```bash
PYTHONPATH=src python -m mypy src
ruff check .
ruff format --check .
```

**Test LLM API connectivity:**
```bash
PYTHONPATH=src python scripts/model_test.py
```

**Run a single ad-hoc question:**
```bash
PYTHONPATH=src python -m llm_frontend.run_webqsp_llm \
  --kg-path datasets/WebQSP_KG \
  --question "what is the name of justin bieber brother" \
  --vendor tamu \
  --max-steps 4 \
  --output artifacts/trace.jsonl
```

**Run WebQSP dataset eval:**
```bash
PYTHONPATH=src python -m llm_frontend.run_webqsp_llm \
  --kg-path datasets/WebQSP_KG \
  --webqsp datasets/WebQSP \
  --split test \
  --limit 5 \
  --vendor tamu \
  --max-steps 6 \
  --output artifacts/webqsp_traces.jsonl
```

**Full eval pipeline (run + score):**
```bash
PYTHONPATH=src python scripts/run_webqsp_eval.py \
  --kg-path datasets/WebQSP_KG \
  --webqsp datasets/WebQSP/data/WebQSP.test.json \
  --predictions-jsonl artifacts/preds.jsonl \
  --eval-output artifacts/metrics.json \
  --vendor tamu --max-steps 6
```

Supported `--vendor` values: `tamu`, `google`, `openai`.

## Architecture

### KG Backend (`src/kg_backend`)

The backend is a pure in-memory graph engine with no external dependencies.

- **`loader.py`** — reads triples and optional entity/relation metadata from TSV or Parquet files, produces raw lists
- **`index.py`** — builds immutable `AdjacencyIndex` from loaded triples: entity/relation ID maps, frozen outgoing and incoming adjacency lists
- **`name_lookup.py`** — case-folded exact-match entity name index (`EntityNameIndex`)
- **`backend.py`** — `UncachedKGBackend` implements the `KGBackend` protocol; dispatches typed query objects via `execute()`; all methods are stateless reads against the frozen index
- **`types.py`** — frozen Pydantic models for every query and result type (`GetNeighborsQuery`, `FollowPathQuery`, `ExtractSubgraphQuery`, etc.)

KG input format: `triples.{tsv,parquet}` with columns `head`, `relation`, `tail`. Optional `entities.tsv/parquet` and `relations.tsv/parquet` with `id` and `label` columns.

### LLM Frontend (`src/llm_frontend`)

The frontend drives an iterative reasoning loop over the KG.

**Control flow:**
1. `run_webqsp_llm.py` — CLI entry point; loads KG and dataset, calls `IterativeKGController`
2. `controller.py` (`IterativeKGController`) — outer loop: calls planner, dispatches KG queries, detects loops (repeated frontier signatures / query counts), collects `LLMRunTrace`
3. `planner.py` (`LLMPlanner`) — builds prompt, calls LLM, parses JSON action (`INITIAL_ENTITY`, `KG_QUERY`, or `FINAL_ANSWER`); has compact fallback prompt on parse failure
4. `backend_adapter.py` (`KGBackendAdapter`) — wraps `UncachedKGBackend`; resolves entity names to IDs (`resolve_initial_frontier`), describes available relations at current frontier (`describe_frontier`), executes typed KG queries
5. `memory.py` (`PlannerMemory`) — tracks action history, failed initial entities, and repeated states
6. `llm_client.py` (`LLMChatClient`) — OpenAI-compatible HTTP client via `httpx`; supports all three vendors
7. `llm_config.py` — vendor presets (`LLMPresetConfig`) and resolved runtime config; reads API keys from CLI args or env vars
8. `config.py` (`LLMFrontendConfig`) — frozen dataclass with tuning parameters: `max_steps`, `temperature`, `initial_entity_search_limit`, relation/frontier/history limits
9. `prompts.py` — prompt builders: system prompt defines allowed JSON schema; user prompt encodes current frontier + history
10. `schemas.py` — data models for actions, observations, and `LLMRunTrace`
11. `trace.py` — serializes traces to JSONL; `summarize_traces()` aggregates metrics

### Data

- `datasets/WebQSP/` — WebQSP Q&A benchmark (JSON splits)
- `datasets/WebQSP_KG/` — subgraph of Freebase built for WebQSP questions; `build_webqsp_subkg.py` constructs it
- `datasets/complex-web-questions-dataset/` — git submodule
- `ref_KG_projects/` — reference KGQA implementations as git submodules (EPR-KGQA, KG-R1, SubgraphRetrievalKBQA)

## Key Design Decisions

- **Uncached baseline** — `UncachedKGBackend` has no memoization; the `KGBackend` protocol exists so caching wrappers can be added transparently
- **Immutable indexes** — adjacency structures are frozen at load time; no mutation after construction
- **Two-phase entity resolution** — initial entity uses name-to-ID lookup; all subsequent hops stay in ID space
- **Typed query dispatch** — `backend.execute()` accepts a discriminated union of query types; adding a new query type requires adding a model to `types.py` and a branch in `execute()`
- **JSON-structured LLM output** — planner emits one of three action types as JSON; parse failures trigger a compact fallback prompt before aborting
