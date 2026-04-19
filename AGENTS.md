# AGENTS.md

## Project purpose

This repository implements a custom uncached adjacency backend for knowledge-graph retrieval in a RoG-style research pipeline.

This codebase is the baseline backend used before adding a separate caching layer. The goal is to measure the cost of repeated entity/relation retrieval and later compare it against cached variants.

This repository is not a general graph database.

## Core definition: what "uncached" means here

"Uncached" does not mean scanning raw files on every request.

"Uncached" does mean:

- load the local KG once at startup;
- build immutable adjacency indexes once at startup;
- answer each request directly from those baseline indexes;
- do not memoize request results or path-expansion results across calls.

Allowed:

- string-to-integer ID maps built at load time;
- immutable adjacency structures built at load time;
- sorted lists or tuples for deterministic outputs;
- one backend instance reused by the API server.

Not allowed:

- `functools.cache`
- `functools.lru_cache`
- manual memoization dictionaries for query results
- Redis
- disk cache
- precomputed multi-hop closure tables
- Neo4j
- SPARQL endpoints
- Cypher as the public query language
- NetworkX as the main retrieval engine

If a later task asks for caching, implement it as a separate wrapper or separate backend class. Do not change the semantics of the uncached baseline.

## Repository layout

Expected layout:

- `src/kg_backend/`
  - `types.py` for query/result schemas
  - `errors.py` for typed exceptions
  - `loader.py` for TSV/Parquet loading
  - `index.py` for adjacency-index construction
  - `backend.py` for the main backend interface and uncached implementation
  - `api.py` for FastAPI routes
  - `main.py` for local startup helpers
- `src/kg_backend_tests/`
  - unit tests
  - API tests
  - small fixtures
- `README.md`
  - architecture, usage, and examples

Keep files small and cohesive.

## Public backend contract

The public backend should expose a small typed interface centered on retrieval operations needed by RoG-style systems:

- `get_out_relations(entity_id)`
- `get_in_relations(entity_id)`
- `get_neighbors(entity_id, relation_id, direction)`
- `follow_path(start_entities, path)`
- `extract_subgraph(seed_entities, max_hops, allowed_relations, direction, max_edges)`
- `execute(query)`

Public APIs should use external string IDs.
Internal implementations may map strings to integer offsets.

All returned lists must be deterministic and sorted.

## Query language

Do not build a free-form query parser.

Do not build Cypher, SPARQL, or SQL translation.

Use a typed query model instead. The HTTP layer should accept a small JSON schema with an `op` field and strongly typed arguments for the supported operations.

## Data format

Support at least:

1. `triples.tsv`
   - columns: `head`, `relation`, `tail`

2. `triples.parquet`
   - columns: `head`, `relation`, `tail`

Optional metadata files:

- `entities.tsv` or `entities.parquet`
  - columns: `id`, `label`

- `relations.tsv` or `relations.parquet`
  - columns: `id`, `label`

Code should validate required columns and raise clear errors on invalid input.

## Engineering rules

Prefer clarity over cleverness.

Prefer explicit adjacency structures over hidden framework behavior.

Use Python type hints throughout.

Public functions and classes need docstrings.

Add comments only where the intent is not obvious from the code.

Avoid hidden global mutable state.

Avoid heavyweight dependencies unless they are clearly justified.

Do not add a database, ORM, or message queue.

Do not add features that are not needed for the uncached baseline.

## Determinism and testing

Deterministic behavior matters.

- sort entity outputs
- sort relation outputs
- document duplicate-triple handling
- make tests robust to ordering because the implementation itself should already be deterministic

Add tests for normal cases and edge cases.

At minimum, test:

- TSV loading
- Parquet loading
- duplicate triples
- missing IDs
- empty answers
- deterministic ordering
- path following
- subgraph extraction
- HTTP API behavior

## Benchmarking rules

Benchmark scripts should separate:

- load time
- index-build time
- per-query latency

Do not hide startup cost inside query latency.

Do not add caches to make the benchmark look better.

If a benchmark uses repeated queries, it should still call the uncached backend directly without memoization.

## Commands

Assume these commands should work once the repo is set up:

- install:
  - `conda env create -f environment.yml`
  - `conda activate kg-cache`
  - `export PYTHONPATH=src:${PYTHONPATH}`

- tests:
  - `PYTHONPATH=src pytest -q`

- lint:
  - `ruff check .`

- format check:
  - `ruff format --check .`

- type check:
  - `PYTHONPATH=src python -m mypy src`

- run local API:
  - `PYTHONPATH=src uvicorn kg_backend.api:app --reload`

If you add or change commands, update this file and the README in the same change.

## Definition of done

A task is done only when all of the following are true:

1. the code matches the uncached-baseline definition in this file;
2. the public backend interface is typed and documented;
3. tests were added or updated;
4. lint, format check, and type check pass;
5. README examples match the actual implementation;
6. no hidden caching or memoization was introduced.

## Future-proofing

Design the uncached backend so a later task can add:

- `CachedKGBackend`
- a cache-decorator around the backend interface
- alternative storage backends

But do not implement those in baseline tasks unless explicitly requested.
