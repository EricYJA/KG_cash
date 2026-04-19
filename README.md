# KG Backend Baseline

This repository now includes a local-file-backed uncached adjacency backend under `src/kg_backend`.

In this repo, "uncached" means:

- triples are loaded from local files once at startup;
- immutable adjacency indexes are built once at startup;
- every query is answered directly from those baseline indexes;
- no query-result memoization or path-expansion cache is used.

It does not mean rescanning raw files per request.

## Setup

The repo expects a Python 3.11 environment. The existing shared Conda environment file is:

```bash
conda env create -f environment.yml
conda activate kg-cache
export PYTHONPATH=src:${PYTHONPATH}
```

There is intentionally no editable-install packaging step. Run the backend directly from the repo root with `PYTHONPATH=src`.

## Data Format

Supported graph inputs:

- `triples.tsv`
- `triples.parquet`

Both formats must contain:

```text
head  relation  tail
```

Optional metadata files are also supported:

- `entities.tsv` or `entities.parquet` with columns `id`, `label`
- `relations.tsv` or `relations.parquet` with columns `id`, `label`

If metadata files are present, they must cover every entity or relation referenced by the triples.

Duplicate triples are accepted at load time and deduplicated when the immutable adjacency index is built.

## Python API

```python
from kg_backend.backend import UncachedKGBackend
from kg_backend.types import PathStep

backend = UncachedKGBackend.from_data_path("src/kg_backend_tests/fixtures/tiny_kg.tsv")

backend.get_out_relations("alice")
backend.get_in_relations("bob")
backend.get_neighbors("alice", "likes", direction="out")
backend.follow_path(
    ["alice"],
    [
        PathStep(relation_id="likes", direction="out"),
        PathStep(relation_id="parent", direction="in"),
    ],
)
backend.extract_subgraph(["alice"], max_hops=2, direction="both")
```

You can also dispatch typed query models through `execute(query)`.

## HTTP API

Start the API server:

```bash
export KG_BACKEND_DATA_PATH=src/kg_backend_tests/fixtures/tiny_kg.tsv
PYTHONPATH=src uvicorn kg_backend.api:app --reload
```

Routes:

- `GET /health`
- `GET /stats`
- `POST /query`

Example request:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "content-type: application/json" \
  -d '{
    "op": "follow_path",
    "start_entities": ["alice"],
    "path": [
      {"relation_id": "likes", "direction": "out"},
      {"relation_id": "parent", "direction": "in"}
    ]
  }'
```

Supported `op` values:

- `get_out_relations`
- `get_in_relations`
- `get_neighbors`
- `follow_path`
- `extract_subgraph`

## Development Commands

Environment update:

```bash
conda env update -f environment.yml --prune
conda activate kg-cache
export PYTHONPATH=src:${PYTHONPATH}
```

Run tests:

```bash
PYTHONPATH=src pytest -q
```

Lint:

```bash
ruff check .
```

Format check:

```bash
ruff format --check .
```

Type check:

```bash
PYTHONPATH=src python -m mypy src
```

Print backend stats locally:

```bash
PYTHONPATH=src python -m kg_backend.main --data-path src/kg_backend_tests/fixtures/tiny_kg.tsv
```

The test fixture graph used in examples lives under `src/kg_backend_tests/fixtures/`.

Build the dataset-specific WebQSP subKG:

```bash
PYTHONPATH=src python datasets/WebQSP_KG/build_webqsp_subkg.py
PYTHONPATH=src python -m kg_backend.main --data-path datasets/WebQSP_KG
```

## Future Cached Variant

The uncached backend is structured so a later task can add a separate `CachedKGBackend` or a cache-decorator around the same public interface.

That future layer should wrap `KGBackend` and leave the uncached baseline semantics unchanged.
