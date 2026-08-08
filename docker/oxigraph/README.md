# Oxigraph (alternative Freebase SPARQL endpoint)

Runs [Oxigraph](https://github.com/oxigraph/oxigraph) as a drop-in alternative to
Virtuoso for serving the same Freebase KG. The point: every KG consumer in this
repo goes through one SPARQL endpoint URL, so proving the cache is
backend-agnostic only requires swapping that URL.

## Swapping backends

Everything reads the endpoint from the `SPARQL_ENDPOINT` env var
(`src/ToG-cache/ToG/freebase_func.py`), defaulting to Virtuoso:

```bash
# use Virtuoso (default)
unset SPARQL_ENDPOINT

# use Oxigraph
export SPARQL_ENDPOINT=http://localhost:7878/query
```

The `scripts/run_tog_*.py` runners additionally understand `KG_BACKEND`
(`virtuoso` | `oxigraph`): they start the right docker compose service, wait for
it, and export `SPARQL_ENDPOINT` for the ToG subprocesses:

```bash
KG_BACKEND=oxigraph python scripts/run_tog_eval.py ...
```

## Data layout

- `datasets/Freebase/` on the host → `/data` (read-only) inside the container.
  Same mount as Virtuoso, so both backends load from the same prepared
  N-triples file (e.g. `FilterFreebase`).
- `oxigraph-db` named Docker volume → `/db`. Persistent RocksDB storage.
  `docker volume rm kg_cash_oxigraph-db` wipes the loaded graph.

## Bring it up

```bash
docker compose up -d oxigraph
```

Health check:
```bash
curl -s 'http://localhost:7878/query?query=ASK%20%7B%7D' -H 'Accept: application/sparql-results+json'
# → {"head":{},"boolean":true}
```

## Load Freebase (first time only)

Bulk loading must run while the server is *stopped* (both open the same RocksDB
at `/db`):

```bash
docker compose stop oxigraph
docker compose run --rm oxigraph load --location /db --file /data/FilterFreebase --format nt --lenient
docker compose up -d oxigraph
```

Notes:
- `--format nt` is needed because `FilterFreebase` has no file extension.
- `--lenient` skips the handful of malformed IRIs/literals in the Freebase dump
  instead of aborting.
- Oxigraph's bulk loader is parallel and handles Wikidata-scale inputs; expect
  a few hours for the ~125 GB filtered dump depending on disk speed.
- Triples land in the default graph, which is what the un-`GRAPH`ed queries in
  `freebase_func.py` expect. (Virtuoso puts them in the `http://freebase.com`
  named graph but also searches all graphs by default, so both backends answer
  the same queries identically.)


