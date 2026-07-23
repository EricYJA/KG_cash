# Virtuoso (Freebase SPARQL endpoint)

Runs [OpenLink Virtuoso 7](https://hub.docker.com/r/openlink/virtuoso-opensource-7) so `src/ToG-cache/ToG/freebase_func.py` can talk to `http://localhost:8890/sparql`.

## Data layout

- `datasets/Freebase/` on the host → `/data` (read-only) inside the container. Put your prepared Freebase N-triples file(s) here (e.g. `FilterFreebase`).
- `virtuoso-db` named Docker volume → `/database` inside the container. Persistent DB storage. Deleting this volume wipes the loaded graph.

## Bring it up

```bash
docker compose up -d virtuoso
docker compose logs -f virtuoso
```

Startup takes ~5-15 s on an empty DB. Once you see `Server online at 1111`, isql and the SPARQL HTTP endpoint are ready.

Health check:
```bash
curl -s 'http://localhost:8890/sparql?query=ASK%20%7B%7D&format=json'
# → {"head":{"link":[]},"boolean":true}
```

## Load Freebase (first time only)

1. Download and prepare the Freebase dump per `src/ToG-cache/Freebase/README.md` (raw 400 GB → filtered ~125 GB `FilterFreebase` file). Put the result in `datasets/Freebase/` on the host.
2. Open an isql shell inside the container:
   ```bash
   docker compose exec virtuoso isql 1111 dba dba
   ```
3. Register and run the loader:
   ```sql
   SQL> ld_dir('/data', 'FilterFreebase', 'http://freebase.com');
   SQL> rdf_loader_run();
   SQL> checkpoint;
   ```
   Loading takes several hours to a day depending on disk speed. The `checkpoint` at the end forces a durable commit.
4. Sanity-check the load from outside:
   ```bash
   curl -s 'http://localhost:8890/sparql?format=json&query=SELECT+%28COUNT%28*%29+AS+%3Fn%29+WHERE+%7B+%3Fs+%3Fp+%3Fo+%7D+LIMIT+1'
   ```

## Tuning

Env vars in `docker-compose.yml` control the important knobs:

| Var | Default | What it does |
|---|---|---|
| `VIRTUOSO_BUFFERS` | 680000 | 8 KB pages Virtuoso keeps in RAM. 680k ≈ 5 GB. Bump to ~1.3M (~10 GB) if you have RAM; Freebase queries speed up substantially. Rule of thumb: 65% of free RAM ÷ 8 KB. |
| `VIRTUOSO_MAX_DIRTY_BUFFERS` | 500000 | Usually 3/4 of `NumberOfBuffers`. |
| `VIRTUOSO_DBA_PASSWORD` | `dba` | Change for anything but local dev. |

Override in your shell or `.env` file:
```bash
export VIRTUOSO_BUFFERS=1300000
export VIRTUOSO_MAX_DIRTY_BUFFERS=1000000
docker compose up -d virtuoso
```

## Endpoints

- **SPARQL HTTP**: http://localhost:8890/sparql — what `freebase_func.py` hits.
- **Web console**: http://localhost:8890/conductor — DBA UI (login dba/dba by default).
- **ISQL TCP**: localhost:1111 — for `isql`-based bulk load.

## Notes on `freebase_func.py`

The existing code has `SPARQLPATH = "http://localhost:8890/sparql"` hard-coded (`src/ToG-cache/ToG/freebase_func.py:4`). That matches this compose port mapping, so no code change needed when running the ToG-cache scripts on the host. If you ever run them from *inside* another container in this compose file, use `http://virtuoso:8890/sparql` (the compose network hostname).

## Stopping / resetting

- Stop: `docker compose stop virtuoso`
- Remove but keep data: `docker compose rm -f virtuoso`
- **Nuke the loaded graph**: `docker compose down -v` (removes named volumes — you'll need to re-run the load procedure)
