# Detailed Setup

The [Quickstart](../README.md#quickstart) covers the Docker path, which is the
recommended way to bring up a SPARQL backend. This document covers the manual
host install, which you only need if you are not using Docker.

## Contents

- [Python environment](#python-environment)
- [SPARQL backend: Virtuoso (manual install)](#sparql-backend-virtuoso-manual-install)
- [SPARQL backend: Oxigraph](#sparql-backend-oxigraph)
- [Running ToG directly](#running-tog-directly)
- [Evaluating a prediction file](#evaluating-a-prediction-file)
- [Useful checks](#useful-checks)

## Python environment

```bash
conda create -n KG_cash python=3.10 -y
conda activate KG_cash

cd src/ToG-cache
pip install -r requirements.txt
pip install "openai==0.28.1"
```

`KG_cash` is the environment name the ToG runners look for by default when
launching subprocesses; override it with `CONDA_ENV` or `--conda-env`.

The `openai==0.28.1` pin is intentional: the vendored ToG code uses the older
`openai.ChatCompletion.create(...)` API.

Configuration is read from `.env` at the repository root. Start from the
template, which documents every variable the runners consult:

```bash
cp .env.example .env
```

At minimum set `LLM_API_KEY` (one key, for whichever vendor you pass to
`--vendor`) and, for RoG runs, `HF_TOKEN`. Values already exported in your
shell take precedence over the file.

## SPARQL backend: Virtuoso (manual install)

### 1. Check the Freebase data file

The setup expects the filtered Freebase dump to exist here:

```bash
ls -lh src/ToG-cache/Freebase/WebQSP_FilterFreebase
```

If it is missing, create or restore it before loading Virtuoso. The runtime
queries the SPARQL endpoint, never the raw file.

### 2. Install and start Virtuoso

```bash
sudo apt update
sudo apt install virtuoso-opensource-7 -y
sudo systemctl start virtuoso-opensource-7
sudo systemctl status virtuoso-opensource-7
```

The endpoint is `http://localhost:8890/sparql`, configured in
`src/ToG-cache/ToG/freebase_func.py`.

### 3. Allow Virtuoso to read the Freebase directory

Virtuoso only loads files from allowed directories. Edit the config:

```bash
sudo nano /etc/virtuoso-opensource-7/virtuoso.ini
```

Find `DirsAllowed` and add the project Freebase directory:

```text
<your_path>/KG_cash/src/ToG-cache/Freebase
```

Then restart:

```bash
sudo systemctl restart virtuoso-opensource-7
```

### 4. Load Freebase

```bash
isql-vt 1111 dba dba    # or: isql 1111 dba dba
```

At the `SQL>` prompt:

```sql
SQL> ld_dir('<your_path>/KG_cash/src/ToG-cache/Freebase', 'WebQSP_FilterFreebase', 'http://freebase.com');
SQL> rdf_loader_run();
SQL> exit;
```

`rdf_loader_run()` takes roughly 85 seconds on the WebQSP subset.

### 5. Verify

```bash
curl -G "http://localhost:8890/sparql" \
  --data-urlencode "query=PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?p ?o WHERE { ns:m.02mjmr ?p ?o } LIMIT 5" \
  --data-urlencode "format=json"
```

A correct load returns 5 bindings.

### Stopping and restarting

```bash
sudo systemctl stop virtuoso-opensource-7
sudo systemctl start virtuoso-opensource-7
```

## SPARQL backend: Oxigraph

Oxigraph serves the same Freebase KG and is interchangeable with Virtuoso.
Point clients at it with:

```bash
export SPARQL_ENDPOINT=http://localhost:7878/query
```

or pass `--engine oxigraph` to the experiment runners. See
[`docker/oxigraph/README.md`](../docker/oxigraph/README.md) for loading
instructions.

One difference matters: Oxigraph rejects prefixed names containing two dots, so
query templates use absolute IRIs rather than `ns:` prefixes.

## Running ToG directly

The experiment runners in `scripts/` are the usual entry point, but the
vendored ToG runtime can be driven directly:

```bash
cd src/ToG-cache/ToG

python main_freebase.py \
  --dataset webqsp \
  --test-limit 10 \
  --max_length 256 \
  --width 3 \
  --depth 3 \
  --remove_unnecessary_rel True \
  --LLM_type gpt-4o \
  --opeani_api_keys <your-api-key> \
  --num_retain_entity 5 \
  --prune_tools llm
```

Notes:

- The API key flag really is spelled `--opeani_api_keys` — that is the upstream
  ToG flag name.
- `main_freebase.py` exposes no temperature flag; requests use the provider
  default.
- `--test-limit 10` runs the first 10 samples, or the whole dataset if smaller.
- **Output is appended, not overwritten.** To force a clean run:
  `rm -f src/ToG-cache/output/ToG_webqsp.jsonl`
- Predictions default to `src/ToG-cache/output/ToG_webqsp.jsonl`.

## Evaluating a prediction file

The ToG run writes predictions but does not print accuracy. Score separately:

```bash
cd src/ToG-cache/eval

python eval.py \
  --dataset webqsp \
  --output_file ../output/ToG_webqsp.jsonl \
  --constraints_refuse True
```

The evaluator accepts both `.json` and `.jsonl`. Console output:

```text
Exact Match: <score>
right: <num_right>, error: <num_error>
```

It also writes `src/ToG-cache/eval/ToG_webqsp_results.json`.

## Useful checks

Count output rows:

```bash
python -c "p='src/ToG-cache/output/ToG_webqsp.jsonl'; print(sum(1 for l in open(p) if l.strip()))"
```

Count unique questions (catches duplicate appends):

```bash
python -c "
import json
p='src/ToG-cache/output/ToG_webqsp.jsonl'
rows=[json.loads(l) for l in open(p) if l.strip()]
print(len(rows), len({r['question'] for r in rows}))"
```

Compile-check edited files:

```bash
python -m py_compile \
  src/ToG-cache/ToG/main_freebase.py \
  src/ToG-cache/ToG/utils.py \
  src/ToG-cache/ToG/freebase_func.py \
  src/ToG-cache/eval/eval.py \
  src/ToG-cache/eval/utils.py
```
