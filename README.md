# KG Cash

This repo contains a local ToG-based knowledge graph QA pipeline under
`src/ToG-cache`. It is based on the upstream ToG project, with local changes for
cached Freebase/WebQSP runs, small test runs, configurable output paths, and
JSONL evaluation.

## Credit

This work builds on the ToG implementation from:

- [GasolSun36/ToG](https://github.com/GasolSun36/ToG)

Please refer to the upstream repository for the original ToG code, paper
context, and citation information.

## Repository Layout

- `src/ToG-cache/`: runnable ToG code and local Freebase/WebQSP resources.
- `src/ToG-cache/ToG/`: main ToG runtime.
- `src/ToG-cache/eval/`: exact-match evaluator.
- `src/ToG-cache/output/`: default output directory for generated predictions.
- `src/ToG-cache/Freebase/WebQSP_FilterFreebase`: filtered Freebase triples used by the current WebQSP setup.

## Setup

### 1. Check The Freebase Data File

The current setup expects the filtered Freebase file to exist here:

```bash
ls -lh KG_cash/src/ToG-cache/Freebase/WebQSP_FilterFreebase
```

If this file is missing, create or restore it before loading Virtuoso. The
runtime queries the local Virtuoso SPARQL endpoint, not the raw file directly.

### 2. Create The Python Environment

Create and activate a conda environment:

```bash
conda create -n kg_cache python=3.10 -y
conda activate kg_cache
```

Install ToG dependencies:

```bash
cd KG_cash/src/ToG-cache
pip install -r requirements.txt
pip install "openai==0.28.1"
```

The `openai==0.28.1` pin is intentional because this ToG code uses the older
`openai.ChatCompletion.create(...)` API.

### 3. Install And Start Virtuoso

Install the open-source Virtuoso package:

```bash
sudo apt update
sudo apt install virtuoso-opensource-7 -y
```

Start the service:

```bash
sudo systemctl start virtuoso-opensource-7
sudo systemctl status virtuoso-opensource-7
```

The ToG Freebase client is configured to query:

```text
http://localhost:8890/sparql
```

That setting is in `src/ToG-cache/ToG/freebase_func.py`.

### 4. Allow Virtuoso To Read The Freebase Directory

Virtuoso only loads files from allowed directories. Edit the Virtuoso config and
add the project Freebase directory to `DirsAllowed`.

Common config location:

```bash
sudo nano /etc/virtuoso-opensource-7/virtuoso.ini
```

Look for `DirsAllowed` and include:

```text
<your_path>/KG_cash/src/ToG-cache/Freebase
```

Then restart Virtuoso:

```bash
sudo systemctl restart virtuoso-opensource-7
```

### 5. Load Freebase Into Virtuoso

Open the Virtuoso SQL shell:

```bash
isql-vt 1111 dba dba
```

If your install uses a different binary name, try:

```bash
isql 1111 dba dba
```

Inside the `SQL>` prompt, run:

```bash
SQL> ld_dir('<your_path>/KG_cash/src/ToG-cache/Freebase', 'WebQSP_FilterFreebase', 'http://freebase.com');
SQL> rdf_loader_run();
```

Expected output is similar to:

```bash
SQL> ld_dir('<your_path>/KG_cash/src/ToG-cache/Freebase', 'WebQSP_FilterFreebase', 'http://freebase.com');
Done. -- 1 msec.

SQL> rdf_loader_run();
Done. -- 84972 msec.
```

Exit the SQL shell:

```bash
SQL> exit;
```

### 6. Verify The SPARQL Endpoint

Run a small query:

```bash
curl -G "http://localhost:8890/sparql" \
  --data-urlencode "query=PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?p ?o WHERE { ns:m.02mjmr ?p ?o } LIMIT 5" \
  --data-urlencode "format=json"
```

If the data is loaded correctly, the JSON response should contain 5 bindings.

## Run ToG

Run from the ToG source directory:

```bash
cd KG_cash/src/ToG-cache/ToG
```

Small WebQSP smoke test:

```bash
python main_freebase.py \
  --dataset webqsp \
  --test-limit 10 \
  --max_length 256 \
  --temperature_exploration 0.4 \
  --temperature_reasoning 0 \
  --width 3 \
  --depth 3 \
  --remove_unnecessary_rel True \
  --LLM_type gpt-4o \
  --opeani_api_keys <your-api-key> \
  --num_retain_entity 5 \
  --prune_tools llm
```

Notes:

- The API key flag is intentionally spelled `--opeani_api_keys` because that is
  the flag name used by the ToG code.
- `--test-limit 10` runs the first 10 samples, or the full dataset if it has
  fewer than 10 samples.
- Output is appended, not overwritten.
- By default, predictions are written to:

```bash
KG_cash/src/ToG-cache/output/ToG_webqsp.jsonl
```

To force a clean run, remove the previous output first:

```bash
rm -f KG_cash/src/ToG-cache/output/ToG_webqsp.jsonl
```

## Evaluate

The main ToG run creates prediction files. It does not print accuracy. Use the
evaluator separately.

The evaluator accepts both `.json` and `.jsonl` output files:

```bash
cd KG_cash/src/ToG-cache/eval

python eval.py \
  --dataset webqsp \
  --output_file ../output/ToG_webqsp.jsonl \
  --constraints_refuse True
```

Expected console output:

```text
Exact Match: <score>
right: <num_right>, error: <num_error>
```

It also writes a summary file in the eval directory:

```text
KG_cash/src/ToG-cache/eval/ToG_webqsp_results.json
```

## Useful Checks

Count output rows:

```bash
python -c "import json; p='KG_cash/src/ToG-cache/output/ToG_webqsp.jsonl'; print(sum(1 for line in open(p) if line.strip()))"
```

Check unique output questions:

```bash
python -c "import json; p='KG_cash/src/ToG-cache/output/ToG_webqsp.jsonl'; rows=[json.loads(l) for l in open(p) if l.strip()]; print(len(rows), len({r['question'] for r in rows}))"
```

Compile-check the edited Python files:

```bash
cd KG_cash
python -m py_compile \
  src/ToG-cache/ToG/main_freebase.py \
  src/ToG-cache/ToG/utils.py \
  src/ToG-cache/ToG/freebase_func.py \
  src/ToG-cache/eval/eval.py \
  src/ToG-cache/eval/utils.py
```

## Stop Virtuoso

When finished:

```bash
sudo systemctl stop virtuoso-opensource-7
```

To start it again later:

```bash
sudo systemctl start virtuoso-opensource-7
```
