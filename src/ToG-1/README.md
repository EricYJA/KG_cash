# ToG-1 Local Test Guide

`src/ToG-1` is a local copy of `datasets/ToG` wired to use the same runtime setup as `src/ToG-cache`:

- Freebase backend: `http://localhost:8890/sparql`
- Default LLM vendor: `tamu`
- API key source for the default vendor: `LLM_API_KEY`
- Default output: `src/ToG-1/output/ToG_<dataset>.jsonl`

## 1. Activate The Environment

Run commands from the repo root unless a step says otherwise.

```bash
cd /home/ccyuan/Project/KG_cash
source /home/ccyuan/miniconda3/etc/profile.d/conda.sh
conda activate kg_cache
```

## 2. Start The KG Backend

Start the local Virtuoso service:

```bash
sudo systemctl start virtuoso-opensource-7
sudo systemctl status virtuoso-opensource-7
```

Confirm the SPARQL endpoint responds:

```bash
curl -G "http://localhost:8890/sparql" \
  --data-urlencode "query=PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?p ?o WHERE { ns:m.02mjmr ?p ?o } LIMIT 5" \
  --data-urlencode "format=json"
```

If the JSON response has bindings, the backend is ready.

### If Freebase Is Not Loaded Yet

`ToG-1` shares the same backend as `ToG-cache`; it does not need its own copy of the large Freebase file. Load the existing filtered file from `src/ToG-cache/Freebase`:

```bash
isql-vt 1111 dba dba
```

Inside the `SQL>` prompt:

```sql
ld_dir('/home/ccyuan/Project/KG_cash/src/ToG-cache/Freebase', 'WebQSP_FilterFreebase', 'http://freebase.com');
rdf_loader_run();
exit;
```

Then rerun the `curl` check above.

## 3. Add The API Key

For the default `tamu` vendor, set one environment variable:

```bash
export LLM_API_KEY='your-api-key-here'
```

Quickly verify that the code can resolve it without calling the model:

```bash
cd /home/ccyuan/Project/KG_cash/src/ToG-1/ToG
python - <<'PY'
from llm_config import resolve_llm_config
config = resolve_llm_config(vendor='tamu')
print(config.vendor, config.model, config.base_url, bool(config.api_key))
PY
```

Expected shape:

```text
tamu protected.gpt-5.2 https://chat-api.tamu.ai/openai True
```

To use OpenAI directly instead, set `OPENAI_API_KEY` and run with `--vendor openai`:

```bash
export OPENAI_API_KEY='your-openai-api-key-here'
python -c "import os; print(bool(os.environ.get('OPENAI_API_KEY')))"
```

The check should print `True`. If it prints `False`, the key is not available in the current shell.

## 4. Run No-Cost Smoke Checks

These checks do not call the LLM.

```bash
cd /home/ccyuan/Project/KG_cash
python -m compileall -q src/ToG-1
```

```bash
cd /home/ccyuan/Project/KG_cash/src/ToG-1/ToG
python main_freebase.py --dataset webqsp --test-limit 0
```

Probe the shared Freebase backend through the ToG-1 code:

```bash
cd /home/ccyuan/Project/KG_cash/src/ToG-1/ToG
python - <<'PY'
from freebase_func import execurte_sparql, sparql_head_relations
rows = execurte_sparql((sparql_head_relations % 'm.02mjmr') + ' LIMIT 1')
print(len(rows))
PY
```

Expected output:

```text
1
```

## 5. Run One Simple End-To-End Test

This calls the KG backend and the LLM once or more. The OpenAI command below requires `OPENAI_API_KEY`.

```bash
cd /home/ccyuan/Project/KG_cash/src/ToG-1/ToG
rm -f ../output/ToG_webqsp_smoke.jsonl
python main_freebase.py \
  --vendor openai \
  --LLM_type gpt-4o-mini \
  --dataset webqsp \
  --test-limit 1 \
  --width 1 \
  --depth 1 \
  --max_length 256 \
  --num_retain_entity 2 \
  --prune_tools llm \
  --output-file ../output/ToG_webqsp_smoke.jsonl
```

Check the output file:

```bash
wc -l ../output/ToG_webqsp_smoke.jsonl
tail -n 1 ../output/ToG_webqsp_smoke.jsonl
```

Expected: `wc -l` prints `1`, and the JSONL row contains `question`, `results`, and `reasoning_chains`.

## 6. Optional Evaluation

```bash
cd /home/ccyuan/Project/KG_cash/src/ToG-1/eval
python eval.py \
  --dataset webqsp \
  --output_file ../output/ToG_webqsp_smoke.jsonl \
  --constraints_refuse True
```

The evaluator prints exact match stats and writes `ToG_webqsp_results.json` in `src/ToG-1/eval`.
