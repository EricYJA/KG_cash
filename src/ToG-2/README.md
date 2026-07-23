# ToG-2 Local Test Guide

`src/ToG-2` is a local copy of `datasets/ToG-2` wired for the same runtime style as `src/ToG-cache`:

- LLM default vendor: `tamu`
- Default TAMU key source: `LLM_API_KEY`
- Direct OpenAI mode: `--vendor openai` plus `OPENAI_API_KEY` or `--opeani_api_keys`
- Wikidata backend: XML-RPC URLs in `src/ToG-2/ToG-2/server_urls.txt`
- Lightweight local ranking default: `--embedding_model_name bm25`

ToG-2 is Wikidata/Wikipedia based. It does not use the Freebase `localhost:8890/sparql` backend from ToG-1.

## 1. Activate The Environment

```bash
cd /home/ccyuan/Project/KG_cash
source /home/ccyuan/miniconda3/etc/profile.d/conda.sh
conda activate kg_cache
```

## 2. Start Or Point To The Wikidata Backend

If you already have the ToG-cache Wikidata XML-RPC backend running, put those URLs in ToG-2's URL file:

```bash
cd /home/ccyuan/Project/KG_cash
cp src/ToG-cache/Wikidata/server_urls.txt src/ToG-2/ToG-2/server_urls.txt
```

If you run a single local Wikidata server on the default port, create the file manually:

```bash
printf 'http://localhost:23546\n' > /home/ccyuan/Project/KG_cash/src/ToG-2/ToG-2/server_urls.txt
```

To start the same bundled Wikidata server code used by ToG-cache, use your processed Wikidata index directory:

```bash
cd /home/ccyuan/Project/KG_cash/src/ToG-cache/Wikidata
python simple_wikidata_db/db_deploy/server.py \
  --data_dir /path/to/processed/wikidata \
  --chunk_number 0 \
  --port 23546 \
  --host_ip localhost
```

For the repo's multi-chunk script, first confirm the hard-coded `data_dir` in `src/ToG-cache/Wikidata/scripts/start_server.sh`, then run:

```bash
cd /home/ccyuan/Project/KG_cash/src/ToG-cache/Wikidata
bash scripts/start_server.sh
```

Then copy or write the generated server URLs into `src/ToG-2/ToG-2/server_urls.txt`.

## 3. Verify The Wikidata Backend

```bash
cd /home/ccyuan/Project/KG_cash/src/ToG-2/ToG-2
python - <<'PY'
from client import MultiServerWikidataQueryClient
with open('server_urls.txt') as f:
    urls = [line.split('#', 1)[0].strip() for line in f if line.split('#', 1)[0].strip()]
client = MultiServerWikidataQueryClient(urls)
client.test_connections()
print(client.query_all('qid2label', 'Q30'))
PY
```

Expected: connection testing succeeds and the label query returns a result for `Q30`.


### If You See Connection Refused

This error means no Wikidata XML-RPC server is listening at the URL in `server_urls.txt`:

```text
Failed to connect to http://localhost:23546. Error: [Errno 111] Connection refused
```

Fix it by starting the Wikidata backend or replacing `server_urls.txt` with the URL of a running backend. The current machine did not have `/dev/shm/wikidump_inmem/wikidump_20230116` loaded when this local copy was created, so the bundled start script needs `WIKIDATA_DATA_DIR` if your processed index lives elsewhere.

```bash
cd /home/ccyuan/Project/KG_cash/src/ToG-2/Wikidata
export WIKIDATA_DATA_DIR=/path/to/processed/wikidata
export WIKIDATA_NUM_CHUNKS=1
bash scripts/start_server.sh
```

In another shell, copy the generated URLs and rerun ToG-2:

```bash
cp /home/ccyuan/Project/KG_cash/src/ToG-2/Wikidata/server_urls.txt \
   /home/ccyuan/Project/KG_cash/src/ToG-2/ToG-2/server_urls.txt
```

## 4. Add An API Key

Default TAMU mode:

```bash
export LLM_API_KEY='your-tamu-api-key-here'
```

Direct OpenAI mode:

```bash
export OPENAI_API_KEY='your-openai-api-key-here'
python -c "import os; print(bool(os.environ.get('OPENAI_API_KEY')))"
```

The check should print `True`.

## 5. Run No-Cost Smoke Checks

These do not call the backend or the LLM.

```bash
cd /home/ccyuan/Project/KG_cash
python -m compileall -q src/ToG-2
```

```bash
cd /home/ccyuan/Project/KG_cash/src/ToG-2/ToG-2
python main_tog2.py --dataset webqsp --samples 0 --embedding_model_name bm25
```

## 6. Run A Simple OpenAI LLM Smoke Test

This confirms the ToG-2 CLI and OpenAI key path. It uses `--gpt_only true`, so it does not require the Wikidata backend.

```bash
cd /home/ccyuan/Project/KG_cash/src/ToG-2/ToG-2
rm -f ../output/ToG2_webqsp_openai_smoke.json
python main_tog2.py \
  --vendor openai \
  --LLM_type gpt-4o-mini \
  --LLM_type_rp gpt-4o-mini \
  --dataset webqsp \
  --samples 1 \
  --start 0 \
  --embedding_model_name bm25 \
  --self_consistency false \
  --gpt_only true \
  --max_length 256 \
  --output-file ../output/ToG2_webqsp_openai_smoke.json
```

Check the result:

```bash
python - <<'PY'
import json
p = '../output/ToG2_webqsp_openai_smoke.json'
rows = json.load(open(p))
print(len(rows), rows[0].keys())
PY
```

Expected: one JSON result row.

## 7. Run A Tiny Graph-Using Test

This requires the Wikidata backend and may fetch Wikipedia pages over the network.

```bash
cd /home/ccyuan/Project/KG_cash/src/ToG-2/ToG-2
rm -f ../output/ToG2_webqsp_graph_smoke.json
python main_tog2.py \
  --vendor openai \
  --LLM_type gpt-4o-mini \
  --LLM_type_rp gpt-4o-mini \
  --dataset webqsp \
  --samples 1 \
  --start 0 \
  --embedding_model_name bm25 \
  --self_consistency false \
  --topic_prune false \
  --width 1 \
  --depth 1 \
  --num_sents_for_reasoning 3 \
  --output-file ../output/ToG2_webqsp_graph_smoke.json
```

If this fails before the first LLM call, check `server_urls.txt` and the Wikidata backend process first.
