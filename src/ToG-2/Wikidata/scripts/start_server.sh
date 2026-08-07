#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${WIKIDATA_DATA_DIR:-/dev/shm/wikidump_inmem/wikidump_20230116}"
HOST_IP="${WIKIDATA_HOST_IP:-localhost}"
START_PORT="${WIKIDATA_START_PORT:-23150}"
NUM_CHUNKS="${WIKIDATA_NUM_CHUNKS:-10}"

cd "$(dirname "$0")/.."
mkdir -p logs
: > server_urls.txt
rm -f server_urls_new.txt

if [ ! -d "$DATA_DIR" ]; then
  echo "Missing Wikidata index directory: $DATA_DIR" >&2
  echo "Set WIKIDATA_DATA_DIR=/path/to/processed/wikidata before running this script." >&2
  exit 1
fi

for ((i=0; i<NUM_CHUNKS; i++)); do
  port=$((START_PORT + i))
  echo "http://${HOST_IP}:${port}" >> server_urls.txt
  python -u simple_wikidata_db/db_deploy/server.py     --data_dir "$DATA_DIR"     --chunk_number "$i"     --port "$port"     --host_ip "$HOST_IP"     > "logs/server_log_${i}.log" 2>&1 &
done

echo "Wikidata server URLs written to $(pwd)/server_urls.txt"
echo "Copy them into ../ToG-2/server_urls.txt before running graph mode:"
echo "  cp $(pwd)/server_urls.txt ../ToG-2/server_urls.txt"
wait
