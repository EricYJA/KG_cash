#!/usr/bin/env bash
set -euo pipefail
: "${API_KEY:?API_KEY is required}"
curl -sS -X POST https://generativelanguage.googleapis.com/v1beta/openai/chat/completions \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  --data-binary @/home/ccyuan/Project/KG_cash/artifacts/google_last_request.json | jq
