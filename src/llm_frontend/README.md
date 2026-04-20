# llm_frontend

`llm_frontend` is a small iterative LLM loop on top of `src/kg_backend`.

At each step, the model does one of three things:

1. emits an `INITIAL_ENTITY` to choose a starting entity name or KG id
2. emits a `KG_QUERY` to request one KG hop
3. emits a `FINAL_ANSWER` to stop

The loop is:

`question -> LLM proposes an initial entity name -> frontend resolves it to entity ids -> one KG hop -> updated frontier -> ... -> final answer`

The intermediate KG accesses are recorded in the output trace under `llm_kg_queries`.

Only the initial entity proposal uses name-to-id lookup. After the initial frontier is resolved, the iterative KG loop stays in backend ID space:

- the frontier contains entity ids
- each `KG_QUERY` traverses from entity ids to entity ids
- the backend returns entity ids after each hop
- later KG queries do not call name-to-id lookup again

## Setup

Create or update the environment:

```bash
conda env create -f environment.yml
conda activate kg-cache
export PYTHONPATH=src:${PYTHONPATH}
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate kg-cache
export PYTHONPATH=src:${PYTHONPATH}
```

Set the API key once:

```bash
export LLM_API_KEY=...
```

You can also pass the key directly with `--API_KEY`.

## Main Parameters

Required:

- `--kg-path`: KG directory or triples file loaded by `kg_backend`
- `--question "..."` or `--webqsp datasets/WebQSP`
- `--API_KEY ...` or `LLM_API_KEY`

Recommended for one manual question:

- no extra seed entity is needed; the LLM must choose the initial entity itself

Common optional parameters:

- `--VENDOR`: one of `tamu`, `google`, `openai`
- `--MODEL`: override the vendor default model
- `--BASE_URL`: override the vendor default base URL
- `--initial-entity-search-limit`: maximum number of failed initial-entity attempts before the run returns an empty answer
- `--max-steps`: maximum number of iterative KG steps
- `--temperature`: planner temperature
- `--output`: JSONL file for traces

## Vendor Defaults

If you do not pass `--MODEL` or `--BASE_URL`, the runner uses preset defaults:

- `tamu`: model `protected.gemini-2.0-flash-lite`, base URL `https://chat-api.tamu.ai/api`
- `google`: model `gemini-2.0-flash`, base URL `https://generativelanguage.googleapis.com/v1beta/openai`
- `openai`: model `gpt-4.1-mini`, base URL `https://api.openai.com/v1`

Resolution order:

- API key: `--API_KEY` > `LLM_API_KEY`
- model: `--MODEL` > preset default
- base URL: `--BASE_URL` > preset default

## Run One Simple Iterative Query

This is the simplest manual run. The model will iteratively generate KG accesses until it stops or reaches `--max-steps`.
The initial entity is now resolved from an entity name through `kg_backend` metadata.

Example with Google:

```bash
export PYTHONPATH=src
export LLM_API_KEY=...

python -m llm_frontend.run_webqsp_llm \
  --kg-path datasets/WebQSP_KG \
  --question "what is the name of justin bieber brother" \
  --VENDOR google \
  --max-steps 4 \
  --output artifacts/google_single_trace.jsonl
```

Example with TAMU:

```bash
export PYTHONPATH=src
export LLM_API_KEY=...

python -m llm_frontend.run_webqsp_llm \
  --kg-path datasets/WebQSP_KG \
  --question "what is the name of justin bieber brother" \
  --VENDOR tamu \
  --max-steps 4 \
  --output artifacts/tamu_single_trace.jsonl
```

If you want to override the default model or endpoint:

```bash
python -m llm_frontend.run_webqsp_llm \
  --kg-path datasets/WebQSP_KG \
  --question "what is the name of justin bieber brother" \
  --VENDOR google \
  --MODEL gemini-2.0-flash \
  --BASE_URL https://generativelanguage.googleapis.com/v1beta/openai \
  --API_KEY ... \
  --output artifacts/google_single_trace.jsonl
```

## Output

The command prints a short JSON summary to stdout.

If `--output` is set, it also writes one JSONL trace per example. Each trace includes:

- `question_id`
- `question`
- `llm_initial_entity`
- `llm_initial_frontier`
- `llm_kg_queries`
- `llm_final_answer`
- `num_steps`
- `stop_reason`

Each `llm_kg_queries` item includes only:

- `step_id`
- `relation`
- `direction`
- `resolved_direction`
- `output_frontier`

## Notes

- `--question` and `--webqsp` are mutually exclusive.
- The adapter uses `UncachedKGBackend` directly. There is no query-result caching in this frontend path.
- Name-to-id lookup is only used for the initial entity proposal. Later KG traversal does not need name lookup because it already operates on entity ids.
- If the LLM cannot resolve any initial entity name within `--initial-entity-search-limit` attempts, the run stops with an empty final answer.
