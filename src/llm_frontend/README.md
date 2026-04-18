# llm_frontend

This package implements the LLM-side iterative frontend for the existing KG backend:

`question -> LLM planner -> one KG hop -> backend frontier update -> LLM planner -> ... -> final answer`

It reuses the backend already present in `src/kg_cache_backend/`:

- `KGStore` for local triples
- `KGCache` for backend caching
- `QueryExecutor.execute_step(...)` for one-hop KG execution
- `load_webqsp_examples(...)` for WebQSP loading

## Files

- `config.py`: model and controller settings
- `prompts.py`: iterative protocol prompt
- `schemas.py`: action, observation, and trace dataclasses
- `memory.py`: short planner memory
- `llm_client.py`: minimal LLM HTTP client
- `backend_adapter.py`: thin adapter over the existing backend
- `planner.py`: prompt assembly and strict JSON action parsing
- `controller.py`: iterative LLM/backend loop with loop detection
- `trace.py`: JSONL trace writer
- `run_webqsp_llm.py`: CLI runner

## Prompt Protocol

The planner is restricted to exactly one JSON object per turn:

```json
{"action":"KG_QUERY","relation":"people.person.parents","direction":"forward","entity":null,"frontier":"CURRENT_FRONTIER","reason":"Need the parent entities."}
```

or

```json
{"action":"FINAL_ANSWER","answers":["m.012345"],"reason":"The current frontier already matches the answer."}
```

The prompt includes:

- the natural-language question
- the topic entity when available
- the current frontier size and whether the shown frontier view is complete or truncated
- the currently shown frontier entities
- candidate forward and backward relations from the current frontier
- a compact summary of prior KG queries

The frontend does not store chain-of-thought. It only stores the structured action, the backend observation, and the final answer trace.

If the model emits `FINAL_ANSWER` with an empty `answers` list, the controller falls back to the current frontier and records `stop_reason="final_answer_frontier_fallback"`.

## Environment

Create or update the shared repository environment and activate it:

```bash
conda env create -f environment.yml
conda activate kg_cache
export PYTHONPATH=src:${PYTHONPATH}
```

If `kg_cache` already exists:

```bash
conda env update -f environment.yml --prune
conda activate kg_cache
export PYTHONPATH=src:${PYTHONPATH}
```

Then set the default Gemini configuration:

```bash
export LLM_API_KEY=...
export LLM_PROVIDER=gemini
export LLM_MODEL=gemini-2.5-flash
```

Optional overrides:

```bash
export LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```

Provider notes:

- If `LLM_PROVIDER` is unset, the frontend now defaults to `gemini`
- `LLM_PROVIDER=gemini` uses `LLM_BASE_URL` or defaults to `https://generativelanguage.googleapis.com/v1beta`
- `LLM_PROVIDER=openai_compatible` uses `LLM_BASE_URL` or defaults to `https://api.openai.com/v1`
- For Gemini, `GEMINI_API_KEY` is also accepted as a fallback if `LLM_API_KEY` is unset

`llm_client.py` uses the standard library HTTP client, so no extra dependency is required beyond the shared `kg_cache` environment.

## How To Run

Run one question:

```bash
PYTHONPATH=src python3 -m llm_frontend.run_webqsp_llm \
  --question "what is the name of justin bieber brother" \
  --topic-entity m.06w2sn5 \
  --triples subKGs/EPR-KGQA/data/dataset/WebQSP/train_simple.jsonl \
  --model gemini-2.5-flash \
  --max-steps 4 \
  --output artifacts/llm_frontend_single.jsonl
```

Run a small WebQSP subset:

```bash
PYTHONPATH=src python3 -m llm_frontend.run_webqsp_llm \
  --webqsp datasets/WebQSP \
  --split test \
  --triples subKGs/EPR-KGQA/data/dataset/WebQSP/test_simple.jsonl \
  --limit 5 \
  --model gemini-2.5-flash \
  --max-steps 6 \
  --cache lru \
  --cache-capacity 512 \
  --output artifacts/webqsp_test_llm_traces.jsonl
```

Run the same workflow while relying on the default Gemini provider and model selection:

```bash
export LLM_PROVIDER=gemini
export LLM_API_KEY=...
export LLM_MODEL=gemini-2.5-flash

PYTHONPATH=src python3 -m llm_frontend.run_webqsp_llm \
  --provider gemini \
  --webqsp datasets/WebQSP \
  --split test \
  --triples subKGs/EPR-KGQA/data/dataset/WebQSP/test_simple.jsonl \
  --limit 5 \
  --model gemini-2.5-flash \
  --max-steps 6 \
  --cache lru \
  --cache-capacity 512 \
  --output artifacts/webqsp_test_llm_traces_gemini.jsonl
```

Use an OpenAI-compatible endpoint instead:

```bash
export LLM_PROVIDER=openai_compatible
export LLM_API_KEY=...
export LLM_MODEL=gpt-4.1-mini
export LLM_BASE_URL=https://api.openai.com/v1

PYTHONPATH=src python3 -m llm_frontend.run_webqsp_llm \
  --provider openai_compatible \
  --webqsp datasets/WebQSP \
  --split test \
  --triples subKGs/EPR-KGQA/data/dataset/WebQSP/test_simple.jsonl \
  --limit 5 \
  --model gpt-4.1-mini \
  --max-steps 6 \
  --cache lru \
  --cache-capacity 512 \
  --output artifacts/webqsp_test_llm_traces_openai_compatible.jsonl
```

## Trace Format

Each JSONL record contains:

- `question_id`
- `question`
- `topic_entity`
- `gold_inferential_chain`
- `llm_kg_queries`
- `llm_final_answer`
- `gold_answers`
- `num_steps`
- `stop_reason`

Each `llm_kg_queries` item contains:

- `step_id`
- `relation`
- `direction`
- `requested_direction`
- `input_frontier`
- `output_frontier`
- `raw_model_output`

## Assumptions

- The adapter uses `QueryExecutor.execute_step(...)` for KG hops and preserves its current semantics.
- `direction` in the model output is treated as a hint; the backend still resolves the actual hop direction automatically and the trace stores that resolved direction.
- For ad-hoc questions there is no entity linker. The best path is to provide `--topic-entity`. If the starting frontier is empty, the model may still supply `entity` in its first `KG_QUERY`.
