# AGENTS.md

This repository is for a KG caching project.

Goal:
Implement only the query -> KG backend for now.
Do not implement LLM planning.
Do not modify submodules directly unless explicitly required.

Relevant paths:
- KG-R1/
- SubgraphRetrievalKBQA/
- WebQSP dataset already downloaded
- New code should live under `kg_cache_backend/`

Requirements:
- Keep code concise and readable.
- No excessive error handling.
- Prefer Python standard library.
- Use WebQSP gold inferential chains as the execution plan.
- Do not build a full SPARQL engine.
- Implement:
  - webqsp_loader.py
  - kg_store.py
  - cache.py
  - executor.py
  - eval_backend.py
  - build_webqsp_query_traces.py
  - README_backend.md
  - environment.yml

Environment:
- Create conda env named `kg-cache`
- Minimal dependencies only

Validation:
- Show the commands used
- Summarize files created
- Explain any assumptions about local triple/subgraph format