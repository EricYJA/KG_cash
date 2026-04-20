# WebQSP_KG

This directory contains a WebQSP dataset-specific subKG derived from the EPR-KGQA WebQSP files under:

- `ref_KG_projects/EPR-KGQA/data/dataset/WebQSP/train_simple.jsonl`
- `ref_KG_projects/EPR-KGQA/data/dataset/WebQSP/dev_simple.jsonl`
- `ref_KG_projects/EPR-KGQA/data/dataset/WebQSP/test_simple.jsonl`

The builder unions and deduplicates every `subgraph.tuples` triple across the requested splits, then writes files loadable by `src/kg_backend`:

- `triples.tsv`
- `entities.tsv`
- `relations.tsv`
- `entity_name_to_ids.tsv`
- `stats.json`

`entities.tsv` keeps the backend-friendly `id` / `label` schema, but the `label` column now uses a recovered human-readable name when one can be mined from:

- `type.object.name` triples in the EPR WebQSP subgraphs
- `datasets/WebQSP/data/WebQSP.train.json`
- `datasets/WebQSP/data/WebQSP.test.json`

If no name is found for an entity, the label falls back to the raw id. `entity_name_to_ids.tsv` is the reverse lookup file for name-to-id search and may contain multiple ids for the same name.

Build or rebuild the dataset-specific subKG:

```bash
PYTHONPATH=src python datasets/WebQSP_KG/build_webqsp_subkg.py
```

Load it with the local backend:

```bash
PYTHONPATH=src python -m kg_backend.main --data-path datasets/WebQSP_KG
```

Notes:

- This is a dataset-specific subKG, not the full Freebase graph.
- Literal values from the source tuples are preserved as string node ids so the backend can load them without special casing.
