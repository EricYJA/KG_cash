# KG_cash
cache optimization for LLM based KG queries

## Update

Currently the repo only contains datasets and knowledge graph (KG) bundle. Assuming the underlining KG is `Freebase`. 

`dataset/` comtains the questions plus annotations such as answers, topic entities, inferential chains, constraints, and SPARQL queries, but not the full Freebase KG inside the dataset package. The official release describes semantic parses and SPARQL over Freebase, which means you still need a KG backend or a local extracted subgraph store to execute retrieval.

`subKGs/` contains implementations with KG retrieval server that can be used for KG query as a complete system, which the datasets can be executed on. 
