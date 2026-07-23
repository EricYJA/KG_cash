## Upon successfully installing all the necessary configurations, you can proceed to execute ToG-2 directly by employing the following command:

```sh
python main_tog2.py \  
--dataset <dataset_name> \  # dataset your wanna test. "hotpot_e" for HotpotQA, "fever", "qald", "creak", "webqsp", "zeroshotre".
--max_length 256 \  # max length for generation.
--temperature_exploration 0 \  # the temperature in exploration stage, our default setting is 0.
--temperature_reasoning 0 \  # the temperature in reasoning stage, our default setting is 0.
--width 3 \  # choose the search width, 3 is the default setting.
--depth 3 \  # choose the search depth, 3 is the default setting.
--remove_unnecessary_rel True \  # whether removing unnecessary relations.
--LLM_type_rp gpt-3.5-turbo \  # the LLM you choose for RP
--LLM_type gpt-3.5-turbo \  # the LLM you choose for other generations.
--opeani_api_keys <api_key> \  # your own api key, if LLM_type is local LLMs, this parameter would be rendered ineffective.
--embedding_model_name <search_model_name> \  # model name for text search. "bge-bi" (bge embedding), "bge-ce" (bge reranker),"bm25","minilm" (ms-marco-MiniLM-L-6-v2) and "colbert" (bge-m3).
--relation_prune_combination True\  # whether perform relation_prune_combination. 
--num_sents_for_reasoning 10 \  # number of sentences retained for reasoning. 10 is the default setting.
--topic_prune True \  # whether perform topic prune.
--self_consistency_threshold 0.8 \  # self-consistency threshold, 0.8 is the default setting.
--clue_query \  # whether use clue query.
```
