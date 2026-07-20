# Eval

We use **Exact Match** as our evaluation metric.

After getting the final result file, use the following command to evaluate the results:

```sh
python eval.py \  
--dataset <dataset_name> \ # dataset your wanna test. "hotpot_e" for HotpotQA, "fever", "qald", "creak", "webqsp", "zeroshotre".
--output_file <output_file_name> \  # the output file should be .json.
--data_question_string <question_string> \  # key to access the question in the output file.
```
