from freebase_func import *
from prompt_list import *
import argparse
import json
import os
import re
import time
from openai import OpenAI
from rank_bm25 import BM25Okapi
from llm_config import resolve_llm_config
from llm_client import LLMChatClient, ChatMessage
from sentence_transformers import util
from sentence_transformers import SentenceTransformer


# Process-wide count of LLM calls made through run_llm(), so a run can report how
# many LLM calls it made (and, against the per-miss average, how many a cache
# saved) -- the ToG analog of RoG's planner_llm_calls / planner_llm_calls_saved.
_LLM_CALLS = 0


def llm_call_count():
    return _LLM_CALLS


# Restart-safe per-question metrics live in cache_metrics (stdlib only) so scripts
# can reuse them without utils.py's ML imports; re-exported here for the callers
# that reach them via `from utils import *`.
from cache_metrics import (  # noqa: E402
    aggregate_run_metrics,
    append_question_metrics,
    metrics_sidecar_path,
)


def parse_test_limit(value):
    """argparse type for --test-limit: an int, or 'all' for the whole split.

    Returns None for 'all', which is what the callers already treat as "no
    limit". The RoG runners spell a full run `--limit all` (see split_for in
    scripts/_rog_common.py); accepting it here lets every ToG caller forward the
    string through unchanged instead of each one special-casing it.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "all", "none"):
        return None
    try:
        limit = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid value {value!r}: expected a positive integer or 'all'")
    if limit <= 0:
        raise argparse.ArgumentTypeError(
            f"invalid value {value!r}: expected a positive integer or 'all'")
    return limit

def retrieve_top_docs(query, docs, model, width=3):
    """
    Retrieve the topn most relevant documents for the given query.

    Parameters:
    - query (str): The input query.
    - docs (list of str): The list of documents to search from.
    - model_name (str): The name of the SentenceTransformer model to use.
    - width (int): The number of top documents to return.

    Returns:
    - list of float: A list of scores for the topn documents.
    - list of str: A list of the topn documents.
    """

    query_emb = model.encode(query)
    doc_emb = model.encode(docs)

    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)

    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]

    return top_docs, top_scores


def compute_bm25_similarity(query, corpus, width=3):
    """
    Computes the BM25 similarity between a question and a list of relations,
    and returns the topn relations with the highest similarity along with their scores.

    Args:
    - question (str): Input question.
    - relations_list (list): List of relations.
    - width (int): Number of top relations to return.

    Returns:
    - list, list: topn relations with the highest similarity and their respective scores.
    """

    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)
    
    relations = bm25.get_top_n(tokenized_query, corpus, n=width)
    doc_scores = sorted(doc_scores, reverse=True)[:width]

    return relations, doc_scores


def clean_relations(string, entity_id, head_relations, tail_relations=None):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    allowed_tail = set(tail_relations) if tail_relations is not None else None
    allowed_head = set(head_relations)
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        if relation not in allowed_head and (allowed_tail is None or relation not in allowed_tail):
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in allowed_head:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    if not relations:
        return False, "No relations found"
    return True, relations


def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)


def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations):
    relations = []
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    for relation in topn_relations:
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False})
    return True, relations


# ToG routes every stage -- relation pruning, entity pruning, the sufficiency
# check, the final answer -- through run_llm, so one model serves planning and
# answering. Printed once per process so a run's log states which model that
# actually was: an omitted --model resolves to the vendor preset in silence, and
# a run tag naming a different one is not evidence it was used.
_ANNOUNCED_MODEL = None


def _announce_model(vendor, model):
    global _ANNOUNCED_MODEL
    if _ANNOUNCED_MODEL != (vendor, model):
        _ANNOUNCED_MODEL = (vendor, model)
        print(f"[llm] all stages using {vendor}/{model}", flush=True)


def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo", vendor=None, model=None):
    global _LLM_CALLS
    _LLM_CALLS += 1
    # An explicit --model overrides the vendor preset (tamu) or the engine id (openai/google).
    model = model or None  # normalise "" -> None so the preset default wins
    if vendor == "tamu":
        config = resolve_llm_config(vendor="tamu", model=model)
        _announce_model(config.vendor, config.model)
        http_client = LLMChatClient(config, timeout_s=180.0)
        messages = [
            ChatMessage(role="system", content="You are an AI assistant that helps people find information."),
            ChatMessage(role="user", content=prompt),
        ]
        print("start tamu")
        result = http_client.complete_json(messages, temperature=temperature)
        print("end tamu")
        return result

    engine = model or engine  # openai/google: the model id is the engine
    _announce_model(vendor or "openai", engine)
    if "llama" in engine.lower():
        client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
        engine = client.models.list().data[0].id
    else:
        client = OpenAI(api_key=opeani_api_keys)

    messages = [
        {"role": "system", "content": "You are an AI assistant that helps people find information."},
        {"role": "user", "content": prompt},
    ]
    print("start openai")
    while True:
        try:
            response = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=0,
                presence_penalty=0,
            )
            result = response.choices[0].message.content
            break
        except Exception:
            print("openai error, retry")
            time.sleep(2)
    print("end openai")
    return result

def construct_relation_prune_prompt(question, entity_name, total_relations, args):
    return extract_relation_prompt % (args.width, args.width) + question + '\nTopic Entity: ' + entity_name + '\nRelations: '+ '; '.join(total_relations) + "\nA: "
        

def construct_entity_score_prompt(question, relation, entity_candidates):
    return score_entity_candidates_prompt.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '

def relation_search_prune(entity_id, entity_name, pre_relations, pre_head, question, args):
    sparql_relations_extract_head = sparql_head_relations % (entity_id)
    head_relations = execurte_sparql(sparql_relations_extract_head)
    head_relations = replace_relation_prefix(head_relations)
    
    sparql_relations_extract_tail= sparql_tail_relations % (entity_id)
    tail_relations = execurte_sparql(sparql_relations_extract_tail)
    tail_relations = replace_relation_prefix(tail_relations)

    if args.remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
    

    if len(pre_relations) != 0 and pre_head !=-1:
        tail_relations = [rel for rel in tail_relations if not pre_head or rel not in pre_relations]
        head_relations = [rel for rel in head_relations if pre_head or rel not in pre_relations]

    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    total_relations = head_relations+tail_relations
    total_relations.sort()  # make sure the order in prompt is always equal
    
    if args.prune_tools == "llm":
        prompt = construct_relation_prune_prompt(question, entity_name, total_relations, args)

        result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type, vendor=getattr(args, "vendor", None), model=getattr(args, "model", None))
        flag, retrieve_relations_with_scores = clean_relations(result, entity_id, head_relations, tail_relations)

    elif args.prune_tools == "bm25":
        topn_relations, topn_scores = compute_bm25_similarity(question, total_relations, args.width)
        flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations) 
    else:
        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
        topn_relations, topn_scores = retrieve_top_docs(question, total_relations, model, args.width)
        flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations) 

    if flag:
        return retrieve_relations_with_scores
    else:
        return [] # format error or too small max_length
    
    
def entity_search(entity, relation, head=True):
    if head:
        tail_entities_extract = sparql_tail_entities_extract% (entity, relation)
        entities = execurte_sparql(tail_entities_extract)
    else:
        head_entities_extract = sparql_head_entities_extract% (entity, relation)
        entities = execurte_sparql(head_entities_extract)


    entity_ids = replace_entities_prefix(entities)
    new_entity = [entity for entity in entity_ids if entity.startswith("m.")]
    return new_entity


def entity_score(question, entity_candidates_id, score, relation, args):
    entity_candidates = [id2entity_name_or_type(entity_id) for entity_id in entity_candidates_id]
    if all_unknown_entity(entity_candidates):
        return [1/len(entity_candidates) * score] * len(entity_candidates), entity_candidates, entity_candidates_id
    entity_candidates = del_unknown_entity(entity_candidates)
    if len(entity_candidates) == 1:
        return [score], entity_candidates, entity_candidates_id
    if len(entity_candidates) == 0:
        return [0.0], entity_candidates, entity_candidates_id
    
    # make sure the id and entity are in the same order
    zipped_lists = sorted(zip(entity_candidates, entity_candidates_id))
    entity_candidates, entity_candidates_id = zip(*zipped_lists)
    entity_candidates = list(entity_candidates)
    entity_candidates_id = list(entity_candidates_id)
    if args.prune_tools == "llm":
        prompt = construct_entity_score_prompt(question, relation, entity_candidates)

        result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type, vendor=getattr(args, "vendor", None), model=getattr(args, "model", None))
        return [float(x) * score for x in clean_scores(result, entity_candidates)], entity_candidates, entity_candidates_id

    elif args.prune_tools == "bm25":
        topn_entities, topn_scores = compute_bm25_similarity(question, entity_candidates, args.width)
    else:
        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
        topn_entities, topn_scores = retrieve_top_docs(question, entity_candidates, model, args.width)
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    return [float(x) * score for x in topn_scores], topn_entities, entity_candidates_id

    
def all_unknown_entity(entity_candidates):
    return all(candidate == "UnName_Entity" for candidate in entity_candidates)

def del_unknown_entity(entity_candidates):
    if len(entity_candidates)==1 and entity_candidates[0]=="UnName_Entity":
        return entity_candidates
    entity_candidates = [candidate for candidate in entity_candidates if candidate != "UnName_Entity"]
    return entity_candidates

def clean_scores(string, entity_candidates):
    scores = re.findall(r'\d+\.\d+', string)
    scores = [float(number) for number in scores]
    if len(scores) == len(entity_candidates):
        return scores
    else:
        print("All entities are created equal.")
        return [1/len(entity_candidates)] * len(entity_candidates)

def update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head):
    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
        entity_candidates_id = ["[FINISH_ID]"]
    candidates_relation = [entity['relation']] * len(entity_candidates)
    topic_entities = [entity['entity']] * len(entity_candidates)
    head_num = [entity['head']] * len(entity_candidates)
    total_candidates.extend(entity_candidates)
    total_scores.extend(scores)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)
    return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head


def generate_answer(question, cluster_chain_of_entities, args): 
    prompt = answer_prompt + question + '\n'
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
    result = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, vendor=getattr(args, "vendor", None), model=getattr(args, "model", None))
    return result


def extract_record_id(data, dataset):
    """Stable per-question id for the output record (matches RoG's `id`)."""
    if dataset == "webqsp":
        return data.get("QuestionId")
    if dataset == "cwq":
        return data.get("ID")
    return data.get("id") or data.get("ID") or data.get("QuestionId")


def extract_ground_truth(data, dataset):
    """Readable gold-answer list for a dataset row (mirrors eval align())."""
    answers = []
    if dataset == "webqsp":
        for parse in data.get("Parses", []) or []:
            for a in parse.get("Answers", []) or []:
                answers.append(a.get("EntityName") or a.get("AnswerArgument"))
    elif dataset == "cwq":
        raw = data.get("answers", data.get("answer"))
        if isinstance(raw, str):
            answers.append(raw)
        elif isinstance(raw, list):
            for a in raw:
                if isinstance(a, dict):
                    answers.extend(a.get("aliases", []) or [])
                    if a.get("answer"):
                        answers.append(a["answer"])
                elif a is not None:
                    answers.append(a)
    return [str(x) for x in answers if x is not None]


def save_2_jsonl(question, answer, cluster_chain_of_entities, file_name, output_file=None,
                 qid=None, ground_truth=None, loop_idx=None):
    dict = {}
    if qid is not None:
        dict["id"] = qid
    dict["question"] = question
    dict["results"] = answer
    if ground_truth is not None:
        dict["ground_truth"] = ground_truth
    if loop_idx is not None:
        dict["loop_idx"] = loop_idx
    dict["reasoning_chains"] = cluster_chain_of_entities
    output_path = output_file or os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "ToG_{}.jsonl".format(file_name))
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")


def entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args):
    zipped = list(zip(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores))
    sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
    sorted_entities_id, sorted_relations, sorted_candidates, sorted_topic_entities, sorted_head, sorted_scores = [x[0] for x in sorted_zipped], [x[1] for x in sorted_zipped], [x[2] for x in sorted_zipped], [x[3] for x in sorted_zipped], [x[4] for x in sorted_zipped], [x[5] for x in sorted_zipped]

    entities_id, relations, candidates, topics, heads, scores = sorted_entities_id[:args.width], sorted_relations[:args.width], sorted_candidates[:args.width], sorted_topic_entities[:args.width], sorted_head[:args.width], sorted_scores[:args.width]
    merged_list = list(zip(entities_id, relations, candidates, topics, heads, scores))
    filtered_list = [(id, rel, ent, top, hea, score) for id, rel, ent, top, hea, score in merged_list if score != 0]
    if len(filtered_list) ==0:
        return False, [], [], [], []
    entities_id, relations, candidates, tops, heads, scores = map(list, zip(*filtered_list))

    tops = [id2entity_name_or_type(entity_id) for entity_id in tops]
    cluster_chain_of_entities = [[(tops[i], relations[i], candidates[i]) for i in range(len(candidates))]]
    return True, cluster_chain_of_entities, entities_id, relations, heads


def reasoning(question, cluster_chain_of_entities, args):
    prompt = prompt_evaluate + question
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '

    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, vendor=getattr(args, "vendor", None), model=getattr(args, "model", None))
    
    result = extract_answer(response)
    if if_true(result):
        return True, response
    else:
        return False, response
    
def extract_answer(text):
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index+1:end_index].strip()
    else:
        return ""
    
def if_true(prompt):
    if prompt.lower().strip().replace(" ","")=="yes":
        return True
    return False

def half_stop(question, cluster_chain_of_entities, args, qid=None, ground_truth=None, loop_idx=None):
    print("No new knowledge added during search depth %d, stop searching." % args.depth)
    answer = generate_answer(question, cluster_chain_of_entities, args)
    save_2_jsonl(question, answer, cluster_chain_of_entities, file_name=args.dataset,
                 output_file=getattr(args, "output_file", None),
                 qid=qid, ground_truth=ground_truth, loop_idx=loop_idx)


def generate_without_explored_paths(question, args):
    prompt = generate_directly + "\n\nQ: " + question + "\nA:"
    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, vendor=getattr(args, "vendor", None), model=getattr(args, "model", None))
    return response

def prepare_dataset(dataset_name):
    if dataset_name == 'cwq':
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'grailqa':
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'simpleqa':
        with open('../data/SimpleQA.json',encoding='utf-8') as f:
            datas = json.load(f)    
        question_string = 'question'
    elif dataset_name == 'qald':
        with open('../data/qald_10-en.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'   
    elif dataset_name == 'webquestions':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'trex':
        with open('../data/T-REX.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'zeroshotre':
        with open('../data/Zero_Shot_RE.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'creak':
        with open('../data/creak.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'sentence'
    elif dataset_name in ('lcquad', 'lcquad_train'):
        with open('../data/lcquad_train.json',encoding='utf-8') as f:
            datas = json.load(f)
        datas = [d for d in datas if d.get('question')]
        question_string = 'question'
    elif dataset_name == 'lcquad_test':
        with open('../data/lcquad_test.json',encoding='utf-8') as f:
            datas = json.load(f)
        datas = [d for d in datas if d.get('question')]
        question_string = 'question'
    else:
        print("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak, lcquad, lcquad_train, lcquad_test}.")
        exit(-1)
    return datas, question_string
