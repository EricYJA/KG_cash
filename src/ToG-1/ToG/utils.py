from prompt_list import *
import json
import os
import time
import re
from prompt_list import *
from openai import OpenAI
from rank_bm25 import BM25Okapi
from llm_config import resolve_llm_config
from llm_client import LLMChatClient, ChatMessage
from sentence_transformers import util
from sentence_transformers import SentenceTransformer


def is_reasoning_model(engine):
    model_name = engine.lower()
    return (
        model_name.startswith("gpt-5")
        or model_name.startswith("o1")
        or model_name.startswith("o3")
        or model_name.startswith("o4")
    )


def normalize_reasoning_effort(reasoning_effort):
    if reasoning_effort is None:
        return "low"
    reasoning_effort = str(reasoning_effort).strip().lower()
    return reasoning_effort or "low"


def reasoning_completion_budget(max_tokens):
    return max(int(max_tokens), 1024)


def extract_message_content(message):
    content = message.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", item)))
        return "".join(parts)
    return str(content)


def should_retry_openai_error(error):
    error_text = str(error).lower()
    non_retryable_markers = [
        "unsupported parameter",
        "unknown parameter",
        "not compatible",
        "invalid_request",
        "invalid request",
        "does not support",
        "model_not_found",
    ]
    return not any(marker in error_text for marker in non_retryable_markers)


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


def clean_relations(string, entity_id, head_relations):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
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
    i=0
    for relation in topn_relations:
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False})
        i+=1
    return True, relations


def run_llm(
    prompt,
    temperature,
    max_tokens,
    opeani_api_keys,
    engine="gpt-3.5-turbo",
    vendor=None,
    reasoning_effort=None,
):
    if vendor == "tamu":
        config = resolve_llm_config(vendor="tamu")
        http_client = LLMChatClient(config, timeout_s=180.0)
        messages = [
            ChatMessage(role="system", content="You are an AI assistant that helps people find information."),
            ChatMessage(role="user", content=prompt),
        ]
        print("start tamu")
        result = http_client.complete_json(messages, temperature=temperature)
        print("end tamu")
        return result

    if "llama" in engine.lower():
        client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
        engine = client.models.list().data[0].id
    else:
        api_key = opeani_api_keys or os.environ.get("OPENAI_API_KEY")
        if vendor == "openai" and not api_key:
            raise RuntimeError(
                "Missing OpenAI API key. Pass --opeani_api_keys or export OPENAI_API_KEY."
            )
        client = OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": "You are an AI assistant that helps people find information."},
        {"role": "user", "content": prompt},
    ]
    print("start openai")
    while True:
        try:
            request_kwargs = {
                "model": engine,
                "messages": messages,
            }
            if is_reasoning_model(engine):
                request_kwargs["max_completion_tokens"] = reasoning_completion_budget(max_tokens)
                request_kwargs["reasoning_effort"] = normalize_reasoning_effort(reasoning_effort)
            else:
                request_kwargs.update({
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                })
            response = client.chat.completions.create(**request_kwargs)
            result = extract_message_content(response.choices[0].message)
            break
        except Exception as e:
            print("openai error:", repr(e))
            if not should_retry_openai_error(e):
                raise
            print("retry")
            time.sleep(2)
    print("end openai")
    return result

    
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
    

def save_2_jsonl(question, answer, cluster_chain_of_entities, file_name, output_file=None):
    dict = {"question":question, "results": answer, "reasoning_chains": cluster_chain_of_entities}
    output_path = output_file or os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "ToG_{}.jsonl".format(file_name))
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")

    
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


def generate_without_explored_paths(question, args):
    prompt = cot_prompt + "\n\nQ: " + question + "\nA:"
    response = run_llm(
        prompt,
        args.temperature_reasoning,
        args.max_length,
        args.opeani_api_keys,
        args.LLM_type,
        vendor=getattr(args, "vendor", None),
        reasoning_effort=getattr(args, "reasoning_effort", None),
    )
    return response


def if_finish_list(lst):
    if all(elem == "[FINISH_ID]" for elem in lst):
        return True, []
    else:
        new_lst = [elem for elem in lst if elem != "[FINISH_ID]"]
        return False, new_lst


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
    else:
        print("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}.")
        exit(-1)
    return datas, question_string