import json
import re
import string


# --- RoG-compatible metrics (Hits@1 / F1) -------------------------------------
# Copied verbatim from ref_KG_projects/RoG/src/qa_prediction/evaluate_results.py
# so ToG's Hits@1 / F1 are computed identically to the RoG experiment and the two
# are directly comparable. Existing Exact Match is left untouched.
def rog_normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def rog_match(s1: str, s2: str) -> bool:
    return rog_normalize(s2) in rog_normalize(s1)


def rog_eval_hit(prediction, answer) -> int:
    """1 if any gold answer appears in the (joined) prediction, else 0."""
    prediction_str = " ".join(prediction) if isinstance(prediction, list) else prediction
    for a in answer:
        if rog_match(prediction_str, a):
            return 1
    return 0


def rog_eval_f1(prediction, answer):
    """(f1, precision, recall); precision divides by number of predicted items."""
    if len(prediction) == 0 or len(answer) == 0:
        return 0.0, 0.0, 0.0
    prediction_str = " ".join(prediction)
    matched = sum(1 for a in answer if rog_match(prediction_str, a))
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0.0, precision, recall
    return 2 * precision * recall / (precision + recall), precision, recall


def prediction_to_list(results):
    """Turn a ToG `results` string into a RoG-style list of predicted answers.

    RoG splits its prediction on newlines; ToG usually emits a single line, so
    this yields a 1-element list in the common case (precision divides by 1).
    """
    if isinstance(results, list):
        items = results
    else:
        text = clean_results(results)
        if text == "NULL":
            text = results if isinstance(results, str) else str(results)
        items = text.split("\n")
    items = [p.strip() for p in items if isinstance(p, str) and p.strip()]
    return items or [str(results)]


def prepare_dataset_for_eval(dataset_name, output_file):
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
    with open(output_file, encoding='utf-8') as f:
        if output_file.endswith(".jsonl"):
            output_datas = [json.loads(line) for line in f if line.strip()]
        else:
            output_datas = json.load(f)
    return datas, question_string, output_datas


def align(dataset_name, question_string, data, ground_truth_datas):
    answer_list= []
    output_question = data.get(question_string, data.get("question"))
    origin_data = [j for j in ground_truth_datas if j[question_string] == output_question][0]
    if dataset_name == 'cwq':
        if 'answers' in origin_data:
            answers = origin_data["answers"]
        else:
            answers = origin_data["answer"]
        if isinstance(answers, str):
            answer_list.append(answers)
        else:
            for answer in answers:
                if isinstance(answer, str):
                    answer_list.append(answer)
                    continue
                alias = answer['aliases']
                ans = answer['answer']
                alias.append(ans)
                answer_list.extend(alias)

    elif dataset_name == 'webqsp':
        answers = origin_data["Parses"]
        for answer in answers:
            for name in answer['Answers']:
                if name['EntityName'] == None:
                    answer_list.append(name['AnswerArgument'])
                else:
                    answer_list.append(name['EntityName'])

    elif dataset_name == 'grailqa':
        answers = origin_data["answer"]
        for answer in answers:
            if "entity_name" in answer:
                answer_list.append(answer['entity_name'])
            else:
                answer_list.append(answer['answer_argument'])

    elif dataset_name == 'simpleqa':
        answers = origin_data["answer"]
        answer_list.append(answers)

    elif dataset_name == 'qald':
        answers = origin_data["answer"]
        for answer in answers:
            answer_list.append(answers[answer])
        
    elif dataset_name == 'webquestions':
        answer_list = origin_data["answers"]

    elif dataset_name == 'trex' or dataset_name == 'zeroshotre':
        answers = origin_data["answer"]
        answer_list.append(answers)

    elif dataset_name == 'creak':
        answer = origin_data['label']
        answer_list.append(answer)

    return list(set(answer_list))


def align_answer_groups(dataset_name, question_string, data, ground_truth_datas):
    answer_groups = []
    output_question = data.get(question_string, data.get("question"))
    origin_data = [j for j in ground_truth_datas if j[question_string] == output_question][0]

    if dataset_name == 'cwq':
        if 'answers' in origin_data:
            answers = origin_data["answers"]
        else:
            answers = origin_data["answer"]
        if isinstance(answers, str):
            answer_groups.append([answers])
        else:
            for answer in answers:
                if isinstance(answer, str):
                    answer_groups.append([answer])
                    continue
                aliases = list(answer['aliases'])
                aliases.append(answer['answer'])
                answer_groups.append(list(set(aliases)))

    elif dataset_name == 'webqsp':
        answers = origin_data["Parses"]
        seen = set()
        for answer in answers:
            for name in answer['Answers']:
                if name['EntityName'] == None:
                    value = name['AnswerArgument']
                else:
                    value = name['EntityName']
                clean_value = value.strip().replace(" ", "").lower()
                if clean_value not in seen:
                    seen.add(clean_value)
                    answer_groups.append([value])

    else:
        answer_groups = [[answer] for answer in align(dataset_name, question_string, data, ground_truth_datas)]

    return answer_groups
    
def check_string(string):
    return "{" in string

def clean_results(string):
    if "{" in string:
        start = string.find("{") + 1
        end = string.find("}")
        content = string[start:end]
        return content
    else:
        return "NULL"
    

def check_refuse(string):
    refuse_words = ["however", "sorry"]
    return any(word in string.lower() for word in refuse_words)


def exact_match(response, answers):
    clean_result = response.strip().replace(" ","").lower()
    for answer in answers:
        clean_answer = answer.strip().replace(" ","").lower()
        if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
            return True
    return False


def split_prediction_answers(response):
    if check_string(response):
        content = extract_content(response)
        if content == "NULL":
            content = response
        split_candidates = re.split(r"\s*(?:;|\||\n)\s*", content)
        predictions = [item.strip(" .,:;{}[]()\"'") for item in split_candidates if item.strip(" .,:;{}[]()\"'")]
        return predictions if predictions else [content]
    cleaned = response.strip()
    return [cleaned] if cleaned else []


def answer_group_match(prediction, answer_group):
    return exact_match(prediction, answer_group)


def precision_recall_f1(predictions, answer_groups):
    if not predictions and not answer_groups:
        return 1.0, 1.0, 1.0
    if not predictions:
        return 0.0, 0.0, 0.0
    if not answer_groups:
        return 0.0, 0.0, 0.0

    matched_gold = set()
    true_positive = 0
    for prediction in predictions:
        for gold_idx, answer_group in enumerate(answer_groups):
            if gold_idx in matched_gold:
                continue
            if answer_group_match(prediction, answer_group):
                matched_gold.add(gold_idx)
                true_positive += 1
                break

    precision = true_positive / len(predictions) if predictions else 0.0
    recall = true_positive / len(answer_groups) if answer_groups else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1

def save_result2json(dataset_name, num_right, num_error, total_nums, method, extra_metrics=None):
    results_data = {
        'dataset': dataset_name,
        'method': method,
        'Exact Match': float(num_right/total_nums),
        'Right Samples': num_right,
        'Error Sampels': num_error
    }
    if extra_metrics:
        results_data.update(extra_metrics)
    with open('ToG_{}_results.json'.format(dataset_name), 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)


def save_result2json_with_prf(dataset_name, metrics, method):
    results_data = {
        'dataset': dataset_name,
        'method': method,
        'Exact Match': metrics['exact_match'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1': metrics['f1'],
        'Right Samples': metrics['right'],
        'Error Samples': metrics['error'],
        'Evaluated Samples': metrics['evaluated'],
        'Total Samples': metrics['total'],
    }
    with open('ToG_{}_results.json'.format(dataset_name), 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)
                     
def extract_content(s):
    matches = re.findall(r'\{(.*?)\}', s)
    if len(matches) >= 2 and matches[0].lower() == 'yes':
        return matches[1]
    elif len(matches) >= 1:
        return matches[0]
    else:
        return 'NULL'
