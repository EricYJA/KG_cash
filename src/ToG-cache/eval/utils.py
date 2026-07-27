import json
import re


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

def save_result2json(dataset_name, num_right, num_error, total_nums, method):
    results_data = {
        'dataset': dataset_name,
        'method': method,
        'Exact Match': float(num_right/total_nums),
        'Right Samples': num_right,
        'Error Sampels': num_error
    }
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
