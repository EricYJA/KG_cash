import json
import re


def prepare_dataset_for_eval(dataset_name, output_file):
    if dataset_name == 'cwq':
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'qald':
        with open('../data/qald_10-en.json', encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
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
    elif dataset_name == 'hotpot_e':
        with open('../data/hotpotadv_dev.json', encoding='utf-8') as file:
            datas = json.load(file)
        question_string = 'question'
    elif dataset_name == 'fin':
        with open('../data/finkg_qa.json', encoding='utf-8') as file:
            datas = json.load(file)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../data/webqsp_test.json', encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'fever':
        with open('../data/fever_1000.json', encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'claim'
    else:
        print("dataset not found")
        exit(-1)
    with open(output_file, encoding='utf-8') as f:
        output_datas = json.load(f)
    return datas, question_string, output_datas


def align(dataset_name, question_string, data, ground_truth_datas, origin_data=None, data_question_string='question'):
    answer_list= []
    if origin_data is None:
        origin_data = [j for j in ground_truth_datas if j[question_string] == data[data_question_string]][0]
    else:
        assert origin_data[question_string] == data[data_question_string], f'not matching origin_data <-> data:\n{origin_data}\n<----->\n{data}'
    if dataset_name == 'cwq':
        if 'answers' in origin_data:
            answers = origin_data["answers"]
        else:
            answers = origin_data["answer"]
        answer_list.append(answers)
    elif dataset_name == 'grailqa':
        answers = origin_data["answer"]
        for answer in answers:
            if "entity_name" in answer:
                answer_list.append(answer['entity_name'])
            else:
                answer_list.append(answer['answer_argument'])
    elif dataset_name == 'qald':
        answers = origin_data["answer"]
        for answer in answers:
            answer_list.append(answers[answer])
    elif dataset_name == 'simpleqa':
        answers = origin_data["answer"]
        answer_list.append(answers)
    elif dataset_name == 'hotpot_e':
        answers = origin_data["answer"]
        answer_list.append(answers)
    elif dataset_name == 'qald':
        answers = origin_data["answer"]
        for answer in answers:
            answer_list.append(answers[answer])
    elif dataset_name == 'fever':
        answer = origin_data['label']
        possible_answer_dict = {
            'REFUTES': ['refutes', 'refute', 'false', 'incorrect', 'not accurate', 'not true', 'not correct',
                        'does not make sense', 'not entirely accurate', 'incomplete'],
            'SUPPORTS': ['supports', 'support', 'true', 'correct'],
            'NOT ENOUGH INFO': ['not enough information', 'not enough info']
        }
        alt_ans = possible_answer_dict[answer]
        answer_list.append(answer)
        for ans in alt_ans:
            answer_list.append(ans)
    elif dataset_name == 'webqsp':
        answer_list = origin_data["answers"]
    elif dataset_name == 'trex' or dataset_name == 'zeroshotre':
        answers = origin_data["answer"]
        answer_list.append(answers)
    elif dataset_name == 'fin':
        answer_list = origin_data["answer"]
    elif dataset_name == 'creak':
        answer = origin_data['label']
        answer_list.append(answer)



    return list(set(answer_list))
    
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
                     
def extract_content(s):
    matches = re.findall(r'\{(.*?)\}', s)
    if len(matches) >= 2 and matches[0].lower() == 'yes':
        return matches[1]
    elif len(matches) >= 1:
        return matches[0]
    else:
        return 'NULL'
