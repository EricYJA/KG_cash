import argparse
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="", help="choose the dataset.")
    parser.add_argument("--output_file", type=str,
                        default="", help="the output file name.")
    parser.add_argument("--data_question_string", type=str,
                        default='question', help="key to access the question in output.")
    args = parser.parse_args()

    ground_truth_datas, question_string, output_datas = prepare_dataset_for_eval(args.dataset, args.output_file)
    ground_truth_datas_question2id = {v[question_string]: i for i, v in enumerate(ground_truth_datas)}
    data_question_string = args.data_question_string

    num_right = 0
    num_error = 0
    err_list = []

    for data_i, data in enumerate(output_datas):
        cur_question = data[data_question_string]
        origin_data_i = ground_truth_datas_question2id[cur_question]
        answers = align(args.dataset, question_string, data, ground_truth_datas, origin_data=ground_truth_datas[origin_data_i], data_question_string=data_question_string)

        results = data['answer'] if 'answer' in data else ''
        if 'fever' in args.dataset:
            if ("no entity find in depth" or "Last situation.Not into depth. whether it trigger") in data['remark']:
                results = 'NOT ENOUGH INFO'
        matched = False
        response = results

        if check_string(results):
            if exact_match(response, answers):
                matched = True
        else:
            if exact_match(response, answers):
                matched = True

        if matched:
            num_right += 1
        else:
            num_error += 1
            err_list.append(data_i)

    print("Exact Match: {}".format(float(num_right/len(output_datas))))
    print("right: {}, error: {}".format(num_right, num_error))

