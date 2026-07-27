import argparse
from utils import *


DEFAULT_OUTPUT_FILES = {
    "cwq": "../output/ToG_cwq.jsonl",
    "webqsp": "../output/ToG_webqsp.jsonl",
}


def evaluate(dataset, output_file, constraints_refuse):
    ground_truth_datas, question_string, output_datas = prepare_dataset_for_eval(dataset, output_file)

    num_right = 0
    num_error = 0
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    evaluated = 0

    for data in output_datas:
        answers = align(dataset, question_string, data, ground_truth_datas)
        answer_groups = align_answer_groups(dataset, question_string, data, ground_truth_datas)
        results = data['results']
        if check_string(results):
            response = extract_content(results)
            if response == "NULL":
                response = results
        else:
            response = results
            if constraints_refuse and check_refuse(response):
                continue

        predictions = split_prediction_answers(response)
        precision, recall, f1 = precision_recall_f1(predictions, answer_groups)
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        evaluated += 1

        if exact_match(response, answers):
            num_right += 1
        else:
            num_error += 1

    metrics = {
        "exact_match": float(num_right / evaluated) if evaluated else 0.0,
        "precision": float(precision_sum / evaluated) if evaluated else 0.0,
        "recall": float(recall_sum / evaluated) if evaluated else 0.0,
        "f1": float(f1_sum / evaluated) if evaluated else 0.0,
        "right": num_right,
        "error": num_error,
        "evaluated": evaluated,
        "total": len(output_datas),
    }

    print("{} Results".format(dataset))
    print("Exact Match: {}".format(metrics["exact_match"]))
    print("Precision: {}".format(metrics["precision"]))
    print("Recall: {}".format(metrics["recall"]))
    print("F1: {}".format(metrics["f1"]))
    print("right: {}, error: {}, evaluated: {}, total: {}".format(
        num_right,
        num_error,
        evaluated,
        len(output_datas),
    ))

    save_result2json_with_prf(dataset, metrics, "ToG")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="cwq", help="choose the dataset.")
    parser.add_argument("--output_file", type=str,
                        default=None, help="the output file name.")
    parser.add_argument("--constraints_refuse", type=bool,
                        default=True, help="LLM may have refuse erorr, enable this option to skip current sample.")
    args = parser.parse_args()

    if args.dataset == "both":
        for dataset in ["webqsp", "cwq"]:
            evaluate(dataset, DEFAULT_OUTPUT_FILES[dataset], args.constraints_refuse)
    else:
        output_file = args.output_file or DEFAULT_OUTPUT_FILES.get(args.dataset, "ToG_{}.json".format(args.dataset))
        evaluate(args.dataset, output_file, args.constraints_refuse)
