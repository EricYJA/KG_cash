import argparse
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="cwq", help="choose the dataset.")
    parser.add_argument("--output_file", type=str,
                        default="ToG_cwq.json", help="the output file name.")
    parser.add_argument("--constraints_refuse", type=bool,
                        default=True, help="LLM may have refuse erorr, enable this option to skip current sample.")
    args = parser.parse_args()

    ground_truth_datas, question_string, output_datas = prepare_dataset_for_eval(args.dataset, args.output_file)

    num_right = 0
    num_error = 0
    # RoG-compatible metrics (computed over every record, alongside Exact Match)
    hit_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    for data in output_datas:
        # Prefer the gold answers stored inline in the record (self-contained,
        # matches RoG); fall back to align() for older records without it.
        answers = data.get("ground_truth") or align(
            args.dataset, question_string, data, ground_truth_datas)
        results = data['results']

        # --- RoG-style Hits@1 / F1 ---
        prediction = prediction_to_list(results)
        hit_list.append(rog_eval_hit(prediction, answers))
        f1, prec, rec = rog_eval_f1(prediction, answers)
        f1_list.append(f1)
        precision_list.append(prec)
        recall_list.append(rec)

        # --- Exact Match (unchanged) ---
        if check_string(results):
            response = clean_results(results)
            if response=="NULL":
                response = results
            else:
                if exact_match(response, answers):
                    num_right+=1
                else:
                    num_error+=1
        else:
            response = results
            if args.constraints_refuse and check_string(response):
                continue
            if exact_match(response, answers):
                num_right+=1
            else:
                num_error+=1

    n = len(output_datas)
    hits1 = sum(hit_list) / n if n else 0.0
    f1 = sum(f1_list) / n if n else 0.0
    precision = sum(precision_list) / n if n else 0.0
    recall = sum(recall_list) / n if n else 0.0

    print("Exact Match: {}".format(float(num_right/n)))
    print("right: {}, error: {}".format(num_right, num_error))
    print("Hits@1: {}".format(hits1))
    print("F1: {}".format(f1))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))

    save_result2json(args.dataset, num_right, num_error, n, "ToG",
                     extra_metrics={
                         "Hits@1": hits1,
                         "F1": f1,
                         "Precision": precision,
                         "Recall": recall,
                     })
    
