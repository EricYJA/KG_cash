import json
import subprocess
import sys
from pathlib import Path


def main():
    webqsp_file = "../datasets/WebQSP/data/WebQSP.test.json"
    kg_path = "../datasets/WebQSP_KG"
    eval_script = "../datasets/WebQSP/eval/eval.py"

    output_traces = "../results_direct_webqsp_test_traces.jsonl"
    output_predictions = "../results_direct_webqsp_test_predictions.json"

    print("Running full LLM + KG pipeline on WebQSP partial test set...")
    print(f"Dataset:    {webqsp_file}")
    print(f"KG path:    {kg_path}")

    # Step 1: run predictions
    cmd = [
        sys.executable, "-m", "llm_frontend.run_webqsp_llm",
        "--webqsp", webqsp_file,
        "--split", "test",
        "--limit", "400",
        "--kg-path", kg_path,
        "--output", output_traces,
        "--controller", "direct",
    ]
    print(f"\nExecuting: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nPrediction step failed with return code {e.returncode}")
        sys.exit(e.returncode)

    print(f"\nTraces saved to: {output_traces}")

    # Step 2: convert traces JSONL → eval.py prediction format
    predictions = []
    with open(output_traces, encoding="utf-8") as fh:
        for line in fh:
            trace = json.loads(line)
            predictions.append({
                "QuestionId": trace["question_id"],
                "Answers": trace["llm_final_answer"],
            })

    with open(output_predictions, "w", encoding="utf-8") as fh:
        json.dump(predictions, fh, indent=2)

    print(f"Predictions saved to: {output_predictions}")

    # Step 3: run official WebQSP eval
    # --all skips the Good+Complete quality filter, required for partial datasets
    # which have no Good+Complete parses (only Good+Partial etc.)
    cmd_eval = [
        sys.executable, eval_script,
        webqsp_file,
        output_predictions,
        "--all",
    ]
    print(f"\nExecuting eval: {' '.join(cmd_eval)}")
    try:
        subprocess.run(cmd_eval, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nEval step failed with return code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
