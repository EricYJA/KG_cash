import argparse

from src.load_data import load_webqsp, load_cwq, inspect_sample, extract_entities
from src.entity_stats import compute_redundancy_stats
from src.subgraph_analysis import analyze_subgraph_overlap

# CACHE_SIZES = [10, 50, 100, 200, 500, 1000, 2000, 5000]

def run_webqsp():
    webqsp = load_webqsp()
    # inspect_sample(webqsp, split='train', index=0)

    _, webqsp_flat = extract_entities(webqsp, split='train')
    webqsp_stats = compute_redundancy_stats(webqsp_flat, "WebQSP")

    print("\n── Subgraph Analysis: WebQSP ──")
    webqsp_sg = analyze_subgraph_overlap(webqsp, split='train')
    webqsp_stats['subgraph_analysis'] = webqsp_sg

    return webqsp_stats


def run_cwq():
    cwq = load_cwq()
    # inspect_sample(cwq, split='train', index=0)

    _, cwq_flat = extract_entities(cwq, split='train')
    cwq_stats = compute_redundancy_stats(cwq_flat, "CWQ")

    print("\n── Subgraph Analysis: CWQ ──")
    cwq_sg = analyze_subgraph_overlap(cwq, split='train')
    cwq_stats['subgraph_analysis'] = cwq_sg

    return cwq_stats


def run_both():
    run_webqsp()
    run_cwq()

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["webqsp", "cwq", "both"],
        default="both",
        help="Select which dataset pipeline to run.",
    )
    args = parser.parse_args()

    if args.dataset == "webqsp":
        run_webqsp()
        print("\nDone.")
    elif args.dataset == "cwq":
        run_cwq()
        print("\nDone.")
    else:
        run_both()


if __name__ == "__main__":
    main()
