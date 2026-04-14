# characterization/sentiment_cache.py
import argparse
import logging
import resource
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

from src.load_data import load_webqsp, load_cwq, extract_entities
from src.sentiment_analysis import get_sentiment_vector, extract_questions_from_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("sentiment_cache_comparison.log", mode="w"),
        logging.StreamHandler()
    ]
)

CACHE_SIZES = [10, 50, 100, 200, 500]
SIMILARITY_THRESHOLDS = [0.99,
                         0.95,
                         0.90,
                         0.85,
                         0.80]


def limit_memory():
    max_memory_gb = 26
    max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))

limit_memory()

def simulate_static_cache(questions_list, cache_size):
    """Simulates a global static exact-match cache."""
    counts = Counter(questions_list)
    cache = set(item for item, _ in counts.most_common(cache_size))
    hits = sum(1 for q in questions_list if q in cache)
    return hits / len(questions_list) if questions_list else 0

def simulate_sentiment_cache(questions_list, entities_list, sentiment_dict, cache_size, threshold):
    """
    Simulates a semantic cache using sentiment analysis.
    Falls back to a cosine similarity match on sentiment vectors if exact match fails.
    """
    counts = Counter(questions_list)

    # Populate the cache with globally most frequent questions
    cached_questions = [item for item, _ in counts.most_common(cache_size)]
    exact_cache = set(cached_questions)

    # Dictionary to get entities for cached questions (union of entities if multiple occurrences)
    cached_entities_dict = {}
    for q in cached_questions:
        cached_entities_dict[q] = set()
    for q, ents in zip(questions_list, entities_list):
        if q in cached_entities_dict:
            cached_entities_dict[q].update(ents)

    # Pre-aggregate cached sentiment vectors for fast matrix operations
    if not cached_questions:
        return 0, 0.0
    cached_vectors = np.array([sentiment_dict[q] for q in cached_questions])

    total_hits = 0
    semantic_hits = 0
    total_overlap = 0.0

    for q, ents in zip(questions_list, entities_list):
        if q in exact_cache:
            # Exact match
            total_hits += 1
        else:
            # Semantic search via sentiment cosine similarity fallback
            q_vector = sentiment_dict[q].reshape(1, -1)
            # Avoid divide by zero for zero-vectors in cosine similarity
            if not np.any(q_vector):
                continue

            similarities = cosine_similarity(q_vector, cached_vectors)[0]
            max_sim_idx = np.argmax(similarities)
            if similarities[max_sim_idx] >= threshold:
                total_hits += 1
                semantic_hits += 1

                # Calculate entity overlap (Jaccard similarity) for semantic hit
                matched_cached_q = cached_questions[max_sim_idx]
                cached_ents = cached_entities_dict[matched_cached_q]
                q_ents = set(ents)

                if not q_ents and not cached_ents:
                    total_overlap += 1.0
                elif not q_ents or not cached_ents:
                    total_overlap += 0.0
                else:
                    intersection = len(q_ents.intersection(cached_ents))
                    union = len(q_ents.union(cached_ents))
                    total_overlap += intersection / union

    hit_rate = total_hits / len(questions_list) if questions_list else 0
    avg_overlap = total_overlap / semantic_hits if semantic_hits > 0 else 0.0
    return hit_rate, avg_overlap

def compare_caches(dataset_name, questions_list, entities_list):
    """Runs and logs the cache simulation comparison over different thresholds."""
    logging.info(f"\n── Sentiment Cache Comparison: {dataset_name}  ──")
    logging.info(f"Total Queries: {len(questions_list)} | Unique Queries: {len(set(questions_list))}")

    # Precompute sentiment vectors to save time during simulation
    unique_questions = list(set(questions_list))
    sentiment_dict = {q: get_sentiment_vector(q) for q in unique_questions}

    for size in CACHE_SIZES:
        hit_global = simulate_static_cache(questions_list, size)

        logging.info(f"\nCache Size: {size} (Exact Match Global Hit Rate: {hit_global:.2%})")
        for threshold in SIMILARITY_THRESHOLDS:
            hit_sentiment, avg_overlap = simulate_sentiment_cache(questions_list, entities_list, sentiment_dict, size, threshold)
            diff = hit_sentiment - hit_global
            diff_str = f"+{diff:.2%}" if diff > 0 else f"{diff:.2%}"
            logging.info(
                f"  Sim Threshold >= {threshold:<7} | Sentiment Hit: {hit_sentiment:>6.2%} | Diff: {diff_str} | Avg Sem Hit Entity Overlap: {avg_overlap:.2%}"
            )

def run_webqsp():
    webqsp = load_webqsp()
    questions = extract_questions_from_dataset(webqsp, split='train')
    entities_per_query, _ = extract_entities(webqsp, split='train')
    compare_caches("WebQSP", questions, entities_per_query)

def run_cwq():
    cwq = load_cwq()
    questions = extract_questions_from_dataset(cwq, split='train')
    entities_per_query, _ = extract_entities(cwq, split='train')
    compare_caches("CWQ", questions, entities_per_query)

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
