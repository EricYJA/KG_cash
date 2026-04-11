# characterization/main.py
import argparse
import logging
import resource
from collections import Counter

from src.load_data import load_webqsp, load_cwq, inspect_sample, extract_entities
from src.entity_stats import compute_redundancy_stats
from src.subgraph_analysis import analyze_subgraph_overlap
from src.clustering import (
    global_entity_clustering_kmeans,
    global_entity_clustering_hdbscan
)

# Configure logging to write to both a file and the console
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("cache_comparison.log", mode="w"),
        logging.StreamHandler()
    ]
)

def limit_memory():
    max_memory_gb = 26
    max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))

# Apply the memory limit immediately
limit_memory()

# Configure logging to write to both a file and the console
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("cache_comparison.log", mode="w"),
        logging.StreamHandler()
    ]
)

CACHE_SIZES = [10, 50, 100, 200, 500, 1000]


def simulate_static_cache(items_list, cache_size):
    """Simulates a global static cache of top `cache_size` frequent items."""
    counts = Counter(items_list)
    cache = set(item for item, _ in counts.most_common(cache_size))
    hits = sum(1 for item in items_list if item in cache)
    return hits / len(items_list) if items_list else 0


def simulate_clustered_cache(items_list, item_clusters, cache_size):
    """
    Simulates a clustered cache. Populates the cache with the globally most frequent
    items up to `cache_size`. If an exact cache miss occurs, falls back to a semantic
    hit if any entity from the same cluster is already in the cache.
    """
    # Count global frequencies
    counts = Counter(items_list)

    # Map each unique item to its cluster ID
    item_to_cluster = dict(zip(items_list, item_clusters))

    cache = set()
    cached_clusters = set()

    # Populate cache with globally most frequent items
    for item, _ in counts.most_common(cache_size):
        cache.add(item)
        cluster_id = item_to_cluster[item]
        if cluster_id != -1:
            cached_clusters.add(cluster_id)

    total_hits = 0
    for item, cluster_id in zip(items_list, item_clusters):
        if item in cache:
            # Primary mechanism: Exact match
            total_hits += 1
        elif cluster_id != -1 and cluster_id in cached_clusters:
            # Secondary mechanism: Semantic match from the same cluster
            total_hits += 1

    return total_hits / len(items_list) if items_list else 0


def compare_caches(dataset_name, flat_items, entity_to_cluster, cache_sizes):
    """Runs and logs the cache simulation comparison."""
    logging.info(f"\n── Cache Comparison (Global vs. Semantic Clustered): {dataset_name}  ──")

    # Map each item occurrence in the dataset to its global entity cluster
    item_clusters = [entity_to_cluster.get(item, -1) for item in flat_items]

    for size in cache_sizes:
        hit_global = simulate_static_cache(flat_items, size)
        hit_clustered = simulate_clustered_cache(flat_items, item_clusters, size)

        diff = hit_clustered - hit_global
        diff_str = f"+{diff:.2%}" if diff > 0 else f"{diff:.2%}"
        logging.info(
            f"  Size {size:<4} | Global: {hit_global:>6.2%} | Clustered: {hit_clustered:>6.2%} | Diff: {diff_str}")


def run_webqsp():
    webqsp = load_webqsp()
    # inspect_sample(webqsp, split='train', index=0)

    _, webqsp_flat = extract_entities(webqsp, split='train')

    # Cast triplets to string representations to ensure predictability and hashability
    webqsp_flat_str = [str(ent) for ent in webqsp_flat]
    unique_entities = list(set(webqsp_flat_str))

    print("\n── Entity Redundancy Stats: WebQSP ──")
    webqsp_stats = compute_redundancy_stats(webqsp_flat_str, "WebQSP")

    print("\n── Subgraph Analysis: WebQSP ──")
    webqsp_sg = analyze_subgraph_overlap(webqsp, split='train')
    webqsp_stats['subgraph_analysis'] = webqsp_sg

    print("\n── Clustering Analysis (K-Means): WebQSP ──")
    kmeans_results = global_entity_clustering_kmeans(unique_entities)

    # Run Cache comparisons for each KMeans cluster size
    for k, result in kmeans_results.items():
        entity_to_cluster = {}
        for label, entities in result["clusters"].items():
            for ent in entities:
                entity_to_cluster[ent] = label
        compare_caches(f"WebQSP (KMeans k={k})", webqsp_flat_str, entity_to_cluster, CACHE_SIZES)

    print("\n── Clustering Analysis (HDBSCAN): WebQSP ──")
    hdbscan_results = global_entity_clustering_hdbscan(unique_entities)

    # Build entity to cluster mapping using HDBSCAN results for caching simulation
    entity_to_cluster = {}
    for label, entities in hdbscan_results["clusters"].items():
        for ent in entities:
            entity_to_cluster[ent] = label

    # Run Cache comparisons for HDBSCAN
    compare_caches("WebQSP (HDBSCAN)", webqsp_flat_str, entity_to_cluster, CACHE_SIZES)

    webqsp_stats['clustering_analysis'] = {
        'kmeans': kmeans_results,
        'hdbscan': hdbscan_results
    }

    return webqsp_stats


def run_cwq():
    cwq = load_cwq()
    # inspect_sample(cwq, split='train', index=0)

    _, cwq_flat = extract_entities(cwq, split='train')

    # Cast triplets to string representations to ensure predictability and hashability
    cwq_flat_str = [str(ent) for ent in cwq_flat]
    unique_entities = list(set(cwq_flat_str))

    print("\n── Entity Redundancy Stats: CWQ ──")
    cwq_stats = compute_redundancy_stats(cwq_flat_str, "CWQ")

    print("\n── Subgraph Analysis: CWQ ──")
    cwq_sg = analyze_subgraph_overlap(cwq, split='train')
    cwq_stats['subgraph_analysis'] = cwq_sg

    print("\n── Clustering Analysis (K-Means): CWQ ──")
    kmeans_results = global_entity_clustering_kmeans(unique_entities)

    # Run Cache comparisons for each KMeans cluster size
    for k, result in kmeans_results.items():
        entity_to_cluster = {}
        for label, entities in result["clusters"].items():
            for ent in entities:
                entity_to_cluster[ent] = label
        compare_caches(f"CWQ (KMeans k={k})", cwq_flat_str, entity_to_cluster, CACHE_SIZES)

    print("\n── Clustering Analysis (HDBSCAN): CWQ ──")
    hdbscan_results = global_entity_clustering_hdbscan(unique_entities)

    # Build entity to cluster mapping using HDBSCAN results for caching simulation
    entity_to_cluster = {}
    for label, entities in hdbscan_results["clusters"].items():
        for ent in entities:
            entity_to_cluster[ent] = label

    # Run Cache comparisons for HDBSCAN
    compare_caches("CWQ (HDBSCAN)", cwq_flat_str, entity_to_cluster, CACHE_SIZES)

    cwq_stats['clustering_analysis'] = {
        'kmeans': kmeans_results,
        'hdbscan': hdbscan_results
    }

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
