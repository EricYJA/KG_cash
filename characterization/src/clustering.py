# characterization/src/clustering.py
import math

import numpy as np
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sentence_transformers import SentenceTransformer


def evaluate_clusters(features, labels):
    """
    Evaluates clustering performance using internal metrics.
    Ignores noise points (-1) which are typical in HDBSCAN.
    """
    mask = labels != -1
    filtered_features = features[mask]
    filtered_labels = labels[mask]

    if len(set(filtered_labels)) < 2:
        return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}

    metrics = {
        "silhouette": silhouette_score(filtered_features, filtered_labels),
        "davies_bouldin": davies_bouldin_score(filtered_features, filtered_labels),
        "calinski_harabasz": calinski_harabasz_score(filtered_features, filtered_labels)
    }
    return metrics


def global_entity_clustering_kmeans(unique_entities, cluster_sizes=None):
    """
    Clusters the global unique entities using KMeans and semantic embeddings.
    """
    if cluster_sizes is None:
        cluster_sizes = list(range(int(math.sqrt(len(unique_entities))), len(unique_entities)//2+1, 100))


    if not unique_entities:
        return {}

    entities_str = [str(ent) for ent in unique_entities]
    print(f"Clustering {len(entities_str)} unique global entities using KMeans.")

    print("Loading embedding model...")
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(entities_str, show_progress_bar=True)

    results = {}
    for num_clusters in cluster_sizes:
        # Ensure we don't ask for more clusters than we have entities
        k = min(num_clusters, len(unique_entities))
        if k < 2:
            continue

        print(f"Clustering into {k} global entity clusters using KMeans...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(embeddings)

        metrics = evaluate_clusters(embeddings, labels)
        print(f"KMeans (k={k}) Metrics: {metrics}")

        cluster_map = {i: [] for i in range(k)}
        for entity_val, label in zip(unique_entities, labels):
            cluster_map[label].append(entity_val)

        results[k] = {"labels": labels, "clusters": cluster_map, "metrics": metrics}

    return results


def global_entity_clustering_hdbscan(unique_entities, min_cluster_size=5):
    """
    Clusters the global unique entities using HDBSCAN and semantic embeddings.
    """
    if not unique_entities:
        return {
            "labels": [],
            "clusters": {},
            "metrics": {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
        }

    entities_str = [str(ent) for ent in unique_entities]
    print(f"Clustering {len(entities_str)} unique global entities using HDBSCAN.")

    print("Loading embedding model...")
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(entities_str, show_progress_bar=True)

    print(f"Running HDBSCAN on global entities...")
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, n_jobs=-1)
    labels = hdbscan_model.fit_predict(embeddings)

    metrics = evaluate_clusters(embeddings, labels)
    print(f"HDBSCAN Metrics: {metrics}")

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {num_clusters} global clusters (excluding noise).")

    cluster_map = {label: [] for label in set(labels)}
    for entity_val, label in zip(unique_entities, labels):
        cluster_map[label].append(entity_val)

    return {"labels": labels, "clusters": cluster_map, "metrics": metrics}
