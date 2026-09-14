"""Evaluation metrics for clusterings of earthquake epicentres.

Silhouette uses great-circle distance (haversine on radians) for every method, so scores are comparable
across methods. Davies-Bouldin needs cluster centroids in a Euclidean space, so it uses 3D unit vectors,
where chord length increases with great-circle distance (quake/features.py).
"""

import itertools
import time

import numpy as np
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score

from quake.features import nearest_neighbour_km

NOISE_LABEL = -1


def timed_fit_predict(model, X):
    start = time.perf_counter()
    labels = model.fit_predict(X)
    return labels, time.perf_counter() - start


def pairwise_ari(label_runs):
    """ARI for every pair of clusterings, e.g. the same method run with different seeds."""
    pairs = list(itertools.combinations(label_runs, 2))
    if not pairs:
        raise ValueError("Need at least two clusterings to compare")
    return np.array([adjusted_rand_score(a, b) for a, b in pairs])


def k_distances_km(lat, lon, min_samples):
    """Sorted great-circle distance (km) from each point to its (min_samples - 1)-th nearest other point.

    A point is a DBSCAN core point when this distance is at most eps (DBSCAN counts the point itself
    towards min_samples), so the curve suggests eps.
    """
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2")
    return np.sort(nearest_neighbour_km(lat, lon, k=min_samples - 1))


def best_density_setting(grid, max_noise=0.5):
    """Grid row with the highest silhouette among real-data settings with at least 2 clusters and at most
    `max_noise` of points as noise. Returns None when no setting qualifies.

    The noise cap matters because silhouette ignores noise: without it, the best-scoring setting is
    one that discards most of the data (../AGENTS.md, lessons).
    """
    candidates = grid[(grid["n_clusters"] >= 2) & (grid["noise_fraction"] <= max_noise)]
    if "data" in candidates.columns:
        candidates = candidates[candidates["data"] == "real"]
    return None if candidates.empty else candidates.sort_values("silhouette", ascending=False).iloc[0]


def evaluate(radians, unit_vectors, labels, silhouette_sample=None, seed=0):
    """Score a clustering of points on the sphere.

    Silhouette (haversine on `radians`) and Davies-Bouldin (on `unit_vectors`) are computed on
    non-noise points only, and are NaN when fewer than two clusters remain.
    """
    labels = np.asarray(labels)
    clustered = labels != NOISE_LABEL
    n_clusters = len(np.unique(labels[clustered]))

    result = {
        "n_clusters": n_clusters,
        "noise_fraction": float((~clustered).mean()),
        "largest_cluster_share": float(np.bincount(labels[clustered]).max() / len(labels)) if n_clusters else 0.0,
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
    }
    if n_clusters >= 2 and clustered.sum() > n_clusters:
        labels_c = labels[clustered]
        sample = min(silhouette_sample, len(labels_c)) if silhouette_sample else None
        result["silhouette"] = silhouette_score(
            np.asarray(radians)[clustered], labels_c, metric="haversine", sample_size=sample, random_state=seed
        )
        result["davies_bouldin"] = davies_bouldin_score(np.asarray(unit_vectors)[clustered], labels_c)
    return result
