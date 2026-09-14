"""Evaluation metrics for clusterings of resale transactions (Euclidean, standardized features)."""

import itertools
import time

import numpy as np
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from hdb.screening import NOISE_LABEL


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


def k_distances(X, min_samples):
    """Sorted distance from each point to its (min_samples - 1)-th nearest other point.

    A point is a DBSCAN core point when this distance is at most eps (DBSCAN counts the point itself
    towards min_samples), so the curve suggests eps.
    """
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2")
    distances, _ = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)
    return np.sort(distances[:, -1])


def knee_distance(sorted_distances):
    """Value at the knee of an increasing curve: the point furthest below the straight line from the
    first to the last value, with both axes scaled to [0, 1]."""
    y = np.sort(np.asarray(sorted_distances, dtype=float))
    span = y[-1] - y[0]
    if span == 0:
        return float(y[0])
    x = np.linspace(0, 1, len(y))
    return float(y[np.argmax(x - (y - y[0]) / span)])


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


def _silhouette(X, labels, silhouette_sample, seed):
    """Silhouette on a random subsample of `silhouette_sample` points, or on all points when the data is
    smaller or the subsample would miss every cluster but one (possible when one cluster is large and
    the rest are tiny)."""
    if silhouette_sample and len(labels) > silhouette_sample:
        rows = np.random.default_rng(seed).choice(len(labels), size=silhouette_sample, replace=False)
        if len(np.unique(labels[rows])) >= 2:
            X, labels = X[rows], labels[rows]
    return silhouette_score(X, labels)


def evaluate(X, labels, truth=None, silhouette_sample=None, seed=0):
    """Score one clustering.

    Silhouette and Davies-Bouldin are computed on non-noise points only, and are NaN when fewer than
    two clusters remain. ARI and NMI against `truth` use every point, with noise as its own group.
    """
    X = np.asarray(X)
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
        X_c, labels_c = X[clustered], labels[clustered]
        result["silhouette"] = _silhouette(X_c, labels_c, silhouette_sample, seed)
        result["davies_bouldin"] = davies_bouldin_score(X_c, labels_c)
    if truth is not None:
        result["ari"] = adjusted_rand_score(truth, labels)
        result["nmi"] = normalized_mutual_info_score(truth, labels)
    return result
