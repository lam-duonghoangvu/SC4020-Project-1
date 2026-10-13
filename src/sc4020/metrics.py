"""Cluster validity metrics and distances on the sphere.

Internal scores (silhouette, Davies-Bouldin, Calinski-Harabasz) are computed on clustered points
only, so they say nothing about points a density method calls noise; the noise fraction and the
largest cluster's share are reported next to them. External scores (ARI, NMI, AMI, Fowlkes-Mallows)
use every point, with noise as one more cluster, so discarding hard points cannot raise them.
"""

import itertools

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import BallTree

from sc4020.datasets.preprocess_dataset import EARTH_RADIUS_KM, to_radians

NOISE_LABEL = -1
SILHOUETTE_SAMPLE = 5000


def silhouette(X, labels, sample_size=SILHOUETTE_SAMPLE, seed=0, metric="euclidean") -> float:
    """Silhouette on a random subsample of `sample_size` points, or on all points when the data is
    smaller or the subsample would miss every cluster but one (one large cluster plus tiny ones)."""
    X, labels = np.asarray(X), np.asarray(labels)
    if sample_size and len(labels) > sample_size:
        rows = np.random.default_rng(seed).choice(len(labels), size=sample_size, replace=False)
        if len(np.unique(labels[rows])) >= 2:
            X, labels = X[rows], labels[rows]
    return float(silhouette_score(X, labels, metric=metric))


def cluster_shares(labels) -> dict:
    """Clusters found (noise excluded), noise fraction and the largest cluster's share of all points."""
    labels = np.asarray(labels)
    clustered = labels[labels != NOISE_LABEL]
    return {
        "n_clusters": int(len(np.unique(clustered))),
        "noise_fraction": float((labels == NOISE_LABEL).mean()),
        "largest_cluster_share": float(np.bincount(clustered - clustered.min()).max() / len(labels)) if len(clustered) else 0.0,
    }


def evaluate(
    X,
    labels,
    y_true=None,
    *,
    silhouette_sample=SILHOUETTE_SAMPLE,
    seed=0,
    metric="euclidean",
    X_davies_bouldin=None,
) -> dict:
    """Score one clustering.

    Internal scores are NaN unless at least 2 clusters remain and there are more clustered points
    than clusters. `metric` is passed to the silhouette (e.g. "haversine" with X in radians);
    Davies-Bouldin needs centroids in a Euclidean space, so `X_davies_bouldin` can supply one
    (e.g. unit vectors for points on the sphere)."""
    X, labels = np.asarray(X), np.asarray(labels)
    X_db = X if X_davies_bouldin is None else np.asarray(X_davies_bouldin)
    result = {**cluster_shares(labels), "silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    clustered = labels != NOISE_LABEL
    n_clusters = result["n_clusters"]
    if n_clusters >= 2 and clustered.sum() > n_clusters:
        lab = labels[clustered]
        result["silhouette"] = silhouette(X[clustered], lab, silhouette_sample, seed, metric)
        result["davies_bouldin"] = float(davies_bouldin_score(X_db[clustered], lab))
        result["calinski_harabasz"] = float(calinski_harabasz_score(X_db[clustered], lab))
    if y_true is not None:
        result.update(
            ari=float(adjusted_rand_score(y_true, labels)),
            nmi=float(normalized_mutual_info_score(y_true, labels)),
            ami=float(adjusted_mutual_info_score(y_true, labels)),
            fmi=float(fowlkes_mallows_score(y_true, labels)),
        )
    return result


def pairwise_ari(label_runs) -> np.ndarray:
    """ARI for every pair of clusterings, e.g. the same method run with different seeds."""
    pairs = list(itertools.combinations(label_runs, 2))
    if not pairs:
        raise ValueError("Need at least two clusterings to compare")
    return np.array([adjusted_rand_score(a, b) for a, b in pairs])


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between points given in degrees."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def nearest_neighbour_km(lat, lon, k=1):
    """Great-circle distance in km from each point to its k-th nearest other point."""
    X = to_radians(lat, lon)
    distances, _ = BallTree(X, metric="haversine").query(X, k=k + 1)
    return distances[:, k] * EARTH_RADIUS_KM
