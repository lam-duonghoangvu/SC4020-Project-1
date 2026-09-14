"""Stage 0 dataset screening (../AGENTS.md): is there cluster structure beyond a null model?

A null model is data with the same marginal properties as the real data but no joint pattern.
Two nulls are provided:
  shuffled_null: each feature column shuffled independently (keeps each feature's distribution).
  uniform_box_null: points uniform in the bounding box of the data (for coordinates).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import silhouette_score

from quake.metrics import NOISE_LABEL

# Project conventions from the LTA screening (../AGENTS.md, "Reading the results").
MIN_SILHOUETTE = 0.26
MIN_SILHOUETTE_RATIO = 2.0


def shuffled_null(X, rng):
    X = np.asarray(X)
    return np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])


def uniform_box_null(X, rng):
    X = np.asarray(X, dtype=float)
    return rng.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)


NULLS = {"shuffled": shuffled_null, "uniform_box": uniform_box_null}


def _kmeans_silhouette(X, k, seed, silhouette_sample):
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    sample = silhouette_sample if silhouette_sample and len(X) > silhouette_sample else None
    return silhouette_score(X, labels, sample_size=sample, random_state=seed)


def silhouette_null_test(X, k_values=range(2, 9), n_nulls=3, seed=0, silhouette_sample=5000, to_features=None):
    """K-means silhouette on real data vs the mean over shuffled nulls, one row per K.

    `to_features` maps the raw columns to the space K-means runs in (e.g. latitude and longitude to
    unit vectors). It is applied after shuffling, so the null shuffles the raw columns and still lies
    in the same space as the real data.

    `passes` is True when the real silhouette is at least MIN_SILHOUETTE_RATIO times the
    null and at least MIN_SILHOUETTE.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    transform = to_features or (lambda data: data)
    features = transform(X)
    nulls = [transform(shuffled_null(X, rng)) for _ in range(n_nulls)]

    rows = []
    for k in k_values:
        real = _kmeans_silhouette(features, k, seed, silhouette_sample)
        null = float(np.mean([_kmeans_silhouette(N, k, seed, silhouette_sample) for N in nulls]))
        rows.append(
            {
                "k": k,
                "real_silhouette": real,
                "null_silhouette": null,
                "ratio": real / null,
                "passes": bool(real >= MIN_SILHOUETTE_RATIO * null and real >= MIN_SILHOUETTE),
            }
        )
    return pd.DataFrame(rows)


def hdbscan_null_test(X, min_cluster_sizes, null="shuffled", metric="euclidean", n_nulls=3, seed=0):
    """HDBSCAN on real data and on each null, one row per (min_cluster_size, data).

    For metric="haversine", X must be [latitude, longitude] in radians.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    datasets = [("real", X)] + [(f"null{i}", NULLS[null](X, rng)) for i in range(n_nulls)]

    rows = []
    for size in min_cluster_sizes:
        for name, data in datasets:
            labels = HDBSCAN(min_cluster_size=size, metric=metric, copy=True).fit_predict(data)
            clustered = labels[labels != NOISE_LABEL]
            rows.append(
                {
                    "min_cluster_size": size,
                    "data": name,
                    "null": null,
                    "n_clusters": len(np.unique(clustered)),
                    "noise_fraction": float((labels == NOISE_LABEL).mean()),
                    "largest_cluster_fraction": np.bincount(clustered).max() / len(labels) if len(clustered) else 0.0,
                }
            )
    return pd.DataFrame(rows)
