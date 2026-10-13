"""Stage 0 dataset screening (../AGENTS.md): is there cluster structure beyond a null model?

The null model shuffles each feature column independently: it keeps each feature's distribution
and destroys the joint pattern.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import silhouette_score

NOISE_LABEL = -1

# Project conventions from the LTA screening (../AGENTS.md, "Reading the results").
MIN_SILHOUETTE = 0.26
MIN_SILHOUETTE_RATIO = 2.0


def shuffled_null(X, rng):
    X = np.asarray(X)
    return np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])


def _kmeans_silhouette(X, k, seed, silhouette_sample):
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    sample = silhouette_sample if silhouette_sample and len(X) > silhouette_sample else None
    return silhouette_score(X, labels, sample_size=sample, random_state=seed)


def silhouette_null_test(X, k_values=range(2, 9), n_nulls=3, seed=0, silhouette_sample=5000):
    """K-means silhouette on real data vs the mean over shuffled nulls, one row per K.

    `passes` is True when the real silhouette is at least MIN_SILHOUETTE_RATIO times the
    null and at least MIN_SILHOUETTE.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    nulls = [shuffled_null(X, rng) for _ in range(n_nulls)]

    rows = []
    for k in k_values:
        real = _kmeans_silhouette(X, k, seed, silhouette_sample)
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


def cluster_counts(labels):
    """Number of clusters, noise fraction and largest cluster fraction for one labelling."""
    labels = np.asarray(labels)
    clustered = labels[labels != NOISE_LABEL]
    return {
        "n_clusters": len(np.unique(clustered)),
        "noise_fraction": float((labels == NOISE_LABEL).mean()),
        "largest_cluster_fraction": np.bincount(clustered).max() / len(labels) if len(clustered) else 0.0,
    }


def hdbscan_null_test(X, min_cluster_sizes, n_nulls=3, seed=0):
    """HDBSCAN on real data and on each shuffled null, one row per (min_cluster_size, data)."""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    datasets = [("real", X)] + [(f"null{i}", shuffled_null(X, rng)) for i in range(n_nulls)]

    rows = []
    for size in min_cluster_sizes:
        for name, data in datasets:
            labels = HDBSCAN(min_cluster_size=size, copy=True).fit_predict(data)
            rows.append({"min_cluster_size": size, "data": name, **cluster_counts(labels)})
    return pd.DataFrame(rows)


def single_value_share(labels, values):
    """For each cluster, the share of its members holding the cluster's most common value.

    A share of 1.0 means the cluster is a single discrete value (e.g. one storey band), which
    suggests HDBSCAN found a tie in the data rather than a group of similar entities.
    """
    frame = pd.DataFrame({"label": np.asarray(labels), "value": np.asarray(values)})
    frame = frame[frame["label"] != NOISE_LABEL]
    return frame.groupby("label")["value"].agg(lambda v: v.value_counts(normalize=True).iloc[0])
