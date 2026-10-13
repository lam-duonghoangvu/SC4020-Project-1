"""Fake-data checks: is there cluster structure beyond a null model?

A null model keeps the data's marginal properties but has no joint pattern (report, Section 3.2
"Fake-data check"):
  shuffled_null     each column shuffled independently, keeping each feature's distribution (tables).
  uniform_box_null  points uniform in the data's bounding box (coordinates).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sc4020 import methods
from sc4020.metrics import NOISE_LABEL, cluster_shares, evaluate

# A dataset passes screening when the real silhouette is at least this high and this many times the null.
MIN_SILHOUETTE = 0.26
MIN_SILHOUETTE_RATIO = 2.0


def shuffled_null(X, rng) -> np.ndarray:
    X = np.asarray(X)
    return np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])


def uniform_box_null(X, rng) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return rng.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)


NULLS = {"shuffled": shuffled_null, "uniform_box": uniform_box_null}


def cluster_counts(labels) -> dict:
    """Clusters, noise fraction and largest cluster fraction for one labelling."""
    shares = cluster_shares(labels)
    return {
        "n_clusters": shares["n_clusters"],
        "noise_fraction": shares["noise_fraction"],
        "largest_cluster_fraction": shares["largest_cluster_share"],
    }


def _kmeans_silhouette(X, k, seed, silhouette_sample):
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    sample = silhouette_sample if silhouette_sample and len(X) > silhouette_sample else None
    return silhouette_score(X, labels, sample_size=sample, random_state=seed)


def silhouette_null_test(X, k_values=range(2, 9), n_nulls=3, seed=0, silhouette_sample=5000, to_features=None) -> pd.DataFrame:
    """K-Means silhouette on real data vs the mean over shuffled nulls, one row per K.

    `to_features` maps the raw columns to the space K-Means runs in (e.g. latitude and longitude
    to unit vectors). It is applied after shuffling, so the null shuffles the raw columns and still
    lies in the same space as the real data. `passes` is True when the real silhouette is at least
    MIN_SILHOUETTE_RATIO times the null and at least MIN_SILHOUETTE."""
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


def hdbscan_null_test(X, min_cluster_sizes, null="shuffled", metric="euclidean", n_nulls=3, seed=0) -> pd.DataFrame:
    """HDBSCAN on real data and on each null, one row per (min_cluster_size, data).

    For metric="haversine", X must be [latitude, longitude] in radians."""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    datasets = [("real", X)] + [(f"null{i}", NULLS[null](X, rng)) for i in range(n_nulls)]
    rows = []
    for size in min_cluster_sizes:
        for name, data in datasets:
            labels = methods.hdbscan(size, metric=metric).fit_predict(data)
            rows.append({"min_cluster_size": size, "data": name, "null": null, **cluster_counts(labels)})
    return pd.DataFrame(rows)


def single_value_share(labels, values) -> pd.Series:
    """For each cluster, the share of its members holding the cluster's most common value.

    A share of 1.0 means the cluster is a single discrete value (e.g. one storey band), which
    suggests HDBSCAN found a tie in the data rather than a group of similar entities."""
    frame = pd.DataFrame({"label": np.asarray(labels), "value": np.asarray(values)})
    frame = frame[frame["label"] != NOISE_LABEL]
    return frame.groupby("label")["value"].agg(lambda v: v.value_counts(normalize=True).iloc[0])


def null_gap(X, fit, null="shuffled", n_nulls=3, seed=0, y_true=None) -> dict:
    """Run `fit` (X -> labels) on the real data and on `n_nulls` fake datasets.

    Returns the real scores and the mean silhouette, cluster count and noise on the fakes."""
    rng = np.random.default_rng(seed)
    fakes = [NULLS[null](X, rng) for _ in range(n_nulls)]
    real = evaluate(X, fit(X), y_true)
    fake = [evaluate(Z, fit(Z)) for Z in fakes]
    sil = [f["silhouette"] for f in fake]
    null_sil = float(np.nanmean(sil)) if np.isfinite(sil).any() else np.nan
    return {
        "null": null,
        "real_silhouette": real["silhouette"],
        "real_clusters": real["n_clusters"],
        "real_noise": real["noise_fraction"],
        "null_silhouette": null_sil,
        "null_clusters_mean": float(np.mean([f["n_clusters"] for f in fake])),
        "null_noise_mean": float(np.mean([f["noise_fraction"] for f in fake])),
        "gap": real["silhouette"] - null_sil,
    }
