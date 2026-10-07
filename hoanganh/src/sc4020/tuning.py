"""Label-free hyperparameter selection for K-Means, GMM, DBSCAN and HDBSCAN.

Ground-truth labels are never used here: in real use the settings have to be
chosen without knowing the true clusters, so labels are kept for evaluation only.

  K-Means  k by silhouette.
  GMM      k and covariance type by BIC (the model's own likelihood criterion).
  DBSCAN   min_samples from the dimension, eps around the k-distance knee,
           then the pair with the best DBCV.
  HDBSCAN  min_cluster_size and min_samples with the best DBCV.

Density methods are scored with DBCV (see dbcv.py) rather than silhouette:
silhouette assumes compact, convex clusters, so it steers density methods
towards merging or chopping up non-convex shapes, the very case they are for.

For the density methods, a setting is rejected when it finds fewer than 2 or
more than `max_clusters` clusters (the same range K-Means searches), or labels
more than `max_noise` of the points as noise. When no setting passes, the one with
the least noise is kept and the selection is marked `valid=False`.

`k` fixes the number of clusters for K-Means and GMM (e.g. the known number
of classes); the label-free choice is still recorded in `label_free`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from sc4020.dbcv import dbcv_score
from sc4020.metrics import NOISE_LABEL

K_RANGE = range(2, 11)
COVARIANCE_TYPES = ("full", "tied", "diag", "spherical")
EPS_FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
MIN_CLUSTER_SIZES = (5, 10, 15, 20, 30, 50, 100, 200, 400)
HDBSCAN_MIN_SAMPLES = (None, 5, 10)  # None: same as min_cluster_size


@dataclass
class Selection:
    """The chosen parameters plus every setting that was tried."""

    params: dict
    score: float
    tried: list[dict] = field(default_factory=list)
    valid: bool = True
    label_free: dict | None = None


def _silhouette(X: np.ndarray, labels: np.ndarray, sample_size: int | None, random_state: int) -> float:
    if sample_size is not None and sample_size >= len(X):
        sample_size = None
    return float(silhouette_score(X, labels, sample_size=sample_size, random_state=random_state))


def _density_trial(
    X: np.ndarray, labels: np.ndarray, max_clusters: int, max_noise: float, sample_size: int | None
) -> dict:
    """Cluster count, noise fraction and DBCV (NaN when the setting is rejected)."""
    mask = labels != NOISE_LABEL
    n_clusters = len(set(labels[mask]))
    noise = float((~mask).mean())
    ok = 2 <= n_clusters <= max_clusters and noise <= max_noise
    score = dbcv_score(X, labels, sample_size) if ok else np.nan
    return {"n_clusters": n_clusters, "noise_fraction": noise, "score": score}


def _pick_density(tried: list[dict], keys: tuple[str, ...]) -> Selection:
    passed = [t for t in tried if not np.isnan(t["score"])]
    if passed:
        best, valid = max(passed, key=lambda t: t["score"]), True
    else:
        fallback = [t for t in tried if t["n_clusters"] >= 2] or tried
        best, valid = min(fallback, key=lambda t: t["noise_fraction"]), False
    return Selection({k: best[k] for k in keys}, best["score"], tried, valid)


def select_kmeans(
    X: np.ndarray,
    k: int | None = None,
    k_range: Sequence[int] = K_RANGE,
    sample_size: int | None = None,
    random_state: int = 42,
) -> Selection:
    tried = []
    for n in k_range:
        labels = KMeans(n_clusters=n, n_init=10, random_state=random_state).fit_predict(X)
        tried.append({"n_clusters": n, "score": _silhouette(X, labels, sample_size, random_state)})
    label_free = max(tried, key=lambda t: t["score"])
    best = label_free if k is None else next(t for t in tried if t["n_clusters"] == k)
    return Selection(
        {"n_clusters": best["n_clusters"]}, best["score"], tried,
        label_free={"n_clusters": label_free["n_clusters"]},
    )


def select_gmm(
    X: np.ndarray,
    k: int | None = None,
    k_range: Sequence[int] = K_RANGE,
    covariance_types: tuple[str, ...] = COVARIANCE_TYPES,
    random_state: int = 42,
) -> Selection:
    """Lowest BIC over (k, covariance type); with `k` fixed, only the covariance
    type is chosen. The score stored is -BIC so higher is better."""
    tried = []
    for cov in covariance_types:
        for n in k_range:
            gmm = GaussianMixture(
                n_components=n, covariance_type=cov, n_init=3, random_state=random_state
            ).fit(X)
            tried.append({"n_components": n, "covariance_type": cov, "score": -gmm.bic(X)})
    keys = ("n_components", "covariance_type")
    label_free = max(tried, key=lambda t: t["score"])
    best = label_free if k is None else max(
        (t for t in tried if t["n_components"] == k), key=lambda t: t["score"]
    )
    return Selection(
        {c: best[c] for c in keys}, best["score"], tried,
        label_free={c: label_free[c] for c in keys},
    )


def k_distance_knee(X: np.ndarray, min_samples: int) -> float:
    """Distance to the `min_samples`-th neighbour at the knee of the sorted curve.

    The knee is the point furthest from the straight line joining the two ends
    of the curve, which is less noisy than a second derivative on large data.
    """
    distances, _ = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)
    curve = np.sort(distances[:, -1])
    x = np.linspace(0.0, 1.0, len(curve))
    y = (curve - curve[0]) / max(curve[-1] - curve[0], 1e-12)
    return float(curve[int(np.argmax(x - y))])


def default_min_samples(n_features: int) -> int:
    return int(np.clip(2 * n_features, 4, 20))


def select_dbscan(
    X: np.ndarray,
    max_clusters: int = max(K_RANGE),
    max_noise: float = 0.5,
    sample_size: int | None = None,
) -> Selection:
    base = default_min_samples(X.shape[1])
    tried = []
    for min_samples in sorted({max(3, base // 2), base, base * 2, base * 4}):
        knee = k_distance_knee(X, min_samples)
        for factor in EPS_FACTORS:
            eps = knee * factor
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
            trial = _density_trial(X, labels, max_clusters, max_noise, sample_size)
            tried.append({"eps": eps, "min_samples": min_samples, **trial})
            # Clusters only merge as eps grows, so once everything is one cluster
            # every larger eps is too (and those are the memory-hungry settings).
            if trial["n_clusters"] == 1 and trial["noise_fraction"] < max_noise:
                break
    return _pick_density(tried, ("eps", "min_samples"))


def select_hdbscan(
    X: np.ndarray,
    max_clusters: int = max(K_RANGE),
    max_noise: float = 0.5,
    sample_size: int | None = None,
) -> Selection:
    cap = max(5, len(X) // 10)
    tried = []
    for mcs in (c for c in MIN_CLUSTER_SIZES if c <= cap):
        for min_samples in HDBSCAN_MIN_SAMPLES:
            labels = HDBSCAN(min_cluster_size=mcs, min_samples=min_samples).fit_predict(X)
            trial = _density_trial(X, labels, max_clusters, max_noise, sample_size)
            tried.append({"min_cluster_size": mcs, "min_samples": min_samples, **trial})
    return _pick_density(tried, ("min_cluster_size", "min_samples"))


def select_all(
    X: np.ndarray,
    k: int | None = None,
    k_range: Sequence[int] = K_RANGE,
    sample_size: int | None = None,
    dbcv_sample_size: int | None = None,
    random_state: int = 42,
) -> dict[str, Selection]:
    """Selections keyed by the method names used in `clustering.four_models`.

    Every method gets the same cluster budget: K-Means and GMM search
    `k_range`, and the density methods may find at most `max(k_range)` clusters.
    `sample_size` and `dbcv_sample_size` estimate silhouette and DBCV on a
    sample for large data.
    """
    return {
        "K-Means": select_kmeans(X, k, k_range, sample_size=sample_size, random_state=random_state),
        "DBSCAN": select_dbscan(X, max_clusters=max(k_range), sample_size=dbcv_sample_size),
        "HDBSCAN": select_hdbscan(X, max_clusters=max(k_range), sample_size=dbcv_sample_size),
        "GMM": select_gmm(X, k, k_range, random_state=random_state),
    }
