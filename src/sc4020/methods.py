"""The clustering methods under comparison: unfitted scikit-learn estimators and a timed runner.

Method names are the ones used in every results table: K-Means, DBSCAN, HDBSCAN, GMM, Spectral,
Agglomerative, Affinity Propagation.
"""

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, AffinityPropagation, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

SEED = 42


@dataclass
class ClusterResult:
    labels: np.ndarray
    model: Any
    params: dict = field(default_factory=dict)
    fit_seconds: float = 0.0


def kmeans(n_clusters, init="k-means++", n_init=10, random_state=SEED):
    """K-Means; init="random" gives plain K-Means, "k-means++" (the default) K-Means++.
    n_init=1 exposes sensitivity to the starting centres (initialisation ablations)."""
    return KMeans(n_clusters=n_clusters, init=init, n_init=n_init, random_state=random_state)


def dbscan(eps, min_samples, metric="euclidean"):
    """DBSCAN. For metric="haversine", X must be [latitude, longitude] in radians and eps in radians."""
    algorithm = "ball_tree" if metric == "haversine" else "auto"
    return DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm)


def hdbscan(min_cluster_size, min_samples=None, metric="euclidean"):
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric, copy=True)


def gmm(n_components, covariance_type="full", n_init=3, random_state=SEED):
    return GaussianMixture(
        n_components=n_components, covariance_type=covariance_type, n_init=n_init, random_state=random_state
    )


def spectral(n_clusters, affinity="rbf", random_state=SEED):
    return SpectralClustering(n_clusters=n_clusters, affinity=affinity, random_state=random_state)


def agglomerative(n_clusters, linkage="ward"):
    return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)


def affinity_propagation(damping=0.5, preference=None, random_state=SEED):
    return AffinityPropagation(damping=damping, preference=preference, random_state=random_state)


FACTORIES = {
    "K-Means": kmeans,
    "DBSCAN": dbscan,
    "HDBSCAN": hdbscan,
    "GMM": gmm,
    "Spectral": spectral,
    "Agglomerative": agglomerative,
    "Affinity Propagation": affinity_propagation,
}
FOUR_METHODS = ["K-Means", "DBSCAN", "HDBSCAN", "GMM"]


def build_models(params: dict[str, dict]) -> dict[str, Any]:
    """One estimator per method from per-method parameters, e.g. `{m: s.params for m, s in selections.items()}`
    from tuning.select_all."""
    return {name: FACTORIES[name](**p) for name, p in params.items()}


def fit_predict_timed(model, X) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    labels = model.fit_predict(X)
    return np.asarray(labels), time.perf_counter() - start


def run_models(
    models: dict[str, Any],
    X: np.ndarray,
    params: dict[str, dict] | None = None,
    inputs: dict[str, np.ndarray] | None = None,
) -> dict[str, ClusterResult]:
    """Fit every model on `X` and time it.

    `params` records each model's settings in its result; `inputs` maps a model name to a
    different feature matrix (e.g. radians for a haversine DBSCAN)."""
    params, inputs = params or {}, inputs or {}
    results = {}
    for name, model in models.items():
        labels, seconds = fit_predict_timed(model, inputs.get(name, X))
        results[name] = ClusterResult(labels, model, dict(params.get(name, {})), seconds)
    return results
