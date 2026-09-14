"""Runs all 5 clustering methods on a dataset with unsupervised hyperparameter selection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN, DBSCAN, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

from sc4020 import hyperparams

METHOD_NAMES = ["KMeans++", "DBSCAN", "HDBSCAN", "GMM", "Spectral Clustering"]


@dataclass
class ClusterResult:
    labels: np.ndarray
    model: Any
    params: dict
    fit_seconds: float


def _timed_fit_predict(fn) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    labels = fn()
    elapsed = time.perf_counter() - start
    return labels, elapsed


def run_all_methods(
    X: np.ndarray,
    n_true_clusters: int | None,
    spectral_affinity: str = "rbf",
    random_state: int = 42,
    k_selection: hyperparams.KSelectionResult | None = None,
) -> dict[str, ClusterResult]:
    """Fit all 5 methods on X.

    `n_true_clusters` sets K for KMeans/GMM/Spectral when ground truth exists;
    otherwise `k_selection` (from `hyperparams.select_k_by_silhouette`) is used.
    """
    if n_true_clusters is not None:
        k = n_true_clusters
    elif k_selection is not None:
        k = k_selection.best_k
    else:
        raise ValueError("Either n_true_clusters or k_selection must be provided")

    results: dict[str, ClusterResult] = {}

    # 1. K-Means++
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
    labels, secs = _timed_fit_predict(lambda: model.fit_predict(X))
    results["KMeans++"] = ClusterResult(labels, model, {"n_clusters": k}, secs)

    # 2. DBSCAN
    dbscan_tune = hyperparams.tune_dbscan(X)
    model = DBSCAN(eps=dbscan_tune.eps, min_samples=dbscan_tune.min_samples)
    labels, secs = _timed_fit_predict(lambda: model.fit_predict(X))
    results["DBSCAN"] = ClusterResult(
        labels, model, {"eps": dbscan_tune.eps, "min_samples": dbscan_tune.min_samples}, secs
    )

    # 3. HDBSCAN
    hdbscan_tune = hyperparams.tune_hdbscan(X)
    model = HDBSCAN(min_cluster_size=hdbscan_tune.min_cluster_size)
    labels, secs = _timed_fit_predict(lambda: model.fit_predict(X))
    results["HDBSCAN"] = ClusterResult(
        labels, model, {"min_cluster_size": hdbscan_tune.min_cluster_size}, secs
    )

    # 4. GMM
    model = GaussianMixture(n_components=k, random_state=random_state)
    labels, secs = _timed_fit_predict(lambda: model.fit(X).predict(X))
    results["GMM"] = ClusterResult(labels, model, {"n_components": k}, secs)

    # 5. Spectral Clustering
    model = SpectralClustering(
        n_clusters=k, affinity=spectral_affinity, random_state=random_state
    )
    labels, secs = _timed_fit_predict(lambda: model.fit_predict(X))
    results["Spectral Clustering"] = ClusterResult(
        labels, model, {"n_clusters": k, "affinity": spectral_affinity}, secs
    )

    return results
