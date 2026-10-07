"""Model factories and a runner for the clustering comparison notebooks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import ClusterMixin
from sklearn.cluster import (
    AffinityPropagation,
    AgglomerativeClustering,
    DBSCAN,
    HDBSCAN,
    KMeans,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture

EARTH_RADIUS_KM = 6371.0088


@dataclass
class ClusterResult:
    labels: np.ndarray
    model: Any
    fit_seconds: float


def wine_models() -> dict[str, ClusterMixin]:
    return {
        "K-Means (k=3)": KMeans(n_clusters=3, random_state=42, n_init=10),
        "Agglomerative (k=3)": AgglomerativeClustering(n_clusters=3, linkage="ward"),
        "GMM (k=3)": GaussianMixture(n_components=3, random_state=42),
        "Spectral (k=3)": SpectralClustering(
            n_clusters=3, affinity="nearest_neighbors", random_state=42
        ),
        "DBSCAN (eps=2.3)": DBSCAN(eps=2.3, min_samples=4),
        "Affinity Propagation": AffinityPropagation(damping=0.8, random_state=42),
    }


def breast_cancer_models() -> dict[str, ClusterMixin]:
    return {
        "K-Means (k=2)": KMeans(n_clusters=2, random_state=42, n_init=10),
        "Agglomerative (k=2)": AgglomerativeClustering(n_clusters=2, linkage="ward"),
        "GMM (k=2)": GaussianMixture(n_components=2, random_state=42),
        "Spectral (k=2)": SpectralClustering(
            n_clusters=2, affinity="nearest_neighbors", random_state=42
        ),
        "DBSCAN (eps=4.5)": DBSCAN(eps=4.5, min_samples=5),
        "Affinity Propagation": AffinityPropagation(
            damping=0.9, preference=-150, random_state=42
        ),
    }


def uber_models(radius_km: float = 0.35) -> dict[str, ClusterMixin]:
    eps_rad = radius_km / EARTH_RADIUS_KM
    return {
        "DBSCAN (Haversine ~350m)": DBSCAN(
            eps=eps_rad, min_samples=30, metric="haversine", algorithm="ball_tree"
        ),
        "K-Means (k=5)": KMeans(n_clusters=5, random_state=42, n_init=10),
        "GMM (k=5)": GaussianMixture(n_components=5, random_state=42),
    }


def four_models(params: dict[str, dict], random_state: int = 42) -> dict[str, ClusterMixin]:
    """K-Means, DBSCAN, HDBSCAN and GMM built from per-method parameters
    (e.g. `{name: selection.params}` from `tuning.select_all`)."""
    return {
        "K-Means": KMeans(**params["K-Means"], n_init=10, random_state=random_state),
        "DBSCAN": DBSCAN(**params["DBSCAN"]),
        "HDBSCAN": HDBSCAN(**params["HDBSCAN"]),
        "GMM": GaussianMixture(**params["GMM"], n_init=3, random_state=random_state),
    }


def run_models(
    models: dict[str, ClusterMixin],
    X: np.ndarray,
    inputs: dict[str, np.ndarray] | None = None,
) -> dict[str, ClusterResult]:
    """Fit every model on `X` and time it.

    `inputs` maps a model name to a different feature matrix for that model
    (e.g. radians for the haversine DBSCAN).
    """
    inputs = inputs or {}
    results = {}
    for name, model in models.items():
        start = time.perf_counter()
        labels = model.fit_predict(inputs.get(name, X))
        results[name] = ClusterResult(labels, model, time.perf_counter() - start)
    return results
