"""Unsupervised hyperparameter selection for the clustering methods.

All selection here is silhouette-driven (no ground-truth labels used), since
in real use these choices have to be made without knowing the true clusters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


@dataclass
class KSelectionResult:
    best_k: int
    k_range: list[int]
    scores: list[float]


def select_k_by_silhouette(
    X: np.ndarray, k_range: range = range(2, 11), random_state: int = 42
) -> KSelectionResult:
    scores = []
    for k in k_range:
        labels = KMeans(
            n_clusters=k, init="k-means++", n_init=10, random_state=random_state
        ).fit_predict(X)
        scores.append(silhouette_score(X, labels))
    best_k = list(k_range)[int(np.argmax(scores))]
    return KSelectionResult(best_k=best_k, k_range=list(k_range), scores=scores)


@dataclass
class KDistanceResult:
    distances: np.ndarray
    knee_idx: int
    eps_at_knee: float


def _find_knee(sorted_distances: np.ndarray) -> int:
    """Max discrete second-derivative point on the sorted k-distance curve."""
    first_deriv = np.diff(sorted_distances)
    second_deriv = np.diff(first_deriv)
    if len(second_deriv) == 0:
        return len(sorted_distances) - 1
    return int(np.argmax(second_deriv)) + 1


def select_dbscan_eps(X: np.ndarray, min_samples: int) -> KDistanceResult:
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, -1])
    knee_idx = _find_knee(k_distances)
    return KDistanceResult(
        distances=k_distances, knee_idx=knee_idx, eps_at_knee=float(k_distances[knee_idx])
    )


def default_min_samples(n_features: int) -> int:
    return int(np.clip(2 * n_features, 4, 20))


@dataclass
class DBSCANTuneResult:
    eps: float
    min_samples: int
    score: float
    tried: list[dict] = field(default_factory=list)


def _score_labels_for_tuning(X: np.ndarray, labels: np.ndarray) -> float:
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    noise_ratio = float(np.mean(labels == -1))
    if n_clusters < 2 or noise_ratio > 0.5:
        return -1.0
    return float(silhouette_score(X[mask], labels[mask]))


def tune_dbscan(
    X: np.ndarray,
    min_samples_candidates: list[int] | None = None,
    eps_candidates: list[float] | None = None,
) -> DBSCANTuneResult:
    n_features = X.shape[1]
    if min_samples_candidates is None:
        base = default_min_samples(n_features)
        min_samples_candidates = sorted({max(3, base // 2), base, base * 2})

    tried = []
    best = None
    for min_samples in min_samples_candidates:
        knee = select_dbscan_eps(X, min_samples)
        if eps_candidates is None:
            eps_grid = knee.eps_at_knee * np.array([0.5, 0.75, 1.0, 1.5, 2.0])
        else:
            eps_grid = np.array(eps_candidates)

        for eps in eps_grid:
            eps = float(eps)
            if eps <= 0:
                continue
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
            score = _score_labels_for_tuning(X, labels)
            tried.append({"eps": eps, "min_samples": min_samples, "score": score})
            if best is None or score > best["score"]:
                best = {"eps": eps, "min_samples": min_samples, "score": score}

    return DBSCANTuneResult(
        eps=best["eps"], min_samples=best["min_samples"], score=best["score"], tried=tried
    )


@dataclass
class HDBSCANTuneResult:
    min_cluster_size: int
    score: float
    tried: list[dict] = field(default_factory=list)


def tune_hdbscan(
    X: np.ndarray, min_cluster_size_candidates: list[int] | None = None
) -> HDBSCANTuneResult:
    n = X.shape[0]
    if min_cluster_size_candidates is None:
        cap = max(5, n // 20)
        min_cluster_size_candidates = sorted(
            {c for c in [5, 10, 20, 30, 50] if c <= cap} | {max(5, cap)}
        )

    tried = []
    best = None
    for mcs in min_cluster_size_candidates:
        labels = HDBSCAN(min_cluster_size=mcs).fit_predict(X)
        score = _score_labels_for_tuning(X, labels)
        tried.append({"min_cluster_size": mcs, "score": score})
        if best is None or score > best["score"]:
            best = {"min_cluster_size": mcs, "score": score}

    return HDBSCANTuneResult(
        min_cluster_size=best["min_cluster_size"], score=best["score"], tried=tried
    )
