"""Density-Based Clustering Validation (DBCV), Moulavi et al., SDM 2014.

Silhouette rewards compact, convex clusters, so it is the wrong yardstick for
density-based methods whose clusters can be any shape. DBCV instead compares,
for every cluster, its sparsest internal region (the longest edge inside its
minimum spanning tree under mutual reachability distance) with its densest
connection to any other cluster. It ranges from -1 to 1 and noise points
count against the score, because each cluster is weighted by |C| / n with n
including noise.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

NOISE_LABEL = -1


def _core_distances(D: np.ndarray, n_features: int) -> np.ndarray:
    """All-points core distance of each point from its within-cluster distances D.

    core(o) = (mean_j (1 / d(o, j)) ** d) ** (-1 / d), computed in log space
    because (1 / d) ** d overflows for tens of features.
    """
    n = len(D)
    D = np.maximum(D, np.finfo(float).tiny)  # duplicate points
    log_inv = -n_features * np.log(D)
    np.fill_diagonal(log_inv, -np.inf)
    log_mean = logsumexp(log_inv, axis=1) - np.log(n - 1)
    return np.exp(-log_mean / n_features)


def _mutual_reachability(D: np.ndarray, core_a: np.ndarray, core_b: np.ndarray) -> np.ndarray:
    return np.maximum(D, np.maximum(core_a[:, None], core_b[None, :]))


def dbcv_score(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: int | None = None,
    random_state: int = 42,
) -> float:
    """DBCV of a labelling; -1 when fewer than 2 clusters have 2+ points.

    Memory grows with the square of the largest cluster, so for large data the
    score is estimated on a random `sample_size` of the points (noise included).
    """
    if sample_size is not None and sample_size < len(X):
        idx = np.random.default_rng(random_state).choice(len(X), sample_size, replace=False)
        X, labels = X[idx], labels[idx]
    clusters = [c for c in np.unique(labels) if c != NOISE_LABEL and (labels == c).sum() > 1]
    if len(clusters) < 2:
        return -1.0

    d = X.shape[1]
    members, cores, internal, sparseness = {}, {}, {}, {}
    for c in clusters:
        idx = np.flatnonzero(labels == c)
        D = cdist(X[idx], X[idx])
        core = _core_distances(D, d)
        mst = minimum_spanning_tree(_mutual_reachability(D, core, core)).toarray()
        mst = np.maximum(mst, mst.T)
        degree = (mst > 0).sum(axis=1)
        inner = degree > 1
        if inner.sum() < 2:  # tiny or star-shaped cluster: fall back to all nodes
            inner = np.ones(len(idx), dtype=bool)
        inner_edges = mst[np.ix_(inner, inner)]
        members[c], cores[c], internal[c] = idx, core, inner
        sparseness[c] = inner_edges.max() if inner_edges.any() else mst.max()

    score = 0.0
    for c in clusters:
        a = members[c][internal[c]]
        separation = min(
            _mutual_reachability(
                cdist(X[a], X[members[o][internal[o]]]),
                cores[c][internal[c]],
                cores[o][internal[o]],
            ).min()
            for o in clusters if o != c
        )
        validity = (separation - sparseness[c]) / max(separation, sparseness[c])
        score += len(members[c]) / len(labels) * validity
    return float(score)
