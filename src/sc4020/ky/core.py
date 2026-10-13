"""Algorithms, evaluation metrics and parameter-selection heuristics."""
import time
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score,
                             normalized_mutual_info_score)

R_EARTH_KM = 6371.0088
SIL_SAMPLE = 5000


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def evaluate(X, labels, y_true=None, seed=0):
    """Internal + external validity. Noise points (-1) are excluded from
    internal indices, since they do not belong to any cluster."""
    out = {}
    mask = labels != -1
    lab_c = labels[mask]
    uniq = np.unique(lab_c)
    out["n_clusters"] = int(len(uniq))
    out["noise_frac"] = float((~mask).mean())

    if len(uniq) >= 2 and mask.sum() > len(uniq):
        Xc = X[mask]
        ss = None if len(Xc) <= SIL_SAMPLE else SIL_SAMPLE
        out["silhouette"] = float(silhouette_score(Xc, lab_c, sample_size=ss,
                                                   random_state=seed))
        out["davies_bouldin"] = float(davies_bouldin_score(Xc, lab_c))
        out["calinski_harabasz"] = float(calinski_harabasz_score(Xc, lab_c))
    else:
        out["silhouette"] = np.nan
        out["davies_bouldin"] = np.nan
        out["calinski_harabasz"] = np.nan

    if y_true is not None:
        # noise is kept as its own label so that discarding points is penalised
        out["ari"] = float(adjusted_rand_score(y_true, labels))
        out["nmi"] = float(normalized_mutual_info_score(y_true, labels))
    else:
        out["ari"] = np.nan
        out["nmi"] = np.nan
    return out


# --------------------------------------------------------------------------
# algorithms
# --------------------------------------------------------------------------
def run_kmeans(X, k, init="k-means++", seed=0, n_init=10):
    t = time.perf_counter()
    m = KMeans(n_clusters=k, init=init, n_init=n_init, random_state=seed).fit(X)
    return m.labels_, time.perf_counter() - t, {"inertia": float(m.inertia_),
                                                "n_iter": int(m.n_iter_)}


def run_dbscan(X, eps, min_samples, metric="euclidean"):
    t = time.perf_counter()
    algo = "ball_tree" if metric == "haversine" else "auto"
    m = DBSCAN(eps=eps, min_samples=min_samples, metric=metric,
               algorithm=algo).fit(X)
    return m.labels_, time.perf_counter() - t, {}


def run_hdbscan(X, min_cluster_size, min_samples=None):
    t = time.perf_counter()
    m = HDBSCAN(min_cluster_size=min_cluster_size,
                min_samples=min_samples).fit(X)
    return m.labels_, time.perf_counter() - t, {}


# --------------------------------------------------------------------------
# parameter selection
# --------------------------------------------------------------------------
def k_distance_curve(X, k, metric="euclidean"):
    """Sorted distance to the k-th nearest neighbour."""
    algo = "ball_tree" if metric == "haversine" else "auto"
    nn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm=algo).fit(X)
    d, _ = nn.kneighbors(X)
    return np.sort(d[:, -1])


def knee(curve):
    """Kneedle-style knee: the point of maximum deviation from the chord
    joining the first and last point of the (monotone) curve.

    Works for both convex-increasing curves (the k-distance plot) and
    convex-decreasing curves (the inertia elbow), because the deviation is
    taken in absolute value.
    """
    y = np.asarray(curve, dtype=float)
    n = len(y)
    x = np.linspace(0.0, 1.0, n)
    y = (y - y[0]) / (y[-1] - y[0] + 1e-12)   # normalised, endpoints 0 and 1
    chord = x                                  # chord from (0,0) to (1,1)
    idx = int(np.argmax(np.abs(y - chord)))
    return idx, float(np.asarray(curve, dtype=float)[idx])


def elbow_k(inertias, ks):
    """Knee rule applied to the K-Means inertia curve."""
    i, _ = knee(np.asarray(inertias, dtype=float))
    return ks[i]
