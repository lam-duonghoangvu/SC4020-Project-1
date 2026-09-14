"""Clustering methods under comparison, each returned as an unfitted sklearn estimator.

All methods use Euclidean distance on standardized features (notebooks/eda.ipynb, section 17).
"""

from sklearn.cluster import DBSCAN, HDBSCAN, KMeans


def kmeans_random(n_clusters, seed=0):
    # n_init=1 on purpose: a single run exposes sensitivity to initialization.
    return KMeans(n_clusters=n_clusters, init="random", n_init=1, random_state=seed)


def kmeans_plus_plus(n_clusters, seed=0):
    return KMeans(n_clusters=n_clusters, init="k-means++", n_init=1, random_state=seed)


def dbscan(eps, min_samples):
    return DBSCAN(eps=eps, min_samples=min_samples)


def hdbscan(min_cluster_size, min_samples=None):
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, copy=True)
