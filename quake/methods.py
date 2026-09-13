"""Clustering methods under comparison, each returned as an unfitted sklearn estimator.

K-means runs on 3D unit vectors. DBSCAN and HDBSCAN run on [latitude, longitude] in radians with the
haversine distance (notebooks/eda.ipynb, section 5).
"""

from sklearn.cluster import DBSCAN, HDBSCAN, KMeans

from quake.features import EARTH_RADIUS_KM


def kmeans_random(n_clusters, seed=0):
    # n_init=1 on purpose: a single run exposes sensitivity to initialization.
    return KMeans(n_clusters=n_clusters, init="random", n_init=1, random_state=seed)


def kmeans_plus_plus(n_clusters, seed=0):
    return KMeans(n_clusters=n_clusters, init="k-means++", n_init=1, random_state=seed)


def dbscan(eps_km, min_samples):
    """eps is a distance in km on the Earth's surface, converted to radians for the haversine metric."""
    return DBSCAN(eps=eps_km / EARTH_RADIUS_KM, min_samples=min_samples, metric="haversine")


def hdbscan(min_cluster_size, min_samples=None):
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="haversine", copy=True)
