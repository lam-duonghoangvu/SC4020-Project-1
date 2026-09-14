import numpy as np
from sklearn.datasets import make_blobs

from hdb import methods


def two_blobs():
    X, _ = make_blobs(n_samples=120, centers=[[0, 0, 0, 0], [10, 10, 10, 10]], cluster_std=0.3, random_state=0)
    return X


def test_kmeans_factories_use_one_initialization_each():
    random_init = methods.kmeans_random(3, seed=7)
    plus_plus = methods.kmeans_plus_plus(3, seed=7)

    assert (random_init.init, random_init.n_init, random_init.random_state) == ("random", 1, 7)
    assert (plus_plus.init, plus_plus.n_init) == ("k-means++", 1)


def test_density_methods_separate_two_blobs():
    X = two_blobs()

    for model in [methods.dbscan(eps=1.0, min_samples=5), methods.hdbscan(min_cluster_size=20, min_samples=5)]:
        labels = model.fit_predict(X)
        assert len(set(labels) - {-1}) == 2


def test_hdbscan_passes_min_samples():
    assert methods.hdbscan(min_cluster_size=50, min_samples=10).min_samples == 10
    assert methods.hdbscan(min_cluster_size=50).min_samples is None
