import numpy as np
from sklearn.datasets import make_blobs, make_moons

from sc4020.dbcv import dbcv_score


def test_separated_dense_clusters_score_high_and_wrong_splits_low():
    X, y = make_moons(n_samples=300, noise=0.05, random_state=0)
    across = (X[:, 0] > 0.5).astype(int)  # a straight cut through both moons

    assert dbcv_score(X, y) > 0.3
    assert dbcv_score(X, across) < dbcv_score(X, y)


def test_fewer_than_two_clusters_scores_minus_one():
    X, _ = make_blobs(n_samples=50, centers=1, random_state=0)

    assert dbcv_score(X, np.zeros(50, dtype=int)) == -1.0
    assert dbcv_score(X, np.full(50, -1)) == -1.0


def test_noise_counts_against_the_score():
    X, y = make_blobs(n_samples=200, centers=[[0, 0], [10, 10]], cluster_std=0.5, random_state=0)
    noisy = y.copy()
    noisy[:100] = -1

    assert dbcv_score(X, noisy) < dbcv_score(X, y)
