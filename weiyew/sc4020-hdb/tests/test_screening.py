import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from hdb.screening import cluster_counts, hdbscan_null_test, shuffled_null, silhouette_null_test, single_value_share


def blobs_10d(seed=0):
    # In 2D, well-separated blobs still fail the silhouette test: each column alone is clumped, so the
    # shuffled null keeps tight groups. In 10D with random centers only the joint pattern separates them.
    centers = np.random.default_rng(seed).uniform(-10, 10, size=(3, 10))
    X, _ = make_blobs(n_samples=300, centers=centers, cluster_std=0.5, random_state=seed)
    return X


def test_shuffled_null_keeps_each_column_distribution_but_breaks_rows():
    X = np.column_stack([np.arange(100.0), np.arange(100.0)])

    N = shuffled_null(X, np.random.default_rng(0))

    assert N.shape == X.shape
    for j in range(X.shape[1]):
        assert np.array_equal(np.sort(N[:, j]), X[:, j])
    assert not np.array_equal(N[:, 0], N[:, 1])


def test_silhouette_test_passes_separated_blobs():
    result = silhouette_null_test(blobs_10d(), k_values=[3]).iloc[0]

    assert result["passes"]
    assert result["ratio"] >= 2.0


def test_silhouette_test_fails_uniform_points():
    X = np.random.default_rng(0).uniform(0, 10, size=(300, 4))

    result = silhouette_null_test(X, k_values=[3]).iloc[0]

    assert not result["passes"]
    assert result["ratio"] == pytest.approx(1.0, abs=0.3)


def test_hdbscan_finds_blobs_that_the_shuffled_null_does_not_produce():
    result = hdbscan_null_test(blobs_10d(), min_cluster_sizes=[20], n_nulls=2).set_index("data")

    assert list(result.index) == ["real", "null0", "null1"]
    assert result.loc["real", "n_clusters"] == 3
    assert result.loc["real", "noise_fraction"] < result.loc[["null0", "null1"], "noise_fraction"].min()


def test_cluster_counts_ignore_noise():
    counts = cluster_counts([-1, -1, 0, 0, 0, 1])

    assert counts == {"n_clusters": 2, "noise_fraction": pytest.approx(2 / 6), "largest_cluster_fraction": pytest.approx(3 / 6)}


def test_cluster_counts_all_noise():
    assert cluster_counts([-1, -1]) == {"n_clusters": 0, "noise_fraction": 1.0, "largest_cluster_fraction": 0.0}


def test_single_value_share_per_cluster_skips_noise():
    shares = single_value_share([0, 0, 0, 0, 1, 1, -1], ["a", "a", "a", "b", "c", "c", "a"])

    assert shares.to_dict() == {0: 0.75, 1: 1.0}
    assert isinstance(shares, pd.Series)
