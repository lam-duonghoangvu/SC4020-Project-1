import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from sc4020.screening import (
    cluster_counts,
    hdbscan_null_test,
    null_gap,
    shuffled_null,
    silhouette_null_test,
    single_value_share,
    uniform_box_null,
)


def separated_blobs(seed=0):
    X, _ = make_blobs(n_samples=300, centers=[[0, 0], [10, 10], [0, 10]], cluster_std=0.5, random_state=seed)
    return X


def blobs_10d(seed=0):
    # In 2D, well-separated blobs still fail the silhouette test: each column alone is clumped, so
    # the shuffled null keeps tight groups. In 10D with random centres only the joint pattern
    # separates them.
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


def test_uniform_box_null_stays_inside_bounding_box():
    X = np.array([[-5.0, 1.0], [5.0, 3.0], [0.0, 2.0]])

    N = uniform_box_null(np.repeat(X, 100, axis=0), np.random.default_rng(0))

    assert N.shape == (300, 2)
    assert (N >= [-5.0, 1.0]).all() and (N <= [5.0, 3.0]).all()


def test_silhouette_test_passes_separated_blobs_and_fails_uniform_points():
    passing = silhouette_null_test(blobs_10d(), k_values=[3]).iloc[0]
    failing = silhouette_null_test(np.random.default_rng(0).uniform(0, 10, size=(300, 4)), k_values=[3]).iloc[0]

    assert passing["passes"] and passing["ratio"] >= 2.0
    assert not failing["passes"] and failing["ratio"] == pytest.approx(1.0, abs=0.3)


def test_silhouette_to_features_is_applied_to_shuffled_raw_columns():
    X = separated_blobs()
    seen = []

    def to_features(data):
        seen.append(data)
        return data

    silhouette_null_test(X, k_values=[3], n_nulls=2, to_features=to_features)

    assert len(seen) == 3
    assert np.array_equal(seen[0], X)
    for null in seen[1:]:
        assert not np.array_equal(null, X)
        assert np.array_equal(np.sort(null, axis=0), np.sort(X, axis=0))


def test_hdbscan_test_one_row_per_size_and_dataset_and_finds_real_blobs():
    sizes = hdbscan_null_test(separated_blobs(), min_cluster_sizes=[10, 20], null="uniform_box", n_nulls=2)
    result = hdbscan_null_test(blobs_10d(), min_cluster_sizes=[20], n_nulls=2).set_index("data")

    assert len(sizes) == 2 * 3 and list(sizes["data"].unique()) == ["real", "null0", "null1"]
    assert result.loc["real", "n_clusters"] == 3
    assert result.loc["real", "noise_fraction"] < result.loc[["null0", "null1"], "noise_fraction"].min()


def test_hdbscan_haversine_accepts_radian_coordinates():
    rng = np.random.default_rng(0)
    centers = np.array([[10.0, 120.0], [-20.0, -70.0], [-30.0, 179.9]])
    degrees = np.vstack([c + rng.normal(0, 0.3, size=(100, 2)) for c in centers])
    degrees[:, 1] = (degrees[:, 1] + 180) % 360 - 180

    result = hdbscan_null_test(np.radians(degrees), min_cluster_sizes=[20], metric="haversine", n_nulls=1)

    assert result.loc[result["data"] == "real", "n_clusters"].item() == 3


def test_cluster_counts_ignore_noise():
    assert cluster_counts([-1, -1, 0, 0, 0, 1]) == {
        "n_clusters": 2, "noise_fraction": pytest.approx(2 / 6), "largest_cluster_fraction": pytest.approx(3 / 6)
    }
    assert cluster_counts([-1, -1]) == {"n_clusters": 0, "noise_fraction": 1.0, "largest_cluster_fraction": 0.0}


def test_single_value_share_per_cluster_skips_noise():
    shares = single_value_share([0, 0, 0, 0, 1, 1, -1], ["a", "a", "a", "b", "c", "c", "a"])

    assert shares.to_dict() == {0: 0.75, 1: 1.0}
    assert isinstance(shares, pd.Series)


def test_null_gap_is_large_for_real_blobs():
    from sc4020 import methods

    gap = null_gap(blobs_10d(), lambda Z: methods.kmeans(3).fit_predict(Z), n_nulls=2)

    assert gap["real_clusters"] == 3
    assert gap["gap"] > 0.3
