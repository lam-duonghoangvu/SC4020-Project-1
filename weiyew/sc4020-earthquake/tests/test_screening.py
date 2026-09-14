import numpy as np
import pytest
from sklearn.datasets import make_blobs

from quake.screening import hdbscan_null_test, shuffled_null, silhouette_null_test, uniform_box_null


def separated_blobs(seed=0):
    X, _ = make_blobs(n_samples=300, centers=[[0, 0], [10, 10], [0, 10]], cluster_std=0.5, random_state=seed)
    return X


def uniform_points(seed=0):
    return np.random.default_rng(seed).uniform(0, 10, size=(300, 2))


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


def test_silhouette_test_passes_blobs_whose_columns_overlap_after_shuffling():
    # In 2D, well-separated blobs still fail: each column alone is clumped, so the shuffled
    # null keeps tight groups (ratio about 1.2). In 10D with random centers the columns
    # overlap, and only the joint pattern separates the blobs.
    centers = np.random.default_rng(0).uniform(-10, 10, size=(3, 10))
    X, _ = make_blobs(n_samples=300, centers=centers, cluster_std=0.5, random_state=0)

    result = silhouette_null_test(X, k_values=[3]).iloc[0]

    assert result["passes"]
    assert result["ratio"] >= 2.0


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


def test_silhouette_test_fails_uniform_points():
    result = silhouette_null_test(uniform_points(), k_values=[3]).iloc[0]

    assert not result["passes"]
    assert result["ratio"] == pytest.approx(1.0, abs=0.3)


def test_hdbscan_test_has_one_row_per_size_and_dataset():
    result = hdbscan_null_test(separated_blobs(), min_cluster_sizes=[10, 20], null="uniform_box", n_nulls=2)

    assert len(result) == 2 * 3
    assert list(result["data"].unique()) == ["real", "null0", "null1"]


def test_hdbscan_finds_blobs_that_the_uniform_null_does_not_produce():
    result = hdbscan_null_test(separated_blobs(), min_cluster_sizes=[20], null="uniform_box").set_index("data")

    assert result.loc["real", "n_clusters"] == 3
    assert result.loc["real", "noise_fraction"] < result.loc[["null0", "null1", "null2"], "noise_fraction"].min()


def test_hdbscan_haversine_accepts_radian_coordinates():
    # Three tight groups of (latitude, longitude) degrees, including one across the 180 degree line.
    rng = np.random.default_rng(0)
    centers = np.array([[10.0, 120.0], [-20.0, -70.0], [-30.0, 179.9]])
    degrees = np.vstack([c + rng.normal(0, 0.3, size=(100, 2)) for c in centers])
    degrees[:, 1] = (degrees[:, 1] + 180) % 360 - 180

    result = hdbscan_null_test(np.radians(degrees), min_cluster_sizes=[20], metric="haversine", n_nulls=1)

    assert result.loc[result["data"] == "real", "n_clusters"].item() == 3
