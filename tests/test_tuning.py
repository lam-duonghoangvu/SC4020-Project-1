import numpy as np
import pytest
from sklearn.datasets import make_blobs

from sc4020 import tuning


def blobs(seed=0):
    X, y = make_blobs(n_samples=300, centers=[[0, 0], [8, 8], [0, 8]], cluster_std=0.6, random_state=seed)
    return X, y


def test_k_distance_curve_uses_neighbours_other_than_the_point_itself():
    points = np.array([[0.0], [1.0], [3.0]])

    assert tuning.k_distance_curve(points, 2) == pytest.approx([1, 1, 2])
    assert tuning.k_distance_curve(points, 3) == pytest.approx([2, 3, 3])
    assert tuning.k_distance_curve(np.array([[5.0], [5.0], [5.0], [9.0]]), 3).tolist() == [0.0, 0.0, 0.0, 4.0]
    with pytest.raises(ValueError, match="at least 2"):
        tuning.k_distance_curve(points, 1)


def test_knee_finds_the_bend_of_increasing_and_decreasing_curves():
    hockey_stick = np.concatenate([np.linspace(0.0, 1.0, 90), np.linspace(2.0, 20.0, 10)])

    assert tuning.knee(hockey_stick) == (89, pytest.approx(1.0))
    assert tuning.knee([2.0, 2.0, 2.0])[1] == 2.0
    assert tuning.elbow_k([100, 40, 20, 18, 17, 16], [2, 3, 4, 5, 6, 7]) == 4


def test_default_min_samples_is_twice_the_dimension_within_4_to_20():
    assert [tuning.default_min_samples(d) for d in (1, 3, 15)] == [4, 6, 20]


def test_select_kmeans_by_silhouette_records_label_free_and_best_ari():
    X, y = blobs()

    free = tuning.select_kmeans(X, k_range=range(2, 6))
    fixed = tuning.select_kmeans(X, k=5, k_range=range(2, 6), y_true=y)

    assert free.params == {"n_clusters": 3}
    assert fixed.params == {"n_clusters": 5} and fixed.label_free == {"n_clusters": 3}
    assert fixed.best_ari == {"n_clusters": 3}


def test_select_dbscan_and_hdbscan_find_the_blobs():
    X, y = blobs()

    for selection in [tuning.select_dbscan(X, y_true=y), tuning.select_hdbscan(X, y_true=y)]:
        best = next(t for t in selection.tried if all(t[k] == v for k, v in selection.params.items()))
        assert selection.valid
        assert best["n_clusters"] == 3 and best["noise_fraction"] <= 0.5
        assert selection.best_ari is not None


def make_trial(name, n_clusters, noise, silhouette, largest=0.5):
    return {"name": name, "n_clusters": n_clusters, "noise_fraction": noise, "largest_cluster_share": largest,
            "silhouette": silhouette, "passes": n_clusters >= 2 and noise <= 0.5 and largest <= 0.9}


def test_shared_rule_caps_noise_then_takes_the_highest_silhouette():
    tried = [
        make_trial("one cluster", 1, 0.1, np.nan),
        make_trial("too much noise", 5, 0.6, 0.95),
        make_trial("valid, lower", 4, 0.3, 0.5),
        make_trial("valid, best", 2, 0.5, 0.7),
    ]

    assert tuning._pick(tried, ("name",)).params == {"name": "valid, best"}


def test_shared_rule_falls_back_to_least_noise_and_marks_invalid():
    tried = [make_trial("one cluster", 1, 0.0, np.nan), make_trial("noisy", 3, 0.9, 0.8), make_trial("noisier", 3, 0.95, 0.9)]

    selection = tuning._pick(tried, ("name",))

    assert selection.params == {"name": "noisy"} and not selection.valid


def test_largest_cluster_cap_rejects_a_giant_plus_tiny_split():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, size=(400, 2)), rng.normal(30, 0.1, size=(8, 2))])

    capped = tuning.select_hdbscan(X, min_cluster_sizes=[5], max_largest_share=0.9)

    assert all(not t["passes"] or t["largest_cluster_share"] <= 0.9 for t in capped.tried)


def test_select_all_keys_match_the_four_methods():
    X, _ = blobs()

    selections = tuning.select_all(X, k=3, k_range=range(2, 5))

    assert list(selections) == ["K-Means", "DBSCAN", "HDBSCAN", "GMM"]
    assert selections["GMM"].params["n_components"] == 3


def test_best_setting_applies_the_rule_to_a_grid_and_ignores_null_rows():
    grid = [
        {"setting": "one cluster", "data": "real", "n_clusters": 1, "noise_fraction": 0.1, "silhouette": np.nan},
        {"setting": "too much noise", "data": "real", "n_clusters": 5, "noise_fraction": 0.6, "silhouette": 0.95},
        {"setting": "null", "data": "uniform null", "n_clusters": 3, "noise_fraction": 0.2, "silhouette": 0.99},
        {"setting": "valid, lower", "data": "real", "n_clusters": 4, "noise_fraction": 0.3, "silhouette": 0.5},
        {"setting": "valid, best", "data": "real", "n_clusters": 2, "noise_fraction": 0.5, "silhouette": 0.7},
    ]

    assert tuning.best_setting(grid)["setting"] == "valid, best"
    assert tuning.best_setting(grid, max_noise=0.4)["setting"] == "valid, lower"
    assert tuning.best_setting(grid[:2]) is None


def test_best_setting_can_cap_the_largest_cluster():
    grid = [
        {"setting": "giant plus tiny", "n_clusters": 2, "noise_fraction": 0.0, "largest_cluster_share": 0.99, "silhouette": 0.6},
        {"setting": "balanced", "n_clusters": 4, "noise_fraction": 0.1, "largest_cluster_share": 0.4, "silhouette": 0.3},
    ]

    assert tuning.best_setting(grid)["setting"] == "giant plus tiny"
    assert tuning.best_setting(grid, max_largest_share=0.5)["setting"] == "balanced"
    assert tuning.best_setting(grid, max_largest_share=0.3) is None
