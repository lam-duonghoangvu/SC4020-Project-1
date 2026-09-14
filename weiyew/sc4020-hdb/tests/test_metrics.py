import math

import numpy as np
import pandas as pd
import pytest

from hdb.metrics import best_density_setting, evaluate, k_distances, knee_distance, pairwise_ari, timed_fit_predict

# Two tight groups and one far point.
X = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0], [50.0, -50.0]])


def test_noise_is_counted_and_excluded_from_internal_metrics():
    result = evaluate(X, [0, 0, 1, 1, -1])

    assert result["n_clusters"] == 2
    assert result["noise_fraction"] == pytest.approx(0.2)
    assert result["largest_cluster_share"] == pytest.approx(0.4)
    assert result["silhouette"] > 0.9
    assert result["davies_bouldin"] < 0.1
    assert "ari" not in result


def test_ari_and_nmi_count_noise_as_its_own_group():
    perfect = evaluate(X, [0, 0, 1, 1, -1], truth=["a", "a", "b", "b", "c"])
    noise_merges_classes = evaluate(X, [0, 0, -1, -1, -1], truth=["a", "a", "b", "b", "c"])

    assert perfect["ari"] == pytest.approx(1.0) and perfect["nmi"] == pytest.approx(1.0)
    assert noise_merges_classes["ari"] < 1.0


def test_silhouette_sample_that_misses_a_tiny_cluster_uses_all_points():
    rng = np.random.default_rng(0)
    points = np.vstack([rng.normal(0, 0.1, size=(1000, 2)), [[20.0, 20.0], [20.1, 20.0]]])
    labels = np.array([0] * 1000 + [1, 1])

    # A 10-point sample almost surely holds only cluster 0; the score must still be computed.
    result = evaluate(points, labels, silhouette_sample=10, seed=0)

    assert not math.isnan(result["silhouette"])
    assert result["silhouette"] > 0.5


def test_silhouette_sample_matches_score_on_subsample():
    rng = np.random.default_rng(1)
    points = np.vstack([rng.normal(0, 1, size=(300, 2)), rng.normal(6, 1, size=(300, 2))])
    labels = np.array([0] * 300 + [1] * 300)

    full = evaluate(points, labels)["silhouette"]
    sampled = evaluate(points, labels, silhouette_sample=200, seed=0)["silhouette"]

    assert sampled == pytest.approx(full, abs=0.05)
    assert sampled != full


def test_single_cluster_gives_nan_internal_metrics():
    result = evaluate(X, [0, 0, 0, 0, -1])

    assert result["n_clusters"] == 1
    assert math.isnan(result["silhouette"]) and math.isnan(result["davies_bouldin"])


def test_all_noise_has_no_clusters():
    result = evaluate(X, [-1] * 5)

    assert result["n_clusters"] == 0
    assert result["largest_cluster_share"] == 0.0
    assert result["noise_fraction"] == 1.0


def test_k_distances_use_neighbours_other_than_the_point_itself():
    points = np.array([[0.0], [1.0], [3.0]])

    assert k_distances(points, min_samples=2) == pytest.approx([1, 1, 2])
    assert k_distances(points, min_samples=3) == pytest.approx([2, 3, 3])


def test_k_distances_of_tied_points_are_zero():
    points = np.array([[5.0], [5.0], [5.0], [9.0]])

    assert k_distances(points, min_samples=3).tolist() == [0.0, 0.0, 0.0, 4.0]


def test_k_distances_rejects_min_samples_below_two():
    with pytest.raises(ValueError, match="at least 2"):
        k_distances(X, min_samples=1)


def test_knee_distance_finds_the_bend_of_a_hockey_stick_curve():
    flat_then_steep = np.concatenate([np.linspace(0.0, 1.0, 90), np.linspace(2.0, 20.0, 10)])

    assert knee_distance(flat_then_steep) == pytest.approx(1.0)


def test_knee_distance_of_constant_curve_is_that_value():
    assert knee_distance([2.0, 2.0, 2.0]) == 2.0


def test_pairwise_ari_compares_every_pair_and_ignores_label_names():
    runs = [np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0]), np.array([0, 1, 0, 1])]

    scores = pairwise_ari(runs)

    assert len(scores) == 3
    assert scores[0] == pytest.approx(1.0)


def test_pairwise_ari_needs_two_runs():
    with pytest.raises(ValueError, match="two"):
        pairwise_ari([np.array([0, 1])])


def test_timed_fit_predict_returns_labels_and_elapsed_seconds():
    class Constant:
        def fit_predict(self, data):
            return np.zeros(len(data), dtype=int)

    labels, seconds = timed_fit_predict(Constant(), X)

    assert labels.tolist() == [0] * 5
    assert seconds >= 0


def test_best_density_setting_caps_noise_and_ignores_other_data():
    grid = pd.DataFrame(
        [
            {"setting": "one cluster", "data": "real", "n_clusters": 1, "noise_fraction": 0.1, "silhouette": np.nan},
            {"setting": "too much noise", "data": "real", "n_clusters": 5, "noise_fraction": 0.6, "silhouette": 0.95},
            {"setting": "null", "data": "shuffled null", "n_clusters": 3, "noise_fraction": 0.2, "silhouette": 0.99},
            {"setting": "valid, lower", "data": "real", "n_clusters": 4, "noise_fraction": 0.3, "silhouette": 0.5},
            {"setting": "valid, best", "data": "real", "n_clusters": 2, "noise_fraction": 0.5, "silhouette": 0.7},
        ]
    )

    assert best_density_setting(grid)["setting"] == "valid, best"


def test_best_density_setting_can_cap_the_largest_cluster():
    grid = pd.DataFrame(
        [
            {"setting": "giant plus tiny", "n_clusters": 2, "noise_fraction": 0.0, "largest_cluster_share": 0.99, "silhouette": 0.6},
            {"setting": "balanced", "n_clusters": 4, "noise_fraction": 0.1, "largest_cluster_share": 0.4, "silhouette": 0.3},
        ]
    )

    assert best_density_setting(grid)["setting"] == "giant plus tiny"
    assert best_density_setting(grid, max_largest=0.5)["setting"] == "balanced"
    assert best_density_setting(grid, max_largest=0.3) is None


def test_best_density_setting_without_data_column_and_with_no_valid_row():
    grid = pd.DataFrame({"n_clusters": [1, 3], "noise_fraction": [0.0, 0.9], "silhouette": [np.nan, 0.8]})

    assert best_density_setting(grid) is None
    assert best_density_setting(grid, max_noise=0.95)["silhouette"] == 0.8
