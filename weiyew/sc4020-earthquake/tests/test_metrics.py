import math

import numpy as np
import pandas as pd
import pytest

from quake.features import to_radians, to_unit_vectors
from quake.metrics import best_density_setting, evaluate, k_distances_km, pairwise_ari, timed_fit_predict

KM_PER_DEGREE_AT_EQUATOR = 111.195

# Two tight groups, one straddling the 180 degree line, plus one far-away point.
LAT = np.array([0.0, 0.05, 0.0, 0.05, 45.0])
LON = np.array([179.98, -179.98, 0.0, 0.05, 90.0])
RADIANS = to_radians(LAT, LON)
UNIT = to_unit_vectors(LAT, LON)


def test_noise_is_counted_and_excluded_from_internal_metrics():
    result = evaluate(RADIANS, UNIT, [0, 0, 1, 1, -1])

    assert result["n_clusters"] == 2
    assert result["noise_fraction"] == pytest.approx(0.2)
    assert result["largest_cluster_share"] == pytest.approx(0.4)
    assert result["davies_bouldin"] < 0.1


def test_silhouette_uses_great_circle_distance_across_180_degree_line():
    # In flat radians the first two points would be almost 2 pi apart and the silhouette negative.
    result = evaluate(RADIANS, UNIT, [0, 0, 1, 1, -1])

    assert result["silhouette"] > 0.9


def test_single_cluster_gives_nan_internal_metrics():
    result = evaluate(RADIANS, UNIT, [0, 0, 0, 0, -1])

    assert result["n_clusters"] == 1
    assert result["largest_cluster_share"] == pytest.approx(0.8)
    assert math.isnan(result["silhouette"])
    assert math.isnan(result["davies_bouldin"])


def test_all_noise_has_no_clusters():
    result = evaluate(RADIANS, UNIT, [-1] * 5)

    assert result["n_clusters"] == 0
    assert result["largest_cluster_share"] == 0.0
    assert result["noise_fraction"] == 1.0


def test_k_distances_km_use_neighbours_other_than_the_point_itself():
    lat, lon = [0, 0, 0], [0, 1, 3]

    assert k_distances_km(lat, lon, min_samples=2) == pytest.approx(KM_PER_DEGREE_AT_EQUATOR * np.array([1, 1, 2]), rel=1e-4)
    assert k_distances_km(lat, lon, min_samples=3) == pytest.approx(KM_PER_DEGREE_AT_EQUATOR * np.array([2, 3, 3]), rel=1e-4)


def test_k_distances_km_rejects_min_samples_below_two():
    with pytest.raises(ValueError, match="at least 2"):
        k_distances_km(LAT, LON, min_samples=1)


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
        def fit_predict(self, X):
            return np.zeros(len(X), dtype=int)

    labels, seconds = timed_fit_predict(Constant(), RADIANS)

    assert labels.tolist() == [0] * 5
    assert seconds >= 0


def test_best_density_setting_caps_noise_and_ignores_null_rows():
    grid = pd.DataFrame(
        [
            {"setting": "one cluster", "data": "real", "n_clusters": 1, "noise_fraction": 0.1, "silhouette": np.nan},
            {"setting": "too much noise", "data": "real", "n_clusters": 5, "noise_fraction": 0.6, "silhouette": 0.95},
            {"setting": "null", "data": "uniform null", "n_clusters": 3, "noise_fraction": 0.2, "silhouette": 0.99},
            {"setting": "valid, lower", "data": "real", "n_clusters": 4, "noise_fraction": 0.3, "silhouette": 0.5},
            {"setting": "valid, best", "data": "real", "n_clusters": 2, "noise_fraction": 0.5, "silhouette": 0.7},
        ]
    )

    assert best_density_setting(grid)["setting"] == "valid, best"


def test_best_density_setting_without_data_column_and_with_no_valid_row():
    grid = pd.DataFrame({"n_clusters": [1, 3], "noise_fraction": [0.0, 0.9], "silhouette": [np.nan, 0.8]})

    assert best_density_setting(grid) is None
    assert best_density_setting(grid, max_noise=0.95)["silhouette"] == 0.8
