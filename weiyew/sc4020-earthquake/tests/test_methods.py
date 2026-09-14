import numpy as np
import pytest

from quake import methods
from quake.features import EARTH_RADIUS_KM, to_radians


def two_groups_one_across_180_degree_line(seed=0):
    rng = np.random.default_rng(seed)
    across = np.column_stack([rng.normal(0, 0.05, 30), rng.normal(180, 0.05, 30)])
    across[:, 1] = (across[:, 1] + 180) % 360 - 180
    atlantic = np.column_stack([rng.normal(0, 0.05, 30), rng.normal(-30, 0.05, 30)])
    return np.vstack([across, atlantic])


def test_dbscan_eps_is_converted_from_km_to_radians():
    model = methods.dbscan(eps_km=100, min_samples=5)

    assert model.eps == pytest.approx(100 / EARTH_RADIUS_KM)
    assert model.metric == "haversine"


def test_dbscan_keeps_group_across_180_degree_line_together():
    degrees = two_groups_one_across_180_degree_line()

    labels = methods.dbscan(eps_km=50, min_samples=3).fit_predict(to_radians(degrees[:, 0], degrees[:, 1]))

    assert len(set(labels[:30])) == 1 and labels[0] != -1
    assert len(set(labels[30:])) == 1 and labels[30] != labels[0]


def test_hdbscan_uses_haversine_and_passes_min_samples():
    degrees = two_groups_one_across_180_degree_line()

    model = methods.hdbscan(min_cluster_size=10, min_samples=5)
    labels = model.fit_predict(to_radians(degrees[:, 0], degrees[:, 1]))

    assert model.metric == "haversine" and model.min_samples == 5
    assert len(set(labels[:30])) == 1 and labels[0] != labels[30]
