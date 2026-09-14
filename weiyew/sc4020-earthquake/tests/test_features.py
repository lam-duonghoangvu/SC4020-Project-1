import numpy as np
import pytest

from quake.features import haversine_km, nearest_neighbour_km, to_radians, to_unit_vectors

KM_PER_DEGREE_AT_EQUATOR = 111.195


def test_haversine_one_degree_on_equator():
    assert haversine_km(0, 0, 0, 1) == pytest.approx(KM_PER_DEGREE_AT_EQUATOR, rel=1e-4)


def test_haversine_across_180_degree_line_is_short():
    assert haversine_km(0, 179.9, 0, -179.9) == pytest.approx(0.2 * KM_PER_DEGREE_AT_EQUATOR, rel=1e-4)


def test_haversine_longitude_degree_shrinks_with_latitude():
    assert haversine_km(60, 0, 60, 1) == pytest.approx(0.5 * KM_PER_DEGREE_AT_EQUATOR, rel=1e-3)


def test_unit_vectors_lie_on_sphere_and_meet_across_180_degree_line():
    vectors = to_unit_vectors([0, 45, -90, 0], [179.9, 10, 0, -179.9])

    assert np.linalg.norm(vectors, axis=1) == pytest.approx(np.ones(4))
    assert np.linalg.norm(vectors[0] - vectors[3]) == pytest.approx(np.radians(0.2), rel=1e-4)


def test_to_radians_keeps_latitude_first():
    assert to_radians([90], [180]) == pytest.approx(np.array([[np.pi / 2, np.pi]]))


def test_nearest_neighbour_distances_along_equator():
    lat, lon = [0, 0, 0], [0, 1, 3]

    assert nearest_neighbour_km(lat, lon) == pytest.approx(KM_PER_DEGREE_AT_EQUATOR * np.array([1, 1, 2]), rel=1e-4)
    assert nearest_neighbour_km(lat, lon, k=2) == pytest.approx(KM_PER_DEGREE_AT_EQUATOR * np.array([3, 2, 3]), rel=1e-4)


def test_nearest_neighbour_uses_great_circle_across_180_degree_line():
    assert nearest_neighbour_km([0, 0], [179.5, -179.5]) == pytest.approx([KM_PER_DEGREE_AT_EQUATOR] * 2, rel=1e-4)
