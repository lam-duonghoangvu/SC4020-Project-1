import pytest

from sc4020.clusters import cross_line_pairs, relabel_by_size, spherical_centre


def test_relabel_orders_clusters_by_size_and_keeps_noise():
    assert relabel_by_size([5, 5, -1, 2, 2, 2, 7]).tolist() == [1, 1, -1, 0, 0, 0, 2]


def test_relabel_breaks_ties_by_old_label():
    assert relabel_by_size([3, 3, 1, 1]).tolist() == [1, 1, 0, 0]


def test_spherical_centre_of_points_either_side_of_180_degree_line():
    lat, lon = spherical_centre([0, 0], [179, -179])

    assert lat == pytest.approx(0, abs=1e-9)
    assert abs(lon) == pytest.approx(180)


# Points 0 and 1 are 22 km apart across the 180 degree line; point 2 is 1,100 km west of point 0's
# side; points 3 and 4 are next to the 0 degree line, which must not count.
LAT = [0.0, 0.0, 0.0, 0.0, 0.0]
LON = [179.9, -179.9, -170.0, 0.1, -0.1]


def test_cross_line_pairs_same_cluster():
    assert cross_line_pairs(LAT, LON, [0, 0, 1, 2, 2], max_km=50) == {
        "cross_line_pairs": 1,
        "pairs_both_clustered": 1,
        "pairs_same_cluster": 1,
    }


def test_cross_line_pairs_split_by_clustering_or_noise():
    assert cross_line_pairs(LAT, LON, [0, 1, 1, 2, 2], max_km=50)["pairs_same_cluster"] == 0
    assert cross_line_pairs(LAT, LON, [0, -1, 1, 2, 2], max_km=50) == {
        "cross_line_pairs": 1,
        "pairs_both_clustered": 0,
        "pairs_same_cluster": 0,
    }


def test_cross_line_pairs_respects_distance_and_empty_sides():
    assert cross_line_pairs(LAT, LON, [0, 0, 0, 0, 0], max_km=2000)["cross_line_pairs"] == 2
    assert cross_line_pairs([0, 0], [10, 20], [0, 0])["cross_line_pairs"] == 0
