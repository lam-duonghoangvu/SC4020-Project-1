import math

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs, make_moons

from sc4020.datasets.preprocess_dataset import to_radians, to_unit_vectors
from sc4020.metrics import (
    dbcv_score,
    evaluate,
    haversine_km,
    nearest_neighbour_km,
    pairwise_ari,
    single_value_share,
)

KM_PER_DEGREE_AT_EQUATOR = 111.195

# Two tight groups and one far point.
X = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0], [50.0, -50.0]])


def test_noise_is_counted_and_excluded_from_internal_metrics():
    result = evaluate(X, [0, 0, 1, 1, -1])

    assert result["n_clusters"] == 2
    assert result["noise_fraction"] == pytest.approx(0.2)
    assert result["largest_cluster_share"] == pytest.approx(0.4)
    assert result["silhouette"] > 0.9
    assert result["davies_bouldin"] < 0.1
    assert result["calinski_harabasz"] > 100
    assert "ari" not in result


def test_external_scores_count_noise_as_its_own_group():
    truth = ["a", "a", "b", "b", "c"]
    perfect = evaluate(X, [0, 0, 1, 1, -1], truth)
    noise_merges_classes = evaluate(X, [0, 0, -1, -1, -1], truth)

    for score in ["ari", "nmi", "ami", "fmi"]:
        assert perfect[score] == pytest.approx(1.0)
    assert noise_merges_classes["ari"] < 1.0


def test_silhouette_sample_that_misses_a_tiny_cluster_uses_all_points():
    rng = np.random.default_rng(0)
    points = np.vstack(
        [rng.normal(0, 0.1, size=(1000, 2)), [[20.0, 20.0], [20.1, 20.0]]]
    )
    labels = np.array([0] * 1000 + [1, 1])

    # A 10-point sample almost surely holds only cluster 0; the score must still be computed.
    result = evaluate(points, labels, silhouette_sample=10, seed=0)

    assert result["silhouette"] > 0.5


def test_silhouette_sample_matches_score_on_subsample():
    rng = np.random.default_rng(1)
    points = np.vstack(
        [rng.normal(0, 1, size=(300, 2)), rng.normal(6, 1, size=(300, 2))]
    )
    labels = np.array([0] * 300 + [1] * 300)

    full = evaluate(points, labels, silhouette_sample=None)["silhouette"]
    sampled = evaluate(points, labels, silhouette_sample=200, seed=0)["silhouette"]

    assert sampled == pytest.approx(full, abs=0.05)
    assert sampled != full


def test_single_cluster_and_all_noise_give_nan_internal_metrics():
    one = evaluate(X, [0, 0, 0, 0, -1])
    none = evaluate(X, [-1] * 5)

    assert one["n_clusters"] == 1 and one["largest_cluster_share"] == pytest.approx(0.8)
    assert math.isnan(one["silhouette"]) and math.isnan(one["davies_bouldin"])
    assert (
        none["n_clusters"] == 0
        and none["largest_cluster_share"] == 0.0
        and none["noise_fraction"] == 1.0
    )


def test_haversine_silhouette_and_unit_vector_davies_bouldin_on_the_sphere():
    # Two tight groups, one straddling the 180 degree line, plus one far-away point.
    lat = np.array([0.0, 0.05, 0.0, 0.05, 45.0])
    lon = np.array([179.98, -179.98, 0.0, 0.05, 90.0])

    result = evaluate(
        to_radians(lat, lon),
        [0, 0, 1, 1, -1],
        metric="haversine",
        X_davies_bouldin=to_unit_vectors(lat, lon),
    )

    # In flat radians the first two points would be almost 2 pi apart and the silhouette negative.
    assert result["silhouette"] > 0.9
    assert result["davies_bouldin"] < 0.1


def test_pairwise_ari_compares_every_pair_and_ignores_label_names():
    scores = pairwise_ari(
        [np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0]), np.array([0, 1, 0, 1])]
    )

    assert len(scores) == 3
    assert scores[0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="two"):
        pairwise_ari([np.array([0, 1])])


def test_haversine_distances():
    assert haversine_km(0, 0, 0, 1) == pytest.approx(KM_PER_DEGREE_AT_EQUATOR, rel=1e-4)
    assert haversine_km(0, 179.9, 0, -179.9) == pytest.approx(
        0.2 * KM_PER_DEGREE_AT_EQUATOR, rel=1e-4
    )
    assert haversine_km(60, 0, 60, 1) == pytest.approx(
        0.5 * KM_PER_DEGREE_AT_EQUATOR, rel=1e-3
    )


def test_nearest_neighbour_distances_along_equator_and_across_180():
    lat, lon = [0, 0, 0], [0, 1, 3]

    assert nearest_neighbour_km(lat, lon) == pytest.approx(
        KM_PER_DEGREE_AT_EQUATOR * np.array([1, 1, 2]), rel=1e-4
    )
    assert nearest_neighbour_km(lat, lon, k=2) == pytest.approx(
        KM_PER_DEGREE_AT_EQUATOR * np.array([3, 2, 3]), rel=1e-4
    )
    assert nearest_neighbour_km([0, 0], [179.5, -179.5]) == pytest.approx(
        [KM_PER_DEGREE_AT_EQUATOR] * 2, rel=1e-4
    )


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
    X, y = make_blobs(
        n_samples=200, centers=[[0, 0], [10, 10]], cluster_std=0.5, random_state=0
    )
    noisy = y.copy()
    noisy[:100] = -1

    assert dbcv_score(X, noisy) < dbcv_score(X, y)


def test_single_value_share_per_cluster_skips_noise():
    shares = single_value_share(
        [0, 0, 0, 0, 1, 1, -1], ["a", "a", "a", "b", "c", "c", "a"]
    )

    assert shares.to_dict() == {0: 0.75, 1: 1.0}
    assert isinstance(shares, pd.Series)
