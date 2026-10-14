import numpy as np
import pytest
from sklearn.datasets import make_moons

from sc4020.datasets import load_dataset
from sc4020.datasets.describe_dataset import describe_shape

MOONS_X, MOONS_Y = make_moons(n_samples=400, noise=0.05, random_state=0)


def test_without_labels_only_density_is_measured():
    shape = describe_shape(MOONS_X)

    assert not shape["has_labels"]
    assert "label_silhouette" not in shape
    assert shape["density_spread"] >= 1
    assert 0 <= shape["far_outlier_share"] <= 1


def test_moons_are_not_convex_but_wine_classes_are():
    moons = describe_shape(MOONS_X, MOONS_Y)
    wine = describe_shape(*load_dataset("wine", return_X_y=True))

    assert (
        moons["nearest_neighbour_accuracy"] - moons["nearest_centroid_accuracy"] > 0.1
    )
    assert wine["nearest_centroid_accuracy"] >= wine["nearest_neighbour_accuracy"]


def test_far_points_count_as_outliers():
    far_points = np.array([[100.0, 100.0]]) + np.arange(5)[:, None] * 30.0
    shape = describe_shape(
        np.vstack([MOONS_X, far_points]), np.concatenate([MOONS_Y, [0] * 5])
    )

    assert shape["far_outlier_share"] >= 5 / 405
    assert shape["smallest_class_share"] == pytest.approx(200 / 405)
