import numpy as np
import pytest
from sklearn.datasets import make_blobs

from sc4020 import methods
from sc4020.datasets.preprocess_dataset import EARTH_RADIUS_KM, to_radians


def two_blobs():
    X, _ = make_blobs(
        n_samples=120,
        centers=[[0, 0, 0, 0], [10, 10, 10, 10]],
        cluster_std=0.3,
        random_state=0,
    )
    return X


def test_kmeans_factory_initialisation_options():
    random_init = methods.kmeans(3, init="random", n_init=1, random_state=7)
    plus_plus = methods.kmeans(3)

    assert (random_init.init, random_init.n_init, random_init.random_state) == (
        "random",
        1,
        7,
    )
    assert (plus_plus.init, plus_plus.n_init) == ("k-means++", 10)


def test_density_methods_separate_two_blobs():
    X = two_blobs()

    for model in [
        methods.dbscan(eps=1.0, min_samples=5),
        methods.hdbscan(min_cluster_size=20, min_samples=5),
    ]:
        assert len(set(model.fit_predict(X)) - {-1}) == 2


def test_hdbscan_passes_min_samples():
    assert methods.hdbscan(min_cluster_size=50, min_samples=10).min_samples == 10
    assert methods.hdbscan(min_cluster_size=50).min_samples is None


def test_haversine_dbscan_keeps_a_group_across_180_degree_line_together():
    rng = np.random.default_rng(0)
    across = np.column_stack([rng.normal(0, 0.05, 30), rng.normal(180, 0.05, 30)])
    across[:, 1] = (across[:, 1] + 180) % 360 - 180
    atlantic = np.column_stack([rng.normal(0, 0.05, 30), rng.normal(-30, 0.05, 30)])
    degrees = np.vstack([across, atlantic])

    model = methods.dbscan(50 / EARTH_RADIUS_KM, 3, metric="haversine")
    labels = model.fit_predict(to_radians(degrees[:, 0], degrees[:, 1]))

    assert model.algorithm == "ball_tree"
    assert len(set(labels[:30])) == 1 and labels[0] != -1
    assert len(set(labels[30:])) == 1 and labels[30] != labels[0]


def test_build_and_run_models_record_params_and_time():
    X = two_blobs()
    params = {
        "K-Means": {"n_clusters": 2},
        "GMM": {"n_components": 2},
        "Agglomerative": {"n_clusters": 2},
    }

    results = methods.run_models(methods.build_models(params), X, params=params)

    assert list(results) == list(params)
    for name, r in results.items():
        assert r.params == params[name] and r.fit_seconds >= 0
        assert len(set(r.labels)) == 2


def test_run_models_uses_per_model_inputs():
    X = two_blobs()

    class Echo:
        def fit_predict(self, data):
            return np.full(len(data), len(data))

    results = methods.run_models({"a": Echo(), "b": Echo()}, X, inputs={"b": X[:10]})

    assert results["a"].labels[0] == len(X) and results["b"].labels[0] == 10


def test_unknown_method_name_is_rejected():
    with pytest.raises(KeyError):
        methods.build_models({"Mean Shift": {}})
