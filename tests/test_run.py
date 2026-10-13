import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from sklearn.utils import Bunch

from sc4020 import run


def blobs_dataset(name="blobs", labelled=True):
    X, y = make_blobs(
        n_samples=150, centers=[[0, 0], [8, 8], [0, 8]], cluster_std=0.6, random_state=0
    )
    return Bunch(
        name=name,
        data=X,
        target=y if labelled else None,
        target_names=["a", "b", "c"] if labelled else None,
        n_true_clusters=3 if labelled else None,
    )


def test_spectral_and_agglomerative_take_the_kmeans_k_without_reporting_kmeans():
    ds = blobs_dataset(labelled=False)

    selections = run.select_params(
        ds, ["Spectral", "Agglomerative", "Affinity Propagation"], range(2, 6)
    )

    assert list(selections) == ["Spectral", "Agglomerative", "Affinity Propagation"]
    assert selections["Spectral"].params == {
        "n_clusters": 3,
        "affinity": run.DEFAULT_SPECTRAL_AFFINITY,
    }
    assert selections["Agglomerative"].params == {"n_clusters": 3, "linkage": "ward"}
    assert (
        selections["Affinity Propagation"].params
        == run.DEFAULT_AFFINITY_PROPAGATION_PARAMS
    )


def test_per_dataset_settings_override_the_defaults():
    selections = run.select_params(
        blobs_dataset(name="wine"), ["Spectral", "Affinity Propagation"], range(2, 6)
    )

    assert selections["Spectral"].params["affinity"] == "rbf"
    assert selections["Affinity Propagation"].params == {"damping": 0.8}


def test_run_dataset_scores_only_the_requested_methods():
    table, tried, params, results = run.run_dataset(
        blobs_dataset(), ["K-Means", "HDBSCAN"]
    )

    assert list(table["method"]) == ["K-Means", "HDBSCAN"] == list(results)
    assert set(run.RUN_COLUMNS) <= set(table.columns)
    assert table.loc[table["method"] == "K-Means", "ari"].iloc[0] == pytest.approx(1.0)
    assert set(tried["method"]) == {"K-Means", "HDBSCAN"}
    assert params["K-Means"] == {"n_clusters": 3}


def test_random_init_kmeans_reuses_the_kmeans_k_and_scores_its_own_row():
    ds = blobs_dataset()
    selections = {"K-Means": run.select_method(ds, "K-Means", k_range=range(2, 6))}
    selections["K-Means (random init)"] = run.select_method(
        ds, "K-Means (random init)", selections
    )

    assert selections["K-Means (random init)"].params == {"n_clusters": 3}
    assert selections["K-Means (random init)"].tried == []
    results = {name: run.fit_method(ds, name, s) for name, s in selections.items()}
    assert results["K-Means (random init)"].model.init == "random"
    table = run.score_methods(ds, selections, results)
    assert list(table["method"]) == ["K-Means", "K-Means (random init)"]
    assert (table["tuning_seconds"] >= 0).all()
    assert set(run.tried_settings(ds, selections)["method"]) == {"K-Means"}


def test_quadratic_methods_are_skipped_on_large_data(monkeypatch):
    monkeypatch.setattr(run, "MAX_POINTS_FOR_QUADRATIC_METHODS", 100)

    table, *_ = run.run_dataset(blobs_dataset(), ["K-Means", "Affinity Propagation"])

    assert list(table["method"]) == ["K-Means"]


def test_rerunning_some_methods_keeps_the_other_methods_rows(tmp_path):
    path = tmp_path / "blobs_results.csv"
    pd.DataFrame(
        {"method": ["K-Means", "DBSCAN", "GMM"], "silhouette": [0.1, 0.2, 0.3]}
    ).to_csv(path, index=False)

    combined = run._replace_rows(
        path, pd.DataFrame({"method": ["DBSCAN"], "silhouette": [0.9]})
    )

    assert list(combined["method"]) == ["K-Means", "DBSCAN", "GMM"]
    assert combined["silhouette"].tolist() == [0.1, 0.9, 0.3]


def test_format_params_rounds_floats():
    assert run.format_params({"eps": 0.123456, "min_samples": 5}) == (
        "eps=0.123, min_samples=5"
    )
    assert run.format_params(None) == "" and np.isnan(
        run.tuning.Selection({}, np.nan).score
    )
