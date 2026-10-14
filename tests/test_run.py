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


def test_spectral_takes_the_kmeans_k():
    selection = run.select_method(
        blobs_dataset(labelled=False), "Spectral", k_range=range(2, 6)
    )

    assert selection.params == {
        "n_clusters": 3,
        "affinity": run.DEFAULT_SPECTRAL_AFFINITY,
    }


def test_per_dataset_settings_override_the_defaults():
    selection = run.select_method(
        blobs_dataset(name="wine"), "Spectral", k_range=range(2, 6)
    )

    assert selection.params["affinity"] == "rbf"


def test_score_methods_adds_the_run_and_fake_data_columns():
    ds = blobs_dataset()
    selections = {
        name: run.select_method(ds, name, k_range=range(2, 6))
        for name in ("K-Means", "HDBSCAN")
    }
    results = {name: run.fit_method(ds, name, s) for name, s in selections.items()}

    table = run.score_methods(ds, selections, results)

    assert list(table["method"]) == ["K-Means", "HDBSCAN"]
    assert set(run.RUN_COLUMNS) <= set(table.columns)
    assert table.loc[table["method"] == "K-Means", "ari"].iloc[0] == pytest.approx(1.0)
    real, null = table.loc[0, ["silhouette", "null_silhouette"]]
    assert real > null
    assert set(run.tried_settings(ds, selections)["method"]) == {"K-Means", "HDBSCAN"}


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


def test_dense_spectral_is_too_large_on_large_data_but_the_sparse_graph_is_not(
    monkeypatch,
):
    monkeypatch.setattr(run, "MAX_POINTS_FOR_DENSE_SPECTRAL", 100)

    assert run.is_too_large(blobs_dataset(name="wine"), "Spectral")
    assert not run.is_too_large(blobs_dataset(), "Spectral")
    assert not run.is_too_large(blobs_dataset(name="wine"), "K-Means")


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
