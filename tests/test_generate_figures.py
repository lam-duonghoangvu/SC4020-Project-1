import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure
from sklearn.datasets import make_blobs

from sc4020 import generate_figures as gf
from sc4020 import methods


def blob_results():
    X, y = make_blobs(
        n_samples=90, centers=3, n_features=4, cluster_std=0.4, random_state=0
    )
    params = {"K-Means": {"n_clusters": 3}, "DBSCAN": {"eps": 1.0, "min_samples": 5}}
    return X, y, methods.run_models(methods.build_models(params), X, params=params)


def test_build_results_table_has_params_and_shared_metric_columns():
    X, y, results = blob_results()

    table = gf.build_results_table("blobs", X, results, y)

    assert table["method"].tolist() == ["K-Means", "DBSCAN"]
    assert table.loc[0, "param_n_clusters"] == 3
    for column in [
        "fit_seconds",
        "n_clusters",
        "noise_fraction",
        "largest_cluster_share",
        "silhouette",
        "davies_bouldin",
        "ari",
        "nmi",
    ]:
        assert column in table
    assert gf.score_labels(table)["K-Means"].startswith("ARI ")


def test_plots_return_figures():
    X, y, results = blob_results()
    coords = np.column_stack([np.linspace(1, 2, 90), np.linspace(100, 101, 90)])
    table = gf.build_results_table("blobs", X, results, y)

    figures = [
        gf.plot_cluster_grid(X, results, y, title="t", scores=gf.score_labels(table)),
        gf.plot_spatial_grid(coords, results, title="t"),
        gf.plot_k_selection(range(2, 5), [0.2, 0.5, 0.3], best_k=3),
        gf.plot_k_distance(np.linspace(0, 1, 10), 5, 0.5),
    ]
    assert all(isinstance(f, Figure) for f in figures)


def test_save_writes_png_under_results_topic(tmp_path, monkeypatch):
    monkeypatch.setattr(gf, "results_dir", lambda topic: tmp_path / topic)

    path = gf.save(gf.plot_k_selection([2, 3], [0.1, 0.2]), "demo", "k")

    assert path == tmp_path / "demo" / "figures" / "k.png" and path.exists()
