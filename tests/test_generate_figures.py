import math

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
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
    assert list(gf.pivot_by_method(table, "ari").columns) == ["K-Means", "DBSCAN"]


def test_plots_return_figures():
    X, y, results = blob_results()
    coords = np.column_stack([np.linspace(1, 2, 90), np.linspace(100, 101, 90)])
    table = gf.build_results_table("blobs", X, results, y)

    figures = [
        gf.plot_cluster_grid(X, results, y, title="t", scores=gf.score_labels(table)),
        gf.plot_spatial_grid(coords, results, title="t"),
        gf.plot_k_selection(range(2, 5), [0.2, 0.5, 0.3], best_k=3),
        gf.plot_k_distance(np.linspace(0, 1, 10), 5, 0.5),
        gf.plot_metrics_bar(table, "ari"),
        gf.plot_case_figure(
            X,
            (y > 0).astype(int),
            {m: r.labels for m, r in results.items()},
            [
                {"method": m, **row}
                for m, row in zip(table["method"], table.to_dict("records"))
            ],
            picks=("K-Means", "DBSCAN"),
        ),
    ]
    assert all(isinstance(f, Figure) for f in figures)


def test_save_writes_png_under_results_topic(tmp_path, monkeypatch):
    monkeypatch.setattr(gf, "results_dir", lambda topic: tmp_path / topic)

    path = gf.save(gf.plot_k_selection([2, 3], [0.1, 0.2]), "demo", "k")

    assert path == tmp_path / "demo" / "figures" / "k.png" and path.exists()


def test_cluster_summary_one_row_per_cluster_without_noise():
    events = pd.DataFrame(
        {
            "latitude": [0, 0, 10, 20],
            "longitude": [179, -179, 0, 50],
            "depth": [10, 30, 5, 600],
            "place": [
                "10 km N of A, Fiji",
                "south of Fiji",
                "Somewhere, Brazil",
                "Far away",
            ],
        }
    )

    summary = gf.cluster_summary(events, [0, 0, -1, 1])

    assert summary["cluster"].tolist() == [0, 1]
    assert summary["events"].tolist() == [2, 1]
    assert summary["% of events"].tolist() == pytest.approx([50, 25])
    first = summary.iloc[0]
    assert first["median km to centre"] == pytest.approx(111.195, rel=1e-3)
    assert first["median depth km"] == 20
    assert first["top regions"] == "Fiji (2)"


def test_threshold_table_counts_events_at_or_above_each_magnitude():
    events = pd.DataFrame(
        {
            "mag": [4.5, 5.0, 5.5, 6.0],
            "latitude": [0, 0, 0, 0],
            "longitude": [0, 1, 3, 6],
        }
    )

    table = gf.magnitude_threshold_table(events)

    assert list(table["events"]) == [4, 3, 2, 1]
    assert list(table["% kept"]) == pytest.approx([100, 75, 50, 25])
    # Gaps to the nearest point are 1, 1, 2 and 3 degrees of longitude on the equator.
    assert table["median nearest neighbour km"].iloc[0] == pytest.approx(
        111.195 * 1.5, rel=1e-4
    )
    assert math.isnan(table["median nearest neighbour km"].iloc[-1])


def test_grid_cell_counts_one_column_per_period_with_zero_fill():
    counts = gf.grid_cell_counts(
        lat=[1, 2, 7, 1], lon=[1, 4, 1, -1], period=["H1", "H2", "H2", "H1"]
    )

    assert list(counts.columns) == ["H1", "H2"]
    assert counts.loc[(0.0, 0.0)].tolist() == [1, 1]
    assert counts.loc[(5.0, 0.0)].tolist() == [0, 1]
    assert counts.loc[(0.0, -5.0)].tolist() == [1, 0]


def test_to_latex_table(tmp_path):
    path = gf.to_latex_table(
        pd.DataFrame({"a": [1.23456]}), tmp_path / "t.tex", "Caption", "tab:t"
    )

    assert "1.235" in path.read_text() and "tab:t" in path.read_text()
