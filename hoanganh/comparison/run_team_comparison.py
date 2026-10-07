"""Run K-Means, DBSCAN, HDBSCAN and GMM on every dataset the team uses.

Usage (from the hoanganh folder):
    python comparison/run_team_comparison.py                 # all 12 datasets
    python comparison/run_team_comparison.py wine ecoli      # a subset

Every dataset goes through the same protocol (tuning.py):
  K-Means  k by silhouette (the true k when labels exist)
  GMM      k and covariance type by BIC (the true k when labels exist)
  DBSCAN   eps around the k-distance knee x min_samples, best DBCV
  HDBSCAN  min_cluster_size x min_samples, best DBCV
Labels, where they exist, are used only for ARI after the fact.

Writes comparison/results/team_*.csv and comparison/figures/team/<dataset>.png.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from team_data import LOADERS  # also puts hoanganh/src on the path

from sc4020 import tuning
from sc4020.clustering import four_models, run_models
from sc4020.dbcv import dbcv_score
from sc4020.metrics import NOISE_LABEL, build_results_table
from sc4020.viz import plot_pca_grid, plot_spatial_grid

RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures" / "team"
METHODS = ["K-Means", "DBSCAN", "HDBSCAN", "GMM"]

# Silhouette and DBCV grow with n^2, so both are estimated on a sample for large data.
SILHOUETTE_SAMPLE = 5000
DBCV_SAMPLE = 3000

# Global earthquake catalogues have dozens of natural belts, so every method may
# use up to 50 clusters there (coarse grid keeps GMM's BIC search affordable).
QUAKE_K_RANGE = (2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50)
K_RANGES = {"earthquakes_1965_2016": QUAKE_K_RANGE, "earthquakes_2023": QUAKE_K_RANGE}


def _sample(n: int, cap: int) -> int | None:
    return cap if n > cap else None


def _format_params(params: dict) -> str:
    return ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())


def _plot(ds, results: dict, table: pd.DataFrame) -> None:
    title = f"{ds.name} ({ds.extra['owner']}): four clustering methods"
    if "coords_deg" in ds.extra:
        fig = plot_spatial_grid(ds.extra["coords_deg"], results, title=title, reference_title="Points")
    else:
        pca = PCA(n_components=2).fit(ds.X)
        if ds.y is None:
            y, gt_title, ari = np.zeros(len(ds.X), dtype=int), "Data (no labels)", dict.fromkeys(results, "n/a")
        else:
            y, gt_title = ds.y, "Ground truth"
            ari = dict(zip(table["method"], table["adjusted_rand_index"].map("{:.3f}".format)))
        fig = plot_pca_grid(pca.transform(ds.X), results, y, pca.explained_variance_ratio_ * 100, ari,
                            gt_title=gt_title, title=title, ncols=5)
    fig.savefig(FIGURES_DIR / f"{ds.name}.png", dpi=110, bbox_inches="tight")
    matplotlib.pyplot.close(fig)


def run_dataset(ds) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(ds.X)
    sil_sample, dbcv_sample = _sample(n, SILHOUETTE_SAMPLE), _sample(n, DBCV_SAMPLE)

    start = time.perf_counter()
    selections = tuning.select_all(
        ds.X, k=ds.n_true_clusters, k_range=K_RANGES.get(ds.name, tuning.K_RANGE),
        sample_size=sil_sample, dbcv_sample_size=dbcv_sample,
    )
    tuning_seconds = time.perf_counter() - start
    results = run_models(four_models({m: s.params for m, s in selections.items()}), ds.X)

    table = build_results_table(ds.name, ds.X, results, ds.y, silhouette_sample_size=sil_sample)
    table.insert(1, "owner", ds.extra["owner"])
    table.insert(2, "n_points", n)
    table.insert(3, "n_features", ds.X.shape[1])
    table.insert(4, "labelled", ds.y is not None)
    table.insert(6, "params", [_format_params(selections[m].params) for m in table["method"]])
    table["noise_fraction"] = table["noise_points"] / n
    table["dbcv"] = [dbcv_score(ds.X, results[m].labels, dbcv_sample) for m in table["method"]]
    table["tuning_valid"] = [selections[m].valid for m in table["method"]]
    table["label_free_choice"] = [
        _format_params(selections[m].label_free) if selections[m].label_free else "" for m in table["method"]
    ]
    table["tuning_seconds_all_methods"] = tuning_seconds
    _plot(ds, results, table)

    tried = pd.DataFrame([{"dataset": ds.name, "method": m, **t} for m, s in selections.items() for t in s.tried])
    return table, tried


def pivot(results: pd.DataFrame, column: str) -> pd.DataFrame:
    return results.pivot(index="dataset", columns="method", values=column).reindex(
        index=results["dataset"].unique(), columns=METHODS
    )


def main(names: list[str]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    tables, tried = [], []
    for name in names:
        start = time.perf_counter()
        print(f"== {name}", end=" ", flush=True)
        table, t = run_dataset(LOADERS[name]())
        tables.append(table)
        tried.append(t)
        print(f"({time.perf_counter() - start:.0f}s)", flush=True)
        # Save after every dataset so a crash late in the run keeps earlier results.
        pd.concat(tables, ignore_index=True).to_csv(RESULTS_DIR / "team_results.csv", index=False)
        pd.concat(tried, ignore_index=True).to_csv(RESULTS_DIR / "team_tuning.csv", index=False)

    results = pd.concat(tables, ignore_index=True)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        for column, label in [
            ("adjusted_rand_index", "ARI (labelled datasets)"),
            ("silhouette", "Silhouette (non-noise points)"),
            ("dbcv", "DBCV"),
            ("noise_fraction", "Noise fraction"),
            ("n_clusters_found", "Clusters found"),
            ("fit_seconds", "Fit seconds"),
        ]:
            table = pivot(results, column)
            if column == "adjusted_rand_index":
                table = table.dropna(how="all")
            print(f"\n{label}\n{table}")
        invalid = results.loc[~results["tuning_valid"], ["dataset", "method"]]
        if len(invalid):
            print("\nNo setting met the limits (least-noise setting shown):")
            print(invalid.to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    names = sys.argv[1:] or list(LOADERS)
    unknown = set(names) - set(LOADERS)
    if unknown:
        sys.exit(f"Unknown datasets {sorted(unknown)}; choose from {list(LOADERS)}")
    np.random.seed(42)
    main(names)
