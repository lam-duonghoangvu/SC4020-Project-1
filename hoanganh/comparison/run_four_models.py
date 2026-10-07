"""Compare K-Means, DBSCAN, HDBSCAN and GMM, one showcase dataset per method.

Usage (from the hoanganh folder):
    python comparison/run_four_models.py

Each dataset has the structure that one method's assumptions fit best:
  digits          -> K-Means   many compact, similar-sized classes in 64-D
  make_moons      -> DBSCAN    non-convex shapes of uniform density
  varied_density  -> HDBSCAN   clusters of very different density plus noise
  breast_cancer   -> GMM       elongated, correlated (ellipsoidal) classes

All four methods run on every dataset with the same label-free tuning, so the
highlighted method has to win on the same terms as the others.
Writes to comparison/results/ and comparison/figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from sc4020 import data, tuning
from sc4020.clustering import four_models, run_models
from sc4020.metrics import build_results_table
from sc4020.viz import plot_pca_grid

RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"

# dataset -> (loader, method it showcases)
SHOWCASES = {
    "digits": (data.load_digits_data, "K-Means"),
    "make_moons": (data.load_moons_data, "DBSCAN"),
    "varied_density": (data.load_varied_density_data, "HDBSCAN"),
    "breast_cancer": (data.load_breast_cancer_data, "GMM"),
}


def _format_params(params: dict) -> str:
    return ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())


def _plot(ds: data.Dataset, results: dict, table: pd.DataFrame, highlight: str) -> None:
    pca = PCA(n_components=2).fit(ds.X)
    ari = dict(zip(table["method"], table["adjusted_rand_index"].map("{:.3f}".format)))
    fig = plot_pca_grid(
        pca.transform(ds.X), results, ds.y, pca.explained_variance_ratio_ * 100, ari,
        gt_title="Ground truth", title=f"{ds.name}: showcase for {highlight}", ncols=5,
    )
    fig.savefig(FIGURES_DIR / f"{ds.name}.png", dpi=120, bbox_inches="tight")


def run_dataset(ds: data.Dataset, highlight: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # K-Means and GMM are given the true k; see tuning.py.
    selections = tuning.select_all(ds.X, k=ds.n_true_clusters)
    results = run_models(four_models({m: s.params for m, s in selections.items()}), ds.X)

    table = build_results_table(ds.name, ds.X, results, ds.y)
    table.insert(1, "highlight", highlight)
    table.insert(3, "params", [_format_params(selections[m].params) for m in table["method"]])
    table.insert(4, "n_points", len(ds.X))
    table["noise_fraction"] = table["noise_points"] / len(ds.X)
    table["tuning_valid"] = [selections[m].valid for m in table["method"]]
    table["label_free_choice"] = [
        _format_params(selections[m].label_free) if selections[m].label_free else ""
        for m in table["method"]
    ]
    _plot(ds, results, table, highlight)

    tried = pd.DataFrame([
        {"dataset": ds.name, "method": m, **t} for m, s in selections.items() for t in s.tried
    ])
    return table, tried


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: the highlighted method's ARI, the best other method, and the margin."""
    rows = []
    for name, g in results.groupby("dataset", sort=False):
        highlight = g["highlight"].iloc[0]
        own = g.loc[g["method"] == highlight, "adjusted_rand_index"].iloc[0]
        others = g[g["method"] != highlight].sort_values("adjusted_rand_index", ascending=False)
        rows.append({
            "dataset": name,
            "highlight": highlight,
            "highlight_ari": own,
            "runner_up": others["method"].iloc[0],
            "runner_up_ari": others["adjusted_rand_index"].iloc[0],
            "margin": own - others["adjusted_rand_index"].iloc[0],
        })
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    tables, tried = [], []
    for name, (load, highlight) in SHOWCASES.items():
        print(f"== {name} (showcase for {highlight})", flush=True)
        table, t = run_dataset(load(), highlight)
        tables.append(table)
        tried.append(t)

    results = pd.concat(tables, ignore_index=True)
    summary = summarize(results)
    results.to_csv(RESULTS_DIR / "four_models_results.csv", index=False)
    summary.to_csv(RESULTS_DIR / "showcase_summary.csv", index=False)
    pd.concat(tried, ignore_index=True).to_csv(RESULTS_DIR / "four_models_tuning.csv", index=False)

    cols = ["dataset", "method", "params", "n_clusters_found", "noise_fraction", "tuning_valid",
            "silhouette", "davies_bouldin", "adjusted_rand_index", "fit_seconds", "label_free_choice"]
    with pd.option_context("display.width", 220, "display.max_colwidth", 45):
        print(results[cols].round(3).to_string(index=False))
        print()
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    np.random.seed(42)
    main()
