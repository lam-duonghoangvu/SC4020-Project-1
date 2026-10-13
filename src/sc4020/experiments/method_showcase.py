"""One showcase dataset per method, each with the structure that method's assumptions fit best:

  digits          -> K-Means   many compact, similar-sized classes in 64-D
  make_moons      -> DBSCAN    non-convex shapes of uniform density
  varied_density  -> HDBSCAN   clusters of very different density plus noise
  breast_cancer   -> GMM       elongated, correlated (ellipsoidal) classes

All four methods run on every dataset with the same label-free tuning, so the highlighted method
has to win on the same terms as the others.

Usage: python -m sc4020.experiments.method_showcase
Writes results/comparison/showcase_results.csv, showcase_summary.csv, showcase_tuning.csv and
figures/showcase_<dataset>.png.
"""

import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from sc4020 import generate_figures as gf
from sc4020 import methods, tuning
from sc4020.datasets import load_breast_cancer, load_digits, make_moons, make_varied_density
from sc4020.experiments.team_comparison import format_params
from sc4020.paths import results_dir

TOPIC = "comparison"
SHOWCASES = {
    "digits": (load_digits, "K-Means"),
    "make_moons": (make_moons, "DBSCAN"),
    "varied_density": (make_varied_density, "HDBSCAN"),
    "breast_cancer": (load_breast_cancer, "GMM"),
}


def run_dataset(ds, highlight: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # K-Means and GMM are given the true k (tuning.select_all).
    selections = tuning.select_all(ds.data, k=ds.n_true_clusters, y_true=ds.target)
    params = {m: s.params for m, s in selections.items()}
    results = methods.run_models(methods.build_models(params), ds.data, params=params)

    table = gf.build_results_table(ds.name, ds.data, results, ds.target)
    table.insert(1, "highlight", highlight)
    table.insert(3, "params", [format_params(params[m]) for m in table["method"]])
    table.insert(4, "n_points", len(ds.data))
    table["tuning_valid"] = [selections[m].valid for m in table["method"]]
    table["label_free_choice"] = [format_params(selections[m].label_free) for m in table["method"]]
    fig = gf.plot_cluster_grid(ds.data, results, ds.target, title=f"{ds.name}: showcase for {highlight}",
                               scores=gf.score_labels(table), ncols=5)
    gf.save(fig, TOPIC, f"showcase_{ds.name}")

    tried = pd.DataFrame([{"dataset": ds.name, "method": m, **t} for m, s in selections.items() for t in s.tried])
    return table, tried


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: the highlighted method's ARI, the best other method, and the margin."""
    rows = []
    for name, g in results.groupby("dataset", sort=False):
        highlight = g["highlight"].iloc[0]
        own = g.loc[g["method"] == highlight, "ari"].iloc[0]
        others = g[g["method"] != highlight].sort_values("ari", ascending=False)
        rows.append({
            "dataset": name, "highlight": highlight, "highlight_ari": own,
            "runner_up": others["method"].iloc[0], "runner_up_ari": others["ari"].iloc[0],
            "margin": own - others["ari"].iloc[0],
        })
    return pd.DataFrame(rows)


def main() -> None:
    out = results_dir(TOPIC)
    tables, tried = [], []
    for name, (load, highlight) in SHOWCASES.items():
        print(f"== {name} (showcase for {highlight})", flush=True)
        table, t = run_dataset(load(), highlight)
        tables.append(table)
        tried.append(t)
    results = pd.concat(tables, ignore_index=True)
    summary = summarize(results)
    results.to_csv(out / "showcase_results.csv", index=False)
    summary.to_csv(out / "showcase_summary.csv", index=False)
    pd.concat(tried, ignore_index=True).to_csv(out / "showcase_tuning.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_colwidth", 45):
        cols = ["dataset", "method", "params", "n_clusters", "noise_fraction", "silhouette", "ari", "fit_seconds"]
        print(results[cols].round(3).to_string(index=False))
        print()
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    main()
