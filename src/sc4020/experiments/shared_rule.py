"""Uber NYC and Ecoli in the shared table columns (report, Section 3.3), at the settings the shared
rule chose in experiments.hotspot and, on labelled Ecoli, at the best-ARI settings too, so the cost of
choosing without labels is visible.

Each setting is refitted for the fit time and the fake-data silhouette (mean of 3): shuffled columns
for K-Means and tables, uniform points in the bounding box for density methods on coordinates.

Usage (after experiments.hotspot): python -m sc4020.experiments.shared_rule
Writes results/hotspot/shared_rule.csv.
"""

import json
import time
import warnings

import pandas as pd
from scipy.stats import spearmanr

from sc4020 import methods
from sc4020.datasets import load_ecoli, load_uber_pickups
from sc4020.metrics import evaluate
from sc4020.paths import results_dir
from sc4020.screening import null_gap

SEED = 42
N_FAKE = 3
COLUMNS = ["n_clusters", "noise_fraction", "largest_cluster_share", "silhouette", "davies_bouldin", "ari", "nmi"]


def fitter(method, p):
    if method == "K-Means (random init)":
        return lambda Z: methods.kmeans(p["n_clusters"], init="random", random_state=SEED).fit_predict(Z)
    if method == "K-Means":
        return lambda Z: methods.kmeans(p["n_clusters"], random_state=SEED).fit_predict(Z)
    if method == "DBSCAN":
        return lambda Z: methods.dbscan(**p).fit_predict(Z)
    return lambda Z: methods.hdbscan(**p).fit_predict(Z)


def main():
    warnings.filterwarnings("ignore")
    out = results_dir("hotspot")
    chosen = json.loads((out / "chosen_params.json").read_text())
    rows = []
    for ds in [load_uber_pickups(), load_ecoli()]:
        spatial = "coords_deg" in ds
        k = chosen["k"][ds.name]
        picks = [(m, "rule", {"n_clusters": k["selected"]}) for m in ("K-Means (random init)", "K-Means")]
        if ds.target is not None:
            picks.append(("K-Means", "best_ari", {"n_clusters": k["best_ari"]}))
        for method, key in [("DBSCAN", "dbscan"), ("HDBSCAN", "hdbscan")]:
            picks.append((method, "rule", chosen[key][ds.name]["rule"]))
            if ds.target is not None:
                picks.append((method, "best_ari", chosen[key][ds.name]["best_ari"]))

        for method, choice, p in picks:
            fit = fitter(method, p)
            start = time.perf_counter()
            labels = fit(ds.data)
            seconds = time.perf_counter() - start
            m = evaluate(ds.data, labels, ds.target, seed=SEED)
            null = "uniform_box" if spatial and method in ("DBSCAN", "HDBSCAN") else "shuffled"
            gap = null_gap(ds.data, fit, null=null, n_nulls=N_FAKE, seed=SEED)
            rows.append({"dataset": ds.name, "method": method, "choice": choice, "params": json.dumps(p),
                         **{c: m.get(c) for c in COLUMNS}, "fit_seconds": seconds,
                         "null_silhouette": gap["null_silhouette"], "null": null})

    table = pd.DataFrame(rows)
    table.to_csv(out / "shared_rule.csv", index=False)
    print(table.drop(columns=["null"]).round(3).to_string())
    e = table[(table.dataset == "ecoli") & (table.choice == "rule")]
    print("Ecoli, rule choice: Spearman(silhouette, ARI) =", round(spearmanr(e.silhouette, e.ari)[0], 3))


if __name__ == "__main__":
    main()
