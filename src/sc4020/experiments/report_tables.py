"""Shared-column results for Wine (D6) and Breast Cancer (D7), plus the D7 success / failure figure.

Adds what the report's shared table needs: largest-cluster share, NMI and the fake-data (shuffled
columns) silhouette, at the settings the shared rule picks (the same call as team_comparison), plus
Agglomerative (Ward) and Spectral (nearest-neighbour graph) at the number of classes.

Usage: python -m sc4020.experiments.report_tables
Writes results/comparison/report_d6_d7.csv and report/figures/bc_cases.png.
"""

import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from sc4020 import generate_figures as gf
from sc4020 import methods, tuning
from sc4020.datasets import load_breast_cancer, load_wine
from sc4020.metrics import evaluate
from sc4020.paths import REPORT_FIGURES_DIR, results_dir
from sc4020.screening import shuffled_null

NULL_SEED = 0  # seed of the column shuffle


def models(X, k):
    """The shared rule's four methods plus Agglomerative and Spectral at k."""
    params = {m: s.params for m, s in tuning.select_all(X, k=k).items()}
    built = methods.build_models(params)
    built["Agglomerative"] = methods.agglomerative(k)
    built["Spectral"] = methods.spectral(k, affinity="nearest_neighbors")
    params["Agglomerative"] = {"n_clusters": k, "linkage": "ward"}
    params["Spectral"] = {"n_clusters": k, "affinity": "nearest_neighbors"}
    return built, params


def run(ds):
    X, y, k = ds.data, ds.target, ds.n_true_clusters
    X_null = shuffled_null(X, np.random.default_rng(NULL_SEED))
    built, params = models(X, k)
    null_models, _ = models(X_null, k)
    rows, labels_by_method = [], {}
    for name, model in built.items():
        labels, seconds = methods.fit_predict_timed(model, X)
        null = evaluate(X_null, null_models[name].fit_predict(X_null))
        rows.append({"dataset": ds.name, "method": name, "params": params[name], **evaluate(X, labels, y),
                     "fit_seconds": seconds, "null_silhouette": null["silhouette"],
                     "null_clusters": null["n_clusters"], "null_noise_fraction": null["noise_fraction"]})
        labels_by_method[name] = labels
    return rows, labels_by_method


def main():
    warnings.filterwarnings("ignore")
    rows_wine, _ = run(load_wine())
    bc = load_breast_cancer()
    rows_bc, labels_bc = run(bc)
    table = pd.DataFrame(rows_wine + rows_bc)
    table.to_csv(results_dir("comparison") / "report_d6_d7.csv", index=False)
    fig = gf.plot_case_figure(bc.data, bc.target, labels_bc, rows_bc, picks=("K-Means", "GMM", "DBSCAN"))
    gf.save_to(fig, REPORT_FIGURES_DIR / "bc_cases.png")
    print(table.drop(columns=["params"]).round(3).to_string())


if __name__ == "__main__":
    main()
