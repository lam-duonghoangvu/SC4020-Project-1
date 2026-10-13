"""K-Means, DBSCAN, HDBSCAN and GMM on every dataset in the report, under one protocol.

Usage:
    python -m sc4020.experiments.team_comparison                 # all 10 datasets
    python -m sc4020.experiments.team_comparison wine ecoli      # a subset

Settings are chosen without labels by tuning.select_all (the shared rule); K-Means and GMM get the
true k on labelled data, and the label-free choice is recorded. Labels are used only for ARI.

Writes results/comparison/team_results.csv, team_tuning.csv, team_params.json (chosen settings) and
figures/team_<dataset>.png.
"""

import json
import sys
import time
import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from sc4020 import generate_figures as gf
from sc4020 import methods, tuning
from sc4020.datasets import DATASETS
from sc4020.dbcv import dbcv_score
from sc4020.paths import results_dir

TOPIC = "comparison"
REPORTED = list(DATASETS)[:10]

# Silhouette and DBCV grow with n^2, so both are estimated on a sample for large data.
SILHOUETTE_SAMPLE = 5000
DBCV_SAMPLE = 3000

# Global earthquake catalogues have dozens of natural belts, so every method may use up to 50
# clusters there (a coarse grid keeps GMM's BIC search affordable).
QUAKE_K_RANGE = (2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50)
K_RANGES = {"earthquakes_1965_2016": QUAKE_K_RANGE, "earthquakes_2023": QUAKE_K_RANGE}


def _sample(n: int, cap: int) -> int | None:
    return cap if n > cap else None


def format_params(params: dict | None) -> str:
    if not params:
        return ""
    return ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())


def run_dataset(ds) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    X, n = ds.data, len(ds.data)
    k_range = K_RANGES.get(ds.name, tuning.K_RANGE)
    sil_sample, dbcv_sample = _sample(n, SILHOUETTE_SAMPLE), _sample(n, DBCV_SAMPLE)

    start = time.perf_counter()
    selections = tuning.select_all(
        X, k=ds.n_true_clusters, k_range=k_range, sample_size=sil_sample, max_clusters=max(k_range), y_true=ds.target
    )
    tuning_seconds = time.perf_counter() - start
    params = {m: s.params for m, s in selections.items()}
    results = methods.run_models(methods.build_models(params), X, params=params)

    table = gf.build_results_table(ds.name, X, results, ds.target, silhouette_sample=sil_sample)
    table.insert(1, "n_points", n)
    table.insert(2, "n_features", X.shape[1])
    table.insert(3, "labelled", ds.target is not None)
    table.insert(5, "params", [format_params(params[m]) for m in table["method"]])
    table["dbcv"] = [dbcv_score(X, results[m].labels, dbcv_sample) for m in table["method"]]
    table["tuning_valid"] = [selections[m].valid for m in table["method"]]
    table["label_free_choice"] = [format_params(selections[m].label_free) for m in table["method"]]
    table["best_ari_choice"] = [format_params(selections[m].best_ari) for m in table["method"]]
    table["tuning_seconds_all_methods"] = tuning_seconds

    title = f"{ds.name}: four clustering methods"
    if "coords_deg" in ds:
        fig = gf.plot_spatial_grid(ds.coords_deg, results, title=title)
    else:
        fig = gf.plot_cluster_grid(X, results, ds.target, title=title, scores=gf.score_labels(table), ncols=5)
    gf.save(fig, TOPIC, f"team_{ds.name}")

    tried = pd.DataFrame([{"dataset": ds.name, "method": m, **t} for m, s in selections.items() for t in s.tried])
    return table, tried, params


def main(names: list[str]) -> None:
    out = results_dir(TOPIC)
    params_path = out / "team_params.json"
    # keep other datasets' settings when only a subset is re-run
    chosen = json.loads(params_path.read_text()) if params_path.exists() else {}
    tables, tried = [], []
    for name in names:
        start = time.perf_counter()
        print(f"== {name}", end=" ", flush=True)
        table, t, chosen[name] = run_dataset(DATASETS[name]())
        tables.append(table)
        tried.append(t)
        print(f"({time.perf_counter() - start:.0f}s)", flush=True)
        # Save after every dataset so a crash late in the run keeps earlier results.
        pd.concat(tables, ignore_index=True).to_csv(out / "team_results.csv", index=False)
        pd.concat(tried, ignore_index=True).to_csv(out / "team_tuning.csv", index=False)
        params_path.write_text(json.dumps(chosen, indent=2, default=str))

    results = pd.concat(tables, ignore_index=True)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        for column in ["ari", "silhouette", "dbcv", "noise_fraction", "n_clusters", "fit_seconds"]:
            if column in results:
                print(f"\n{column}\n{gf.pivot_by_method(results, column)}")
        invalid = results.loc[~results["tuning_valid"], ["dataset", "method"]]
        if len(invalid):
            print("\nNo setting met the limits (least-noise setting shown):")
            print(invalid.to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    names = sys.argv[1:] or REPORTED
    unknown = set(names) - set(DATASETS)
    if unknown:
        sys.exit(f"Unknown datasets {sorted(unknown)}; choose from {list(DATASETS)}")
    np.random.seed(42)
    main(names)
