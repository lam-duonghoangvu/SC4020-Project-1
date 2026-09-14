"""Stage 4 scalability: runtime against number of sales for each method at its Stage 3 setting.

Usage: python -m experiments.scalability
Writes results/scalability.csv, one row per run, saved after every run so progress survives an interruption.

Features are standardized once on all sales, then random subsets (seed 0) are clustered.
Settings: K-means with K=4 (n_init=1, random and K-means++), DBSCAN and HDBSCAN at the settings the
project rule chose on real data in notebooks/clustering.ipynb (read from results/dbscan_grid.csv and
results/hdbscan_grid.csv).

DBSCAN stores every neighbour within eps for every point. Before each DBSCAN run the number of neighbour
pairs is estimated from 2,000 probe points; runs above MAX_NEIGHBOUR_PAIRS are skipped and recorded as
skipped, to stay within 16 GB of memory.
"""

import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from hdb.features import load_price_index, load_resale, screening_features, standardize
from hdb.methods import dbscan, hdbscan, kmeans_plus_plus, kmeans_random
from hdb.metrics import best_density_setting, timed_fit_predict
from hdb.paths import results_dir
from hdb.screening import cluster_counts

SEED = 0
K = 4
SIZES = (1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, None)  # None: all sales
REPEATS_UP_TO = 20_000  # sizes up to this run 3 times; larger sizes once
MAX_NEIGHBOUR_PAIRS = 150_000_000  # DBSCAN stores neighbourhoods as int64 indices: about 1.2 GB
PROBE_POINTS = 2_000


def chosen_density_settings():
    dbscan_row = best_density_setting(pd.read_csv(results_dir() / "dbscan_grid.csv"))
    hdbscan_row = best_density_setting(pd.read_csv(results_dir() / "hdbscan_grid.csv"))
    min_samples = None if pd.isna(hdbscan_row["min_samples"]) or hdbscan_row["min_samples"] == "None" else int(hdbscan_row["min_samples"])
    return (float(dbscan_row["eps"]), int(dbscan_row["min_samples"])), (int(hdbscan_row["min_cluster_size"]), min_samples)


def estimated_neighbour_pairs(X, eps, rng):
    probe = X[rng.choice(len(X), size=min(PROBE_POINTS, len(X)), replace=False)]
    counts = NearestNeighbors(radius=eps).fit(X).radius_neighbors(probe, return_distance=False)
    return float(np.mean([len(c) for c in counts]) * len(X))


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = screening_features(load_resale(), load_price_index())
    X_all = standardize(features)
    (eps, dbscan_min_samples), (min_cluster_size, hdbscan_min_samples) = chosen_density_settings()
    print(f"{len(X_all)} sales; DBSCAN eps={eps:.4f}, min_samples={dbscan_min_samples}; "
          f"HDBSCAN min_cluster_size={min_cluster_size}, min_samples={hdbscan_min_samples}", flush=True)

    methods = {
        "K-means (random)": lambda: kmeans_random(K, SEED),
        "K-means++": lambda: kmeans_plus_plus(K, SEED),
        "DBSCAN": lambda: dbscan(eps, dbscan_min_samples),
        "HDBSCAN": lambda: hdbscan(min_cluster_size, hdbscan_min_samples),
    }
    target = results_dir() / "scalability.csv"
    rows = []
    rng = np.random.default_rng(SEED)
    for size in SIZES:
        n = len(X_all) if size is None else size
        X = X_all[np.sort(rng.choice(len(X_all), size=n, replace=False))]
        for method, make in methods.items():
            base = {"method": method, "n": n}
            if method == "DBSCAN":
                pairs = estimated_neighbour_pairs(X, eps, np.random.default_rng(SEED))
                base["estimated_neighbour_pairs"] = pairs
                if pairs > MAX_NEIGHBOUR_PAIRS:
                    rows.append({**base, "skipped": f"estimated {pairs:,.0f} neighbour pairs"})
                    pd.DataFrame(rows).to_csv(target, index=False)
                    print(f"{method} n={n}: skipped ({pairs:,.0f} neighbour pairs)", flush=True)
                    continue
            for repeat in range(3 if n <= REPEATS_UP_TO else 1):
                labels, seconds = timed_fit_predict(make(), X)
                rows.append({**base, "repeat": repeat, "seconds": seconds, **cluster_counts(labels)})
                pd.DataFrame(rows).to_csv(target, index=False)
                print(f"{method} n={n} repeat {repeat}: {seconds:.2f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
