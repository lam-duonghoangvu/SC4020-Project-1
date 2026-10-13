"""Runtime against number of HDB sales for each method at the setting the shared rule chose.

Usage (after experiments.team_comparison): python -m sc4020.experiments.hdb_scalability
Writes results/hdb_resale/scalability.csv, one row per run, saved after every run so progress
survives an interruption.

All sales go through the same preprocessing as load_hdb_resale (ties jittered, standardized), then
random subsets (seed 0) are clustered. K-Means runs with random and K-Means++ starts (n_init=1);
DBSCAN and HDBSCAN use the hdb_resale settings in results/comparison/team_params.json.

DBSCAN stores every neighbour within eps for every point. Before each DBSCAN run the number of
neighbour pairs is estimated from 2,000 probe points; runs above MAX_NEIGHBOUR_PAIRS are skipped
and recorded as skipped, to stay within 16 GB of memory.
"""

import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from sc4020 import methods
from sc4020.datasets import load_hdb_resale
from sc4020.paths import results_dir
from sc4020.screening import cluster_counts

SEED = 0
SIZES = (1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, None)  # None: all sales
REPEATS_UP_TO = 20_000  # sizes up to this run 3 times; larger sizes once
MAX_NEIGHBOUR_PAIRS = 150_000_000  # DBSCAN stores neighbourhoods as int64 indices: about 1.2 GB
PROBE_POINTS = 2_000


def chosen_settings() -> dict:
    path = results_dir("comparison") / "team_params.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run: python -m sc4020.experiments.team_comparison hdb_resale")
    return json.loads(path.read_text())["hdb_resale"]


def estimated_neighbour_pairs(X, eps, rng):
    probe = X[rng.choice(len(X), size=min(PROBE_POINTS, len(X)), replace=False)]
    counts = NearestNeighbors(radius=eps).fit(X).radius_neighbors(probe, return_distance=False)
    return float(np.mean([len(c) for c in counts]) * len(X))


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_all = load_hdb_resale(n_samples=None, random_state=SEED).data
    chosen = chosen_settings()
    k, db, hdb = chosen["K-Means"]["n_clusters"], chosen["DBSCAN"], chosen["HDBSCAN"]
    print(f"{len(X_all)} sales; K={k}; DBSCAN {db}; HDBSCAN {hdb}", flush=True)

    makers = {
        "K-Means (random init)": lambda: methods.kmeans(k, init="random", n_init=1, random_state=SEED),
        "K-Means": lambda: methods.kmeans(k, n_init=1, random_state=SEED),
        "DBSCAN": lambda: methods.dbscan(**db),
        "HDBSCAN": lambda: methods.hdbscan(**hdb),
    }
    target = results_dir("hdb_resale") / "scalability.csv"
    rows = []
    rng = np.random.default_rng(SEED)
    for size in SIZES:
        n = len(X_all) if size is None else size
        X = X_all[np.sort(rng.choice(len(X_all), size=n, replace=False))]
        for method, make in makers.items():
            base = {"method": method, "n": n}
            if method == "DBSCAN":
                pairs = estimated_neighbour_pairs(X, db["eps"], np.random.default_rng(SEED))
                base["estimated_neighbour_pairs"] = pairs
                if pairs > MAX_NEIGHBOUR_PAIRS:
                    rows.append({**base, "skipped": f"estimated {pairs:,.0f} neighbour pairs"})
                    pd.DataFrame(rows).to_csv(target, index=False)
                    print(f"{method} n={n}: skipped ({pairs:,.0f} neighbour pairs)", flush=True)
                    continue
            for repeat in range(3 if n <= REPEATS_UP_TO else 1):
                labels, seconds = methods.fit_predict_timed(make(), X)
                rows.append({**base, "repeat": repeat, "fit_seconds": seconds, **cluster_counts(labels)})
                pd.DataFrame(rows).to_csv(target, index=False)
                print(f"{method} n={n} repeat {repeat}: {seconds:.2f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
