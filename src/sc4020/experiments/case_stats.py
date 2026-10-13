"""Facts behind the Uber (D3) and Ecoli (D4) case paragraphs of the report, one row per number.

Uber: k-distance spread, share of pickups in lower and midtown Manhattan, and the clusters each
chosen setting returns (share, centre, airport if any). Ecoli: class sizes, and how much of each
class each case keeps or recovers. Ablation A: gap of each seed's inertia to the best one.

Usage (after experiments.hotspot): python -m sc4020.experiments.case_stats
Writes results/hotspot/case_stats.csv.
"""

import json

import numpy as np
import pandas as pd

from sc4020 import methods
from sc4020.datasets import load_ecoli, load_uber_pickups
from sc4020.metrics import NOISE_LABEL
from sc4020.paths import results_dir

SEED = 42
AIRPORTS = {"JFK": (40.645, -73.780), "LaGuardia": (40.774, -73.872)}
MIDTOWN_BOX = {"lat": (40.70, 40.77), "lon": (-74.02, -73.96)}  # lower and midtown Manhattan


def main():
    out = results_dir("hotspot")
    chosen = json.loads((out / "chosen_params.json").read_text())
    curves = json.loads((out / "kdist_curves.json").read_text())
    rows = []

    def add(dataset, setting, item, value):
        rows.append({"dataset": dataset, "setting": setting, "item": item, "value": value})

    uber = load_uber_pickups()
    X, deg = uber.data, uber.coords_deg
    kd = chosen["kdist"]["uber_pickups"]
    for q in (5, 50, 95):
        add("uber_pickups", f"k-distance, min_samples {kd['min_samples']}", f"p{q}_km", float(np.percentile(curves["uber_pickups"], q)))
    (lat_lo, lat_hi), (lon_lo, lon_hi) = MIDTOWN_BOX["lat"], MIDTOWN_BOX["lon"]
    box = (deg[:, 0] > lat_lo) & (deg[:, 0] < lat_hi) & (deg[:, 1] > lon_lo) & (deg[:, 1] < lon_hi)
    add("uber_pickups", "data", "share_in_midtown_box", float(box.mean()))

    db, hdb, k = chosen["dbscan"]["uber_pickups"]["rule"], chosen["hdbscan"]["uber_pickups"]["rule"], chosen["k"]["uber_pickups"]["selected"]
    settings = {
        f"DBSCAN knee (eps {kd['knee']:.3f} km, min_samples {kd['min_samples']})": methods.dbscan(kd["knee"], kd["min_samples"]),
        f"DBSCAN rule (eps {db['eps']:.3f} km, min_samples {db['min_samples']})": methods.dbscan(**db),
        f"HDBSCAN rule (mcs {hdb['min_cluster_size']}, min_samples {hdb['min_samples']})": methods.hdbscan(**hdb),
        f"K-Means K={k}": methods.kmeans(k, random_state=SEED),
    }
    for name, model in settings.items():
        labels = model.fit_predict(X)
        add("uber_pickups", name, "noise_fraction", float((labels == NOISE_LABEL).mean()))
        add("uber_pickups", name, "n_clusters", int(len(np.unique(labels[labels != NOISE_LABEL]))))
        for c in np.unique(labels[labels != NOISE_LABEL]):
            members = labels == c
            lat, lon = deg[members].mean(0)
            near = [a for a, (pa, po) in AIRPORTS.items() if abs(lat - pa) < 0.02 and abs(lon - po) < 0.03]
            add("uber_pickups", name, f"cluster {c} ({lat:.3f}, {lon:.3f}){' ' + near[0] if near else ''}", float(members.mean()))

    ecoli = load_ecoli()
    X, y = ecoli.data, ecoli.target
    sizes = np.bincount(y)
    for c, n in enumerate(sizes):
        add("ecoli", "data", f"class {c} ({ecoli.target_names[c]}) size", int(n))
    hdb_best, k_best = chosen["hdbscan"]["ecoli"]["best_ari"], chosen["k"]["ecoli"]["best_ari"]
    cases = {
        f"HDBSCAN best-ARI (mcs {hdb_best['min_cluster_size']}, min_samples {hdb_best['min_samples']})": methods.hdbscan(**hdb_best).fit_predict(X),
        f"K-Means K={k_best} (best ARI)": methods.kmeans(k_best, random_state=SEED).fit_predict(X),
    }
    for name, labels in cases.items():
        for c, n in enumerate(sizes):
            members = y == c
            add("ecoli", name, f"class {c} (n={n}) noise_fraction", float((labels[members] == NOISE_LABEL).mean()))
            kept = labels[members][labels[members] != NOISE_LABEL]
            add("ecoli", name, f"class {c} (n={n}) share in its main cluster", float(np.bincount(kept).max() / n) if len(kept) else 0.0)

    seeds = pd.read_csv(out / "init_seeds.csv")
    for name in ["uber_pickups", "ecoli"]:
        s = seeds[seeds.dataset == name]
        best = s.inertia.min()
        for method, g in s.groupby("method"):
            setting = f"Ablation A, {method}, K={int(g.k.iloc[0])}"
            gap = 100 * (g.inertia / best - 1)
            add(name, setting, "gap_to_best_mean_pct", float(gap.mean()))
            add(name, setting, "gap_to_best_max_pct", float(gap.max()))
            if g.ari.notna().any():
                add(name, setting, "seeds_with_ari_ge_0.7", int((g.ari >= 0.7).sum()))

    table = pd.DataFrame(rows)
    table.to_csv(out / "case_stats.csv", index=False)
    print(table.to_string())


if __name__ == "__main__":
    main()
