"""Facts behind the D3 and D4 paragraphs of the team report. Run after run.py.
Writes results/case_stats.csv (one row per number: dataset, setting, item, value).

D3 Uber: k-distance spread, share of pickups in lower and midtown Manhattan,
and the clusters each chosen setting returns (share, centre, airport if any).
D4 Ecoli: class sizes, and how much of each class each case keeps or recovers.
"""
import json
import os
import numpy as np
import pandas as pd

import data as D
import core as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
SEED = 42
AIRPORTS = {"JFK": (40.645, -73.780), "LaGuardia": (40.774, -73.872)}
MIDTOWN_BOX = {"lat": (40.70, 40.77), "lon": (-74.02, -73.96)}   # lower and midtown Manhattan
rows = []


def add(ds, setting, item, value):
    rows.append({"dataset": ds, "setting": setting, "item": item, "value": value})


# ---- D3 Uber
ub = D.load_uber()
X, deg = ub["X"], ub["X_deg"]
curve = np.array(json.load(open(f"{RES}/kdist_curves.json"))["UberNYC"])
for q in (5, 50, 95):
    add("UberNYC", "k-distance, minPts 8", f"p{q}_km", np.percentile(curve, q))
box = (deg[:, 0] > MIDTOWN_BOX["lat"][0]) & (deg[:, 0] < MIDTOWN_BOX["lat"][1]) & \
      (deg[:, 1] > MIDTOWN_BOX["lon"][0]) & (deg[:, 1] < MIDTOWN_BOX["lon"][1])
add("UberNYC", "data", "share_in_midtown_box", box.mean())

knee = json.load(open(f"{RES}/chosen_params.json"))["kdist"]["UberNYC"]["knee"]
settings = {
    "DBSCAN knee (eps %.3f km, minPts 8)" % knee: C.run_dbscan(X, knee, 8)[0],
    "DBSCAN rule (eps 1.576 km, minPts 32)": C.run_dbscan(X, 1.576078, 32)[0],
    "HDBSCAN rule (mcs 250)": C.run_hdbscan(X, 250)[0],
    "K-Means++ K=5": C.run_kmeans(X, 5, "k-means++", SEED)[0],
}
for name, lab in settings.items():
    add("UberNYC", name, "noise_frac", float((lab == -1).mean()))
    add("UberNYC", name, "n_clusters", int(len(np.unique(lab[lab != -1]))))
    for c in np.unique(lab[lab != -1]):
        m = lab == c
        la, lo = deg[m].mean(0)
        near = [a for a, (pa, po) in AIRPORTS.items() if abs(la - pa) < 0.02 and abs(lo - po) < 0.03]
        tag = f"cluster {c} ({la:.3f}, {lo:.3f}){' ' + near[0] if near else ''}"
        add("UberNYC", name, tag, float(m.mean()))

# ---- D4 Ecoli
ec = D.load_ecoli()
X, y = ec["X"], ec["y"]
sizes = np.bincount(y)
for c, n in enumerate(sizes):
    add("Ecoli", "data", f"class {c} size", int(n))
cases = {
    "HDBSCAN best-ARI (mcs 12)": C.run_hdbscan(X, 12)[0],
    "K-Means++ K=7": C.run_kmeans(X, 7, "k-means++", SEED)[0],
}
for name, lab in cases.items():
    for c, n in enumerate(sizes):
        m = y == c
        add("Ecoli", name, f"class {c} (n={n}) noise_frac", float((lab[m] == -1).mean()))
        kept = lab[m][lab[m] != -1]
        purity = np.bincount(kept).max() / n if len(kept) else 0.0
        add("Ecoli", name, f"class {c} (n={n}) share in its main cluster", float(purity))

# ---- Ablation A: gap of each seed's inertia to the best inertia of all 60 runs
seeds = pd.read_csv(f"{RES}/init_seeds.csv")
for nm in ["UberNYC", "Ecoli"]:
    s = seeds[seeds.dataset == nm]
    best = s.inertia.min()
    for meth, g in s.groupby("method"):
        gap = 100 * (g.inertia / best - 1)
        add(nm, f"Ablation A, {meth}, K={int(g.k.iloc[0])}", "gap_to_best_mean_pct", gap.mean())
        add(nm, f"Ablation A, {meth}, K={int(g.k.iloc[0])}", "gap_to_best_max_pct", gap.max())
        if g.ari.notna().any():
            add(nm, f"Ablation A, {meth}, K={int(g.k.iloc[0])}", "seeds_with_ari_ge_0.7",
                int((g.ari >= 0.7).sum()))

pd.DataFrame(rows).to_csv(f"{RES}/case_stats.csv", index=False)
print(pd.DataFrame(rows).to_string())
