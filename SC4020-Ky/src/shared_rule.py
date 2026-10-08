"""Ky's datasets (D3 Uber, D4 Ecoli) under the team's shared selection rule
(report/03b-protocol.tex), in the shared table columns of report/03c-comparison.tex.

DBSCAN / HDBSCAN: among settings in results/sweep.csv with >=2 clusters and
<=50% noise, the highest silhouette. K-Means: D3 uses K=5, where the gap to
fake data is larger than at the silhouette's K=2 (results/fake_data_check.csv);
D4 uses the silhouette's K. On D4 the best-ARI setting is added too, so the
cost of choosing without labels is visible.

Each setting is refitted to get the largest-cluster share, the fit time and the
silhouette on fake data (mean of 3): shuffled columns for K-Means and for
tables, uniform points in the bounding box for density methods on coordinates.
Run after run.py. Writes results/shared_rule.csv.
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import data as D
import core as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
SEED = 42
N_FAKE = 3
K_D3 = 5
COLS = ["n_clusters", "noise_frac", "silhouette", "davies_bouldin", "ari", "nmi"]


def shuffled(X, rng):
    return np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])


def uniform_box(X, rng):
    return rng.uniform(X.min(0), X.max(0), size=X.shape)


def fitter(method, p):
    if method == "KMeans":
        return lambda Z: C.run_kmeans(Z, p["k"], "random", SEED, n_init=10)
    if method == "KMeans++":
        return lambda Z: C.run_kmeans(Z, p["k"], "k-means++", SEED, n_init=10)
    if method == "DBSCAN":
        return lambda Z: C.run_dbscan(Z, p["eps"], p["min_samples"])
    return lambda Z: C.run_hdbscan(Z, p["min_cluster_size"])


sweep = pd.read_csv(f"{RES}/sweep.csv")
rows = []
for ds in [D.load_uber(), D.load_ecoli()]:
    X, y, nm = ds["X"], ds["y"], ds["name"]
    rng = np.random.default_rng(SEED)
    fake_sh = [shuffled(X, rng) for _ in range(N_FAKE)]
    fake_dn = [uniform_box(X, rng) for _ in range(N_FAKE)] if ds["spatial"] else fake_sh

    km = sweep[(sweep.dataset == nm) & (sweep.method == "KMeans++")]
    k_sil = json.loads(km.loc[km.silhouette.idxmax(), "params"])["k"]
    picks = []
    for meth in ["KMeans", "KMeans++"]:
        picks.append((meth, "rule", {"k": K_D3 if ds["spatial"] else k_sil}))
    if y is not None:
        picks.append(("KMeans++", "best_ari",
                      json.loads(km.loc[km.ari.idxmax(), "params"])))
    for meth in ["DBSCAN", "HDBSCAN"]:
        g = sweep[(sweep.dataset == nm) & (sweep.method == meth)]
        ok = g[(g.n_clusters >= 2) & (g.noise_frac <= 0.5) & g.silhouette.notna()]
        picks.append((meth, "rule", json.loads(ok.loc[ok.silhouette.idxmax(), "params"])))
        if y is not None:
            picks.append((meth, "best_ari", json.loads(g.loc[g.ari.idxmax(), "params"])))

    for meth, choice, p in picks:
        fit = fitter(meth, p)
        lab, rt, _ = fit(X)
        m = C.evaluate(X, lab, y, SEED)
        kept = lab[lab != -1]
        largest = np.bincount(kept).max() / len(lab) if len(kept) else np.nan
        fakes = fake_sh if meth.startswith("KMeans") else fake_dn
        fsil = [C.evaluate(Z, fit(Z)[0], None, SEED)["silhouette"] for Z in fakes]
        rows.append({"dataset": nm, "method": meth, "choice": choice,
                     "params": json.dumps(p), **{c: m[c] for c in COLS},
                     "largest_frac": largest, "runtime_s": rt,
                     "null_silhouette": np.nanmean(fsil) if np.isfinite(fsil).any() else np.nan,
                     "fake": "shuffled" if fakes is fake_sh else "uniform_box"})

out = pd.DataFrame(rows)
out.to_csv(f"{RES}/shared_rule.csv", index=False)
print(out.drop(columns=["fake"]).round(3).to_string())
e = out[(out.dataset == "Ecoli") & (out.choice == "rule")]
print("Ecoli, rule choice: Spearman(silhouette, ARI) =",
      round(spearmanr(e.silhouette, e.ari)[0], 3))
