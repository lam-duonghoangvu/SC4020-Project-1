"""Ky's datasets under the team's shared selection rule (report/03b-protocol.tex).

DBSCAN / HDBSCAN: among settings with >=2 clusters and <=50% noise, highest
silhouette (read from results/sweep.csv). K-Means: k with the highest
silhouette in the sweep. On Ecoli the best-ARI setting is written too, so the
cost of choosing without labels is visible. Run after run.py.
Writes results/shared_rule.csv.
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
COLS = ["n_clusters", "noise_frac", "silhouette", "davies_bouldin", "ari", "nmi"]

sweep = pd.read_csv(f"{RES}/sweep.csv")
rows = []
for ds in [D.load_uber(), D.load_ecoli()]:
    X, y, nm = ds["X"], ds["y"], ds["name"]
    g = sweep[(sweep.dataset == nm) & (sweep.method == "KMeans++")]
    k = json.loads(g.loc[g.silhouette.idxmax(), "params"])["k"]
    for meth, init in [("KMeans", "random"), ("KMeans++", "k-means++")]:
        lab, rt, _ = C.run_kmeans(X, k, init, SEED, n_init=10)
        m = C.evaluate(X, lab, y, SEED)
        rows.append({"dataset": nm, "method": meth, "choice": "rule",
                     "params": json.dumps({"k": k}), "runtime_s": rt,
                     **{c: m[c] for c in COLS}})
    for meth in ["DBSCAN", "HDBSCAN"]:
        g = sweep[(sweep.dataset == nm) & (sweep.method == meth)]
        ok = g[(g.n_clusters >= 2) & (g.noise_frac <= 0.5) & g.silhouette.notna()]
        picks = [("rule", ok.loc[ok.silhouette.idxmax()])]
        if y is not None:
            picks.append(("best_ari", g.loc[g.ari.idxmax()]))
        for choice, r in picks:
            rows.append({"dataset": nm, "method": meth, "choice": choice,
                         "params": r["params"], "runtime_s": r["runtime_s"],
                         **{c: r[c] for c in COLS}})

out = pd.DataFrame(rows)
out.to_csv(f"{RES}/shared_rule.csv", index=False)
print(out.round(3).to_string())
e = out[(out.dataset == "Ecoli") & (out.choice == "rule")]
print("Ecoli, rule choice: Spearman(silhouette, ARI) =",
      round(spearmanr(e.silhouette, e.ari)[0], 3))
