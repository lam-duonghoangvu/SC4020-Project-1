"""Main experiment driver. Writes every result to results/*.csv."""
import json
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import data as D
import core as C

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
import warnings; warnings.filterwarnings("ignore")
SEED = 42
R = 6371.0088
rows_main, rows_sweep, rows_abl = [], [], []


def rec(store, ds, method, params, labels, rt, X_eval, y, extra=None):
    m = C.evaluate(X_eval, labels, y, seed=SEED)
    m.update({"dataset": ds, "method": method,
              "params": json.dumps(params, sort_keys=True), "runtime_s": rt})
    if extra:
        m.update(extra)
    store.append(m)
    return m


def best_dbscan_ari(X, y, ms=5, n_grid=30):
    """Grid-search eps around the k-distance knee and return the best-ARI run.
    Used inside ablations so that DBSCAN is always given its own best shot."""
    curve = C.k_distance_curve(X, ms)
    _, e = C.knee(curve)
    best = (-np.inf, None, None)
    for g in np.linspace(0.10 * e, 2.5 * e, n_grid):
        lab, _, _ = C.run_dbscan(X, g, ms)
        m = C.evaluate(X, lab, y, SEED)
        if m["n_clusters"] >= 2 and m["ari"] > best[0]:
            best = (m["ari"], g, m)
    if best[1] is None:                      # no multi-cluster solution exists
        lab, _, _ = C.run_dbscan(X, e, ms)
        return e, C.evaluate(X, lab, y, SEED)
    return best[1], best[2]


def score_solution(m, y):
    """Selection criterion.

    With ground truth we maximise ARI. Without it we must use an internal
    index, but an unconstrained silhouette search collapses to a trivial
    two-cluster solution that discards most of the data. We therefore impose
    the operational requirements of the application - at least three hotspots
    must be returned and at most 40% of events may be discarded as noise -
    and maximise silhouette within that feasible region.
    """
    if y is not None:
        return m["ari"]
    if not (3 <= m["n_clusters"] <= 150):
        return -np.inf
    if m["noise_frac"] > 0.40:
        return -np.inf
    return m["silhouette"] if np.isfinite(m["silhouette"]) else -np.inf


# ===========================================================================
print("loading datasets ...")
eq = D.load_earthquakes()
ub = D.load_uber()
ec = D.load_ecoli()
DSETS = [eq, ub, ec]
for d in DSETS:
    print(f"  {d['name']:12s} n={d['n']:6d} d={d['d']}")

# ===========================================================================
# 1. K-MEANS : k sweep  (elbow + silhouette + ARI where available)
# ===========================================================================
print("\n[1] k sweep for K-Means")
KGRID = list(range(2, 26))
best_k = {}
for ds in DSETS:
    X, y = ds["X"], ds["y"]
    inertias = []
    for k in KGRID:
        lab, rt, ex = C.run_kmeans(X, k, "k-means++", SEED, n_init=10)
        m = rec(rows_sweep, ds["name"], "KMeans++", {"k": k}, lab, rt, X, y,
                {"inertia": ex["inertia"], "n_iter": ex["n_iter"]})
        inertias.append(ex["inertia"])
    sub = pd.DataFrame(rows_sweep)
    sub = sub[(sub.dataset == ds["name"]) & (sub.method == "KMeans++")]
    k_elbow = C.elbow_k(inertias, KGRID)
    k_sil = KGRID[int(np.nanargmax(sub["silhouette"].to_numpy()))]
    # selected k uses the SAME protocol as the density methods
    scores = [score_solution(r, y) for r in
              sub.to_dict("records")]
    k_sel = KGRID[int(np.argmax(scores))]
    best_k[ds["name"]] = {"elbow": int(k_elbow), "silhouette": int(k_sil),
                          "selected": int(k_sel)}
    print(f"  {ds['name']:12s} elbow k={k_elbow:3d}  unconstrained-silhouette k={k_sil:3d}"
          f"  selected k={k_sel:3d}")

# ===========================================================================
# 2. DBSCAN : k-distance knee, then eps x min_samples grid
# ===========================================================================
print("\n[2] DBSCAN grid")
dbscan_best = {}
kdist_store = {}
for ds in DSETS:
    X, y = ds["X"], ds["y"]
    ms_grid = [4, 8, 16, 32] if ds["n"] > 1000 else [3, 5, 8, 12]
    base_ms = ms_grid[1]
    curve = C.k_distance_curve(X, base_ms)
    _, eps_knee = C.knee(curve)
    kdist_store[ds["name"]] = {"curve": curve.tolist(), "knee": eps_knee,
                               "min_samples": base_ms}
    eps_grid = [eps_knee * f for f in (0.10, 0.15, 0.22, 0.32, 0.45,
                                       0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0)]
    best, best_key = None, -np.inf
    for ms in ms_grid:
        for eps in eps_grid:
            lab, rt, _ = C.run_dbscan(X, eps, ms)
            m = rec(rows_sweep, ds["name"], "DBSCAN",
                    {"eps": round(eps, 6), "min_samples": ms}, lab, rt, X, y)
            key = score_solution(m, y)
            if key > best_key:
                best_key, best = key, (eps, ms, m)
    assert best is not None, f"no feasible DBSCAN solution for {ds['name']}"
    dbscan_best[ds["name"]] = {"eps": best[0], "min_samples": best[1]}
    print(f"  {ds['name']:12s} knee eps={eps_knee:.4g}  best eps={best[0]:.4g} "
          f"ms={best[1]}  clusters={best[2]['n_clusters']} "
          f"noise={best[2]['noise_frac']:.2f}")

# ===========================================================================
# 3. HDBSCAN : min_cluster_size sweep
# ===========================================================================
print("\n[3] HDBSCAN sweep")
hdb_best = {}
for ds in DSETS:
    X, y = ds["X"], ds["y"]
    grid = ([10, 15, 25, 40, 60, 100, 150, 250] if ds["n"] > 5000 else
            [10, 15, 25, 40, 60, 100] if ds["n"] > 1000 else [5, 8, 12, 20, 30])
    best, best_key = None, -np.inf
    for mcs in grid:
        lab, rt, _ = C.run_hdbscan(X, mcs)
        m = rec(rows_sweep, ds["name"], "HDBSCAN", {"min_cluster_size": mcs},
                lab, rt, X, y)
        key = score_solution(m, y)
        if key > best_key:
            best_key, best = key, (mcs, m)
    assert best is not None, f"no feasible HDBSCAN solution for {ds['name']}"
    hdb_best[ds["name"]] = {"min_cluster_size": best[0]}
    print(f"  {ds['name']:12s} best mcs={best[0]}  clusters={best[1]['n_clusters']} "
          f"noise={best[1]['noise_frac']:.2f}")

# ===========================================================================
# 4. MAIN COMPARISON TABLE  (tuned settings, 4 methods x 3 datasets)
# ===========================================================================
print("\n[4] main comparison")
best_labels = {}
for ds in DSETS:
    X, y, nm = ds["X"], ds["y"], ds["name"]
    k = best_k[nm]["selected"]
    for meth, init in [("KMeans", "random"), ("KMeans++", "k-means++")]:
        lab, rt, ex = C.run_kmeans(X, k, init, SEED, n_init=10)
        rec(rows_main, nm, meth, {"k": k}, lab, rt, X, y,
            {"inertia": ex["inertia"]})
        best_labels[(nm, meth)] = lab
    p = dbscan_best[nm]
    lab, rt, _ = C.run_dbscan(X, p["eps"], p["min_samples"])
    rec(rows_main, nm, "DBSCAN", p, lab, rt, X, y)
    best_labels[(nm, "DBSCAN")] = lab
    p = hdb_best[nm]
    lab, rt, _ = C.run_hdbscan(X, p["min_cluster_size"])
    rec(rows_main, nm, "HDBSCAN", p, lab, rt, X, y)
    best_labels[(nm, "HDBSCAN")] = lab

# ===========================================================================
# 5. ABLATION A - initialisation, 30 seeds
# ===========================================================================
print("\n[5] ablation A: initialisation (30 seeds)")
for ds in DSETS:
    X, nm = ds["X"], ds["name"]
    k = best_k[nm]["selected"]
    for init, label in [("random", "KMeans"), ("k-means++", "KMeans++")]:
        inert, iters, times = [], [], []
        for s in range(30):
            lab, rt, ex = C.run_kmeans(X, k, init, s, n_init=1)
            inert.append(ex["inertia"]); iters.append(ex["n_iter"]); times.append(rt)
        rows_abl.append({"ablation": "init", "dataset": nm, "method": label,
                         "k": k, "inertia_mean": np.mean(inert),
                         "inertia_std": np.std(inert), "inertia_min": np.min(inert),
                         "inertia_cv_pct": 100 * np.std(inert) / np.mean(inert),
                         "iter_mean": np.mean(iters), "iter_std": np.std(iters),
                         "runtime_mean_s": np.mean(times)})
        print(f"  {nm:12s} {label:9s} inertia {np.mean(inert):.4g}"
              f" +/- {np.std(inert):.3g}  iters {np.mean(iters):.1f}")

# ===========================================================================
# 6. ABLATION B - distance metric on spatial data
# ===========================================================================
print("\n[6] ablation B: distance metric (spatial)")
for ds in [eq, ub]:
    nm = ds["name"]
    p = dbscan_best[nm]
    eps_km, ms = p["eps"], p["min_samples"]
    # (i) haversine, eps expressed in km -> radians
    lab, rt, _ = C.run_dbscan(ds["X_rad"], eps_km / R, ms, metric="haversine")
    m = C.evaluate(ds["X"], lab, None, SEED)
    rows_abl.append({"ablation": "metric", "dataset": nm, "variant": "haversine",
                     "eps": eps_km, "min_samples": ms, "runtime_s": rt, **m})
    # (ii) naive euclidean on raw degrees, eps converted deg
    eps_deg = eps_km / 111.32
    lab2, rt2, _ = C.run_dbscan(ds["X_deg"], eps_deg, ms)
    m2 = C.evaluate(ds["X"], lab2, None, SEED)
    rows_abl.append({"ablation": "metric", "dataset": nm, "variant": "euclidean_deg",
                     "eps": eps_deg, "min_samples": ms, "runtime_s": rt2, **m2})
    # (iii) projected km euclidean (the setting used everywhere else)
    lab3, rt3, _ = C.run_dbscan(ds["X"], eps_km, ms)
    m3 = C.evaluate(ds["X"], lab3, None, SEED)
    rows_abl.append({"ablation": "metric", "dataset": nm, "variant": "euclidean_km",
                     "eps": eps_km, "min_samples": ms, "runtime_s": rt3, **m3})
    print(f"  {nm:12s} hav={m['n_clusters']}cl/{m['noise_frac']:.2f}n  "
          f"deg={m2['n_clusters']}cl/{m2['noise_frac']:.2f}n  "
          f"km={m3['n_clusters']}cl/{m3['noise_frac']:.2f}n")

# ===========================================================================
# 7. ABLATION C - feature scaling (Ecoli)
# ===========================================================================
print("\n[7] ablation C: feature scaling (Ecoli)")
ec_raw = D.load_ecoli(scale=False)
for tag, X in [("raw", ec_raw["X_raw"]), ("standardised", ec["X"])]:
    y = ec["y"]
    k = best_k["Ecoli"]["selected"]
    lab, rt, _ = C.run_kmeans(X, k, "k-means++", SEED)
    m = C.evaluate(X, lab, y, SEED)
    rows_abl.append({"ablation": "scaling", "dataset": "Ecoli", "variant": tag,
                     "method": "KMeans++", **m})
    e_best, m2 = best_dbscan_ari(X, y, ms=5)
    rows_abl.append({"ablation": "scaling", "dataset": "Ecoli", "variant": tag,
                     "method": "DBSCAN", "eps": e_best, **m2})
    print(f"  {tag:14s} KMeans++ ARI={m['ari']:.3f}   DBSCAN ARI={m2['ari']:.3f}")

# ===========================================================================
# 8. ABLATION D - dimensionality (Ecoli, PCA)
# ===========================================================================
print("\n[8] ablation D: dimensionality (Ecoli)")
for nd in [2, 3, 4, 5, 7]:
    Xp = PCA(n_components=nd, random_state=SEED).fit_transform(ec["X"]) if nd < 7 else ec["X"]
    y = ec["y"]
    curve = C.k_distance_curve(Xp, 5); _, e = C.knee(curve)
    # width of the eps region that yields >=2 clusters and <50% noise
    grid = np.linspace(0.2 * e, 4 * e, 40)
    ok = 0
    for g in grid:
        lab, _, _ = C.run_dbscan(Xp, g, 5)
        mm = C.evaluate(Xp, lab, y, SEED)
        if mm["n_clusters"] >= 2 and mm["noise_frac"] < 0.5:
            ok += 1
    e_best, m = best_dbscan_ari(Xp, y, ms=5)
    lab2, _, _ = C.run_kmeans(Xp, best_k["Ecoli"]["selected"], "k-means++", SEED)
    m2 = C.evaluate(Xp, lab2, y, SEED)
    rows_abl.append({"ablation": "dimensionality", "dataset": "Ecoli",
                     "variant": f"{nd}D", "dims": nd, "eps_knee": e, "eps_best": e_best,
                     "viable_eps_frac": ok / len(grid),
                     "dbscan_ari": m["ari"], "dbscan_noise": m["noise_frac"],
                     "dbscan_clusters": m["n_clusters"], "kmeans_ari": m2["ari"]})
    print(f"  {nd}D  viable-eps={ok/len(grid):.2f}  DBSCAN ARI={m['ari']:.3f}"
          f"  KMeans ARI={m2['ari']:.3f}")

# ===========================================================================
np.save(f"{OUT}/labels.npy", np.array(
    [(k[0], k[1], v.tolist()) for k, v in best_labels.items()], dtype=object),
    allow_pickle=True)
pd.DataFrame(rows_main).to_csv(f"{OUT}/main.csv", index=False)
pd.DataFrame(rows_sweep).to_csv(f"{OUT}/sweep.csv", index=False)
pd.DataFrame(rows_abl).to_csv(f"{OUT}/ablation.csv", index=False)
json.dump({"best_k": best_k, "dbscan": dbscan_best, "hdbscan": hdb_best,
           "kdist": {k: {"knee": v["knee"], "min_samples": v["min_samples"]}
                     for k, v in kdist_store.items()}},
          open(f"{OUT}/chosen_params.json", "w"), indent=2)
json.dump({k: v["curve"] for k, v in kdist_store.items()},
          open(f"{OUT}/kdist_curves.json", "w"))
print("\nDONE -> results/")
