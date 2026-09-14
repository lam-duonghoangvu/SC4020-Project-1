"""All figures for the report."""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import data as D
import core as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG, RES = os.path.join(ROOT, "figs"), os.path.join(ROOT, "results")
SEED = 42
plt.rcParams.update({"font.size": 8, "figure.dpi": 200,
                     "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})

eq, ub, ec = D.load_earthquakes(), D.load_uber(), D.load_ecoli()
DS = {"Earthquakes": eq, "UberNYC": ub, "Ecoli": ec}
P = json.load(open(f"{RES}/chosen_params.json"))
sweep = pd.read_csv(f"{RES}/sweep.csv")
abl = pd.read_csv(f"{RES}/ablation.csv")
main = pd.read_csv(f"{RES}/main.csv")


def plot_coords(ds):
    """2-D coordinates used for display."""
    if ds["spatial"]:
        return ds["X_deg"][:, 1], ds["X_deg"][:, 0], "Longitude", "Latitude"
    p = PCA(n_components=2, random_state=SEED).fit(ds["X"])
    Z = p.transform(ds["X"])
    v = p.explained_variance_ratio_ * 100
    return Z[:, 0], Z[:, 1], f"PC1 ({v[0]:.0f}%)", f"PC2 ({v[1]:.0f}%)"


def scatter(ax, ds, labels, title, s=1.2):
    x, y, xl, yl = plot_coords(ds)
    if labels is None:
        ax.scatter(x, y, s=s, c="#444", lw=0, alpha=.55)
    else:
        noise = labels == -1
        ax.scatter(x[noise], y[noise], s=s * .8, c="#cccccc", lw=0, alpha=.6)
        lab = labels[~noise]
        if len(lab):
            u = np.unique(lab)
            cmap = plt.get_cmap("tab20" if len(u) <= 20 else "gist_ncar")
            cols = cmap(np.linspace(0, .95, len(u)))[np.searchsorted(u, lab)]
            ax.scatter(x[~noise], y[~noise], s=s, c=cols, lw=0)
    ax.set_title(title, fontsize=7.5)
    ax.set_xlabel(xl, fontsize=6.5); ax.set_ylabel(yl, fontsize=6.5)
    ax.tick_params(labelsize=6)


# --------------------------------------------------------------- F1 datasets
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.3))
for ax, (nm, ds) in zip(axes, DS.items()):
    scatter(ax, ds, None, f"{nm}  (n={ds['n']}, d={ds['d']})",
            s=.6 if ds["n"] > 5000 else 3)
fig.tight_layout(); fig.savefig(f"{FIG}/f1_datasets.png"); plt.close(fig)

# ------------------------------------------------------- F2 qualitative grid
METHODS = ["KMeans", "KMeans++", "DBSCAN", "HDBSCAN"]
fig, axes = plt.subplots(3, 5, figsize=(9.6, 5.6))
for r, (nm, ds) in enumerate(DS.items()):
    for c, meth in enumerate(METHODS):
        row = main[(main.dataset == nm) & (main.method == meth)].iloc[0]
        pr = json.loads(row["params"])
        if meth in ("KMeans", "KMeans++"):
            init = "random" if meth == "KMeans" else "k-means++"
            lab, _, _ = C.run_kmeans(ds["X"], pr["k"], init, SEED)
            ttl = f"{meth}  k={pr['k']}"
        elif meth == "DBSCAN":
            lab, _, _ = C.run_dbscan(ds["X"], pr["eps"], pr["min_samples"])
            ttl = f"DBSCAN  eps={pr['eps']:.3g}, mPts={pr['min_samples']}"
        else:
            lab, _, _ = C.run_hdbscan(ds["X"], pr["min_cluster_size"])
            ttl = f"HDBSCAN  mcs={pr['min_cluster_size']}"
        extra = f"\n{row.n_clusters} cl, {100*row.noise_frac:.0f}% noise, S={row.silhouette:.2f}"
        if np.isfinite(row.ari):
            extra += f", ARI={row.ari:.2f}"
        scatter(axes[r, c], ds, lab, ttl + extra, s=.6 if ds["n"] > 5000 else 3)
    y = ds["y"]
    scatter(axes[r, 4], ds, y if y is not None else None,
            "Ground truth" if y is not None else "No ground truth available",
            s=.6 if ds["n"] > 5000 else 3)
    axes[r, 0].set_ylabel(f"{nm}\n" + axes[r, 0].get_ylabel(), fontsize=7)
fig.tight_layout(); fig.savefig(f"{FIG}/f2_qualitative.png"); plt.close(fig)

# ------------------------------------------------------------- F3 k sweep
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.3))
for ax, (nm, ds) in zip(axes, DS.items()):
    s = sweep[(sweep.dataset == nm) & (sweep.method == "KMeans++")].copy()
    s["k"] = s["params"].map(lambda p: json.loads(p)["k"])
    s = s.sort_values("k")
    ax.plot(s.k, s.inertia / s.inertia.max(), "o-", ms=2.5, lw=1,
            label="inertia (norm.)", color="#333")
    ax.plot(s.k, s.silhouette, "s-", ms=2.5, lw=1, label="silhouette",
            color="#c0392b")
    if s.ari.notna().any():
        ax.plot(s.k, s.ari, "^-", ms=2.5, lw=1, label="ARI", color="#2980b9")
    ax.axvline(P["best_k"][nm]["elbow"], ls="--", lw=.8, c="#333", alpha=.6)
    ax.axvline(P["best_k"][nm]["selected"], ls=":", lw=1.1, c="#c0392b")
    ax.set_title(f"{nm}: elbow k={P['best_k'][nm]['elbow']}, "
                 f"selected k={P['best_k'][nm]['selected']}", fontsize=7.5)
    ax.set_xlabel("k"); ax.legend(fontsize=5.5, loc="best")
fig.tight_layout(); fig.savefig(f"{FIG}/f3_ksweep.png"); plt.close(fig)

# --------------------------------------------------------- F4 k-distance
curves = json.load(open(f"{RES}/kdist_curves.json"))
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.2))
for ax, (nm, ds) in zip(axes, DS.items()):
    c = np.array(curves[nm]); i, e = C.knee(c)
    ax.plot(c, lw=1.1, color="#333")
    ax.axhline(e, ls="--", lw=.9, c="#c0392b")
    ax.plot([i], [e], "o", ms=4, c="#c0392b")
    unit = "km" if ds["spatial"] else "(std. units)"
    ax.set_title(f"{nm}: knee eps={e:.3g} {unit}\n(min_samples={P['kdist'][nm]['min_samples']})",
                 fontsize=7.5)
    ax.set_xlabel("points sorted"); ax.set_ylabel("k-th NN distance")
fig.tight_layout(); fig.savefig(f"{FIG}/f4_kdistance.png"); plt.close(fig)

# ------------------------------------------------------ F5 DBSCAN heatmaps
fig, axes = plt.subplots(2, 3, figsize=(7.8, 4.0))
for c_, (nm, ds) in enumerate(DS.items()):
    s = sweep[(sweep.dataset == nm) & (sweep.method == "DBSCAN")].copy()
    s["eps"] = s["params"].map(lambda p: round(json.loads(p)["eps"], 4))
    s["ms"] = s["params"].map(lambda p: json.loads(p)["min_samples"])
    for r_, val in enumerate(["n_clusters", "noise_frac"]):
        piv = s.pivot_table(index="ms", columns="eps", values=val)
        ax = axes[r_, c_]
        im = ax.imshow(piv.values, aspect="auto", cmap="viridis",
                       norm=(matplotlib.colors.LogNorm() if val == "n_clusters"
                             else None))
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{v:.3g}" for v in piv.columns], rotation=90, fontsize=4.5)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=6)
        ax.set_title(f"{nm}: {val}", fontsize=7)
        ax.set_xlabel("eps", fontsize=6); ax.set_ylabel("min_samples", fontsize=6)
        ax.grid(False)
        plt.colorbar(im, ax=ax, fraction=.04)
fig.tight_layout(); fig.savefig(f"{FIG}/f5_dbscan_grid.png"); plt.close(fig)

# ------------------------------------------------------- F6 init ablation
a = abl[abl.ablation == "init"]
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.2))
for ax, nm in zip(axes, DS):
    s = a[a.dataset == nm]
    xs = np.arange(2)
    norm = s.inertia_mean.max()
    ax.bar(xs - .17, s.inertia_mean / norm, .33,
           yerr=s.inertia_std / norm, capsize=3, label="inertia (norm.)",
           color="#34495e")
    ax2 = ax.twinx()
    ax2.bar(xs + .17, s.iter_mean, .33, yerr=s.iter_std, capsize=3,
            label="iterations", color="#e67e22")
    ax2.grid(False)
    ax.set_xticks(xs); ax.set_xticklabels(s.method, fontsize=7)
    ax.set_title(f"{nm} (k={int(s.k.iloc[0])}, 30 seeds)", fontsize=7.5)
    ax.set_ylabel("inertia (norm.)", fontsize=6.5)
    ax2.set_ylabel("iterations", fontsize=6.5)
fig.tight_layout(); fig.savefig(f"{FIG}/f6_init.png"); plt.close(fig)

# -------------------------------------------------- F7 dimensionality + scaling
d = abl[abl.ablation == "dimensionality"]
sc = abl[abl.ablation == "scaling"]
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.2))
axes[0].plot(d.dims, d.kmeans_ari, "o-", ms=3, label="K-Means++", color="#2980b9")
axes[0].plot(d.dims, d.dbscan_ari, "s-", ms=3, label="DBSCAN", color="#c0392b")
axes[0].set_xlabel("PCA dimensions"); axes[0].set_ylabel("ARI")
axes[0].set_title("Ecoli: ARI vs dimensionality", fontsize=7.5); axes[0].legend(fontsize=6)
axes[1].plot(d.dims, d.viable_eps_frac, "o-", ms=3, color="#c0392b")
axes[1].set_xlabel("PCA dimensions"); axes[1].set_ylabel("fraction of eps grid viable")
axes[1].set_title("Ecoli: width of usable eps region", fontsize=7.5)
w = sc.pivot_table(index="variant", columns="method", values="ari")
w.plot(kind="bar", ax=axes[2], rot=0, width=.7,
       color=["#c0392b", "#2980b9"], legend=True)
axes[2].set_ylabel("ARI"); axes[2].set_xlabel("")
axes[2].set_title("Ecoli: effect of feature scaling", fontsize=7.5)
axes[2].legend(fontsize=6)
fig.tight_layout(); fig.savefig(f"{FIG}/f7_ablations.png"); plt.close(fig)

# ------------------------------------------ F8 internal vs external agreement
e = main[main.dataset == "Ecoli"]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
x = np.arange(len(e)); w = .38
axes[0].bar(x - w/2, e.silhouette, w, label="Silhouette (internal)", color="#7f8c8d")
axes[0].bar(x + w/2, e.ari, w, label="ARI (external)", color="#c0392b")
axes[0].set_xticks(x); axes[0].set_xticklabels(e.method, rotation=20, fontsize=6.5)
axes[0].legend(fontsize=6); axes[0].set_title("Ecoli: internal and external indices disagree", fontsize=7.5)
r_int = e.silhouette.rank(ascending=False).to_numpy()
r_ext = e.ari.rank(ascending=False).to_numpy()
axes[1].scatter(r_int, r_ext, s=40, c="#c0392b", zorder=3)
for xi, yi, m in zip(r_int, r_ext, e.method):
    axes[1].annotate(m, (xi, yi), fontsize=6, xytext=(3, 3), textcoords="offset points")
axes[1].plot([1, 4], [1, 4], ls="--", lw=.8, c="#333")
axes[1].set_xlabel("rank by Silhouette (1 = best)")
axes[1].set_ylabel("rank by ARI (1 = best)")
axes[1].set_title(f"Rank correlation = {np.corrcoef(r_int, r_ext)[0,1]:.2f}", fontsize=7.5)
fig.tight_layout(); fig.savefig(f"{FIG}/f8_metric_disagreement.png"); plt.close(fig)

# ---------------------------------------------------------- F9 runtime
fig, ax = plt.subplots(figsize=(4.0, 2.2))
piv = main.pivot_table(index="dataset", columns="method", values="runtime_s")
piv = piv[METHODS].loc[list(DS)]
piv.plot(kind="bar", ax=ax, rot=0, logy=True, width=.78,
         color=["#95a5a6", "#34495e", "#c0392b", "#e67e22"])
ax.set_ylabel("runtime (s, log scale)"); ax.set_xlabel("")
ax.legend(fontsize=6); ax.set_title("Runtime at the selected operating point", fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/f9_runtime.png"); plt.close(fig)

print("figures written:", sorted(os.listdir(FIG)))
print(f"\nEcoli rank correlation silhouette vs ARI: {np.corrcoef(r_int, r_ext)[0,1]:.3f}")

# ------------------------------- F10 operating point: heuristic vs silhouette
fig, axes = plt.subplots(2, 3, figsize=(8.0, 4.2))
for r_, nm in enumerate(["Earthquakes", "UberNYC"]):
    ds = DS[nm]
    ke = P["best_k"][nm]["elbow"]
    lab, _, _ = C.run_kmeans(ds["X"], ke, "k-means++", SEED)
    m = C.evaluate(ds["X"], lab, None, SEED)
    scatter(axes[r_, 0], ds, lab,
            f"K-Means++ at elbow k={ke}\n{m['n_clusters']} cl, S={m['silhouette']:.2f}",
            s=.6 if ds["n"] > 5000 else 3)
    ms = P["kdist"][nm]["min_samples"]; e = P["kdist"][nm]["knee"]
    lab, _, _ = C.run_dbscan(ds["X"], e, ms)
    m = C.evaluate(ds["X"], lab, None, SEED)
    scatter(axes[r_, 1], ds, lab,
            f"DBSCAN at k-distance knee\neps={e:.3g} km, {m['n_clusters']} cl, "
            f"{100*m['noise_frac']:.0f}% noise, S={m['silhouette']:.2f}",
            s=.6 if ds["n"] > 5000 else 3)
    pr = json.loads(main[(main.dataset == nm) & (main.method == "DBSCAN")].iloc[0]["params"])
    lab, _, _ = C.run_dbscan(ds["X"], pr["eps"], pr["min_samples"])
    m = C.evaluate(ds["X"], lab, None, SEED)
    scatter(axes[r_, 2], ds, lab,
            f"DBSCAN at silhouette optimum\neps={pr['eps']:.3g} km, {m['n_clusters']} cl, "
            f"{100*m['noise_frac']:.0f}% noise, S={m['silhouette']:.2f}",
            s=.6 if ds["n"] > 5000 else 3)
    axes[r_, 0].set_ylabel(f"{nm}\n" + axes[r_, 0].get_ylabel(), fontsize=7)
fig.tight_layout(); fig.savefig(f"{FIG}/f10_operating_point.png"); plt.close(fig)
print("f10 done")
