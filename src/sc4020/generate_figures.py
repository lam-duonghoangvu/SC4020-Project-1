"""Every visualisation and output table in the project.

Experiments and notebooks compute; this module draws and tabulates. Sections:
  style and saving     one dpi, one noise colour, save(fig, topic, name)
  tables               build_results_table, pivot_by_method, cluster_summary, earthquake EDA tables,
                       to_latex_table
  generic plots        cluster grids, maps, k selection, k-distance, metric bars, screening, case figure
  hotspot figures      the 11 figures of the hotspot study, drawn from results/hotspot/

Usage:
    python -m sc4020.generate_figures                 # every topic below
    python -m sc4020.generate_figures hotspot comparison

Topics redraw what can be rebuilt from saved results: hotspot (all 11 figures), comparison (metric
bars of the team comparison and the showcase), datasets (metric bars of the per-dataset notebooks).
Figures that need per-point labels (cluster grids, maps, screening) are drawn by the experiment or
notebook that computes the labels, with the functions below.
"""

import json
import sys
from pathlib import Path

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from sc4020 import methods
from sc4020.datasets.preprocess_dataset import chord_to_km, region_from_place
from sc4020.metrics import NOISE_LABEL, evaluate, haversine_km, nearest_neighbour_km
from sc4020.paths import results_dir

# --- style and saving -------------------------------------------------------------------------

DPI = 150
NOISE_COLOR = "#bbbbbb"
SEED = 42


def save_to(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def save(fig, topic: str, name: str) -> Path:
    """Write results/<topic>/figures/<name>.png and close the figure."""
    return save_to(fig, results_dir(topic) / "figures" / f"{name}.png")


def _colours(n: int):
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20" if n <= 20 else "gist_ncar")
    return cmap(np.linspace(0, 0.95, max(n, 1))) if n > 20 else [cmap(i) for i in range(max(n, 1))]


def scatter_labels(ax, xy, labels, title="", class_names=None, size=12, legend_max=12, alpha=0.75):
    """Points coloured by label; noise (-1) as grey crosses."""
    xy = np.asarray(xy)
    if labels is None:
        ax.scatter(xy[:, 0], xy[:, 1], s=size, c="#444444", lw=0, alpha=0.55)
    else:
        labels = np.asarray(labels)
        clusters = [c for c in sorted(set(labels.tolist())) if c != NOISE_LABEL]
        colours = _colours(len(clusters))
        if (labels == NOISE_LABEL).any():
            noise = labels == NOISE_LABEL
            ax.scatter(xy[noise, 0], xy[noise, 1], c=NOISE_COLOR, marker="x", s=size, alpha=0.6, label="noise")
        for colour, c in zip(colours, clusters):
            name = class_names[c] if class_names is not None and c < len(class_names) else str(c)
            members = labels == c
            ax.scatter(xy[members, 0], xy[members, 1], color=colour, s=size, lw=0, alpha=alpha, label=name)
        if len(clusters) + 1 <= legend_max:
            ax.legend(fontsize=7, loc="best", markerscale=1.5, framealpha=0.6)
    ax.set_title(title, fontsize=10)


def project_2d(X):
    """(xy, x label, y label): the data itself if 2-D, else its first two principal components."""
    X = np.asarray(X)
    if X.shape[1] <= 2:
        return X, "x1", "x2"
    pca = PCA(n_components=2, random_state=SEED).fit(X)
    v = pca.explained_variance_ratio_ * 100
    return pca.transform(X), f"PC1 ({v[0]:.0f}%)", f"PC2 ({v[1]:.0f}%)"


# --- tables -----------------------------------------------------------------------------------


def build_results_table(dataset, X, results, y_true=None, silhouette_sample=5000) -> pd.DataFrame:
    """One row per method: fit time, the methods' parameters and every metric in metrics.evaluate."""
    rows = []
    for name, r in results.items():
        row = {"dataset": dataset, "method": name, "fit_seconds": r.fit_seconds}
        row.update({f"param_{k}": v for k, v in r.params.items()})
        row.update(evaluate(X, r.labels, y_true, silhouette_sample=silhouette_sample))
        rows.append(row)
    return pd.DataFrame(rows)


def pivot_by_method(results: pd.DataFrame, column: str) -> pd.DataFrame:
    order = [m for m in methods.FACTORIES if m in set(results["method"])]
    return results.pivot(index="dataset", columns="method", values=column).reindex(
        index=results["dataset"].unique(), columns=order
    )


def score_labels(table: pd.DataFrame) -> dict:
    """Short score per method for figure titles: ARI when labelled, else the silhouette."""
    column = "ari" if "ari" in table and table["ari"].notna().any() else "silhouette"
    short = "ARI" if column == "ari" else "S"
    return {m: f"{short} {v:.3f}" for m, v in zip(table["method"], table[column])}


def cluster_summary(events, labels, top_regions=3) -> pd.DataFrame:
    """One row per earthquake cluster (noise excluded): size, centre on the sphere, distance of
    members to the centre, median depth and the most common regions."""
    from sc4020.clusters import spherical_centre

    labels = np.asarray(labels)
    regions = region_from_place(events["place"]).to_numpy()
    rows = []
    for cluster in sorted(set(labels.tolist()) - {NOISE_LABEL}):
        mask = labels == cluster
        members = events[mask]
        lat, lon = spherical_centre(members["latitude"], members["longitude"])
        to_centre = haversine_km(members["latitude"], members["longitude"], lat, lon)
        common = pd.Series(regions[mask]).value_counts().head(top_regions)
        rows.append({
            "cluster": cluster,
            "events": int(mask.sum()),
            "% of events": 100 * mask.sum() / len(labels),
            "centre latitude": lat,
            "centre longitude": lon,
            "median km to centre": float(np.median(to_centre)),
            "95th percentile km to centre": float(np.percentile(to_centre, 95)),
            "median depth km": float(members["depth"].median()),
            "top regions": ", ".join(f"{region} ({count})" for region, count in common.items()),
        })
    return pd.DataFrame(rows)


def magnitude_threshold_table(events, thresholds=(4.5, 5.0, 5.5, 6.0)) -> pd.DataFrame:
    """What each minimum magnitude keeps: events, share of events, median nearest-neighbour distance."""
    rows = []
    for threshold in thresholds:
        kept = events[events["mag"] >= threshold]
        spacing = nearest_neighbour_km(kept["latitude"], kept["longitude"]) if len(kept) > 1 else [np.nan]
        rows.append({
            "min magnitude": threshold,
            "events": len(kept),
            "% kept": 100 * len(kept) / len(events),
            "median nearest neighbour km": float(np.median(spacing)),
        })
    return pd.DataFrame(rows)


def grid_cell_counts(lat, lon, period, cell_degrees=5) -> pd.DataFrame:
    """Event counts per (latitude cell, longitude cell), one column per period."""
    cells = pd.DataFrame({
        "cell_lat": np.floor(np.asarray(lat, dtype=float) / cell_degrees) * cell_degrees,
        "cell_lon": np.floor(np.asarray(lon, dtype=float) / cell_degrees) * cell_degrees,
        "period": np.asarray(period),
    })
    return cells.groupby(["cell_lat", "cell_lon", "period"]).size().unstack("period", fill_value=0)


def to_latex_table(df: pd.DataFrame, path, caption: str, label: str, float_fmt="%.3f") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_latex(index=False, float_format=float_fmt, escape=True, caption=caption, label=label, position="t"))
    return path


# --- generic plots ----------------------------------------------------------------------------


def plot_cluster_grid(X, results, y_true=None, class_names=None, title="", scores=None, ncols=3):
    """Ground truth (if any) then one panel per method, on the data or its first two PCs.

    `results` maps a method name to a ClusterResult (or an array of labels); `scores` maps a method
    name to text appended to its panel title (e.g. score_labels(table))."""
    xy, xlabel, ylabel = project_2d(X)
    panels = [("Ground truth", y_true, class_names)] if y_true is not None else []
    panels += [(name, getattr(r, "labels", r), None) for name, r in results.items()]
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    axes = axes.flatten()
    for ax, (name, labels, names) in zip(axes, panels):
        suffix = f" ({scores[name]})" if scores and name in scores else ""
        scatter_labels(ax, xy, labels, name + suffix, class_names=names)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_spatial_grid(coords_deg, results, title="", reference_title="Points"):
    """All points first, then one map per method (longitude on x, latitude on y)."""
    coords_deg = np.asarray(coords_deg)
    xy = coords_deg[:, ::-1]
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(6 * (len(results) + 1), 6), squeeze=False)
    axes = axes[0]
    axes[0].scatter(xy[:, 0], xy[:, 1], c="black", alpha=0.15, s=2, edgecolors="none")
    axes[0].set_title(reference_title, fontsize=11)
    for ax, (name, r) in zip(axes[1:], results.items()):
        scatter_labels(ax, xy, getattr(r, "labels", r), name, size=4, legend_max=7, alpha=0.6)
    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_k_selection(k_range, scores, best_k=None, ylabel="Silhouette score"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(k_range), scores, marker="o")
    if best_k is not None:
        ax.scatter([best_k], [scores[list(k_range).index(best_k)]], color="red", zorder=5, label=f"best k={best_k}")
        ax.legend()
    ax.set_xlabel("k")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. k")
    fig.tight_layout()
    return fig


def plot_k_distance(curve, knee_idx, eps_at_knee, min_samples=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve)
    ax.axvline(knee_idx, color="red", linestyle="--", label=f"knee (eps={eps_at_knee:.3g})")
    ax.set_xlabel("Points sorted by distance")
    ax.set_ylabel(f"distance to neighbour {min_samples}" if min_samples else "k-th nearest neighbour distance")
    ax.set_title("DBSCAN eps selection (k-distance curve)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_metrics_bar(df: pd.DataFrame, metric: str, log_scale=False):
    pivot = df.pivot_table(index="dataset", columns="method", values=metric)
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by dataset and method")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.figure.tight_layout()
    return ax.figure


def plot_screening(panels, xlabel, ylabel, title, silhouette: pd.DataFrame | None = None):
    """Optional silhouette-vs-null curve, then one scatter per (xy, labels, title) panel with the
    cluster count and noise share in its title."""
    n = len(panels) + (silhouette is not None)
    fig, axes = plt.subplots(1, n, figsize=(7 * n if silhouette is None else 5.3 * n, 4.5), squeeze=False)
    axes = list(axes[0])
    if silhouette is not None:
        ax = axes.pop(0)
        ax.plot(silhouette["k"], silhouette["real_silhouette"], "o-", label="real")
        ax.plot(silhouette["k"], silhouette["null_silhouette"], "s--", label="shuffled null (mean of 3)")
        ax.set(xlabel="K", ylabel="Silhouette", title="K-Means silhouette", ylim=(0, None))
        ax.legend()
    for ax, (xy, labels, name) in zip(axes, panels):
        noise = labels == NOISE_LABEL
        ax.scatter(xy[noise, 0], xy[noise, 1], s=2, c="lightgrey")
        ax.scatter(xy[~noise, 0], xy[~noise, 1], s=2, c=labels[~noise], cmap="tab20")
        n_clusters = len(np.unique(labels[~noise]))
        ax.set(xlabel=xlabel, ylabel=ylabel, title=f"{name}: {n_clusters} clusters, {noise.mean():.0%} noise")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_case_figure(X, y, labels_by_method, rows, picks=("K-Means", "GMM", "DBSCAN")):
    """Two-class success / failure figure on 2 PCs: ground truth, then each picked method with its
    clusters coloured by the class most of their points belong to."""
    xy = PCA(n_components=2).fit_transform(X)
    scores = {r["method"]: r for r in rows}
    colours = np.array(["#3b6fb6", "#d1602f"])
    fig, axes = plt.subplots(1, len(picks) + 1, figsize=(3.5 * (len(picks) + 1), 3.6), sharex=True, sharey=True)
    axes[0].scatter(*xy.T, c=colours[y], s=9, linewidths=0)
    axes[0].set_title("Diagnosis (blue: benign, orange: malignant)", fontsize=9)
    for ax, m in zip(axes[1:], picks):
        lab = labels_by_method[m]
        noise = lab == NOISE_LABEL
        col = np.empty(len(lab), dtype=object)
        for c in np.unique(lab[~noise]):
            col[lab == c] = colours[int(round(y[lab == c].mean()))]
        ax.scatter(*xy[noise].T, c=NOISE_COLOR, s=9, linewidths=0)
        ax.scatter(*xy[~noise].T, c=list(col[~noise]), s=9, linewidths=0)
        r = scores[m]
        ax.set_title(f"{m}\nARI {r['ari']:.3f}, silhouette {r['silhouette']:.3f}, noise {100 * r['noise_fraction']:.1f}%", fontsize=9)
    for ax in axes:
        ax.set_xlabel("PC1")
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("PC2")
    fig.tight_layout()
    return fig


# --- hotspot figures --------------------------------------------------------------------------

HOTSPOT = "hotspot"
HOTSPOT_NAMES = {"earthquakes_1965_2016": "Earthquakes", "uber_pickups": "UberNYC", "ecoli": "Ecoli"}
HOTSPOT_METHODS = ["K-Means (random init)", "K-Means", "DBSCAN", "HDBSCAN"]
_HOTSPOT_STYLE = {"font.size": 8, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False}


def plot_coords(ds):
    """(x, y, x label, y label) for display: longitude/latitude for maps, else two PCs."""
    if "coords_deg" in ds:
        return ds.coords_deg[:, 1], ds.coords_deg[:, 0], "Longitude", "Latitude"
    pca = PCA(n_components=2, random_state=SEED).fit(ds.data)
    Z = pca.transform(ds.data)
    v = pca.explained_variance_ratio_ * 100
    return Z[:, 0], Z[:, 1], f"PC1 ({v[0]:.0f}%)", f"PC2 ({v[1]:.0f}%)"


def _hotspot_scatter(ax, ds, labels, title):
    x, y, xl, yl = plot_coords(ds)
    size = 0.6 if len(ds.data) > 5000 else 3
    if labels is None:
        ax.scatter(x, y, s=size, c="#444", lw=0, alpha=0.55)
    else:
        noise = labels == NOISE_LABEL
        ax.scatter(x[noise], y[noise], s=size * 0.8, c=NOISE_COLOR, lw=0, alpha=0.6)
        lab = labels[~noise]
        if len(lab):
            u = np.unique(lab)
            ax.scatter(x[~noise], y[~noise], s=size, c=np.asarray(_colours(len(u)))[np.searchsorted(u, lab)], lw=0)
    ax.set_title(title, fontsize=7.5)
    ax.set_xlabel(xl, fontsize=6.5)
    ax.set_ylabel(yl, fontsize=6.5)
    ax.tick_params(labelsize=6)


def _eps_text(ds, eps):
    """eps as shown in titles: km for coordinates (unit vectors are converted), else plain."""
    if "coords_deg" not in ds:
        return f"{eps:.3g}"
    km = chord_to_km(eps) if ds.data.shape[1] == 3 else eps
    return f"{float(km):.3g} km"


def _fit(method, params, X):
    if method.startswith("K-Means"):
        init = "random" if method == "K-Means (random init)" else "k-means++"
        return methods.kmeans(params["n_clusters"], init=init, random_state=SEED).fit_predict(X)
    if method == "DBSCAN":
        return methods.dbscan(params["eps"], params["min_samples"]).fit_predict(X)
    return methods.hdbscan(params["min_cluster_size"], params.get("min_samples")).fit_predict(X)


def load_hotspot_results():
    out = results_dir(HOTSPOT)
    return {
        "main": pd.read_csv(out / "main.csv"),
        "sweep": pd.read_csv(out / "sweep.csv"),
        "ablation": pd.read_csv(out / "ablation.csv"),
        "seeds": pd.read_csv(out / "init_seeds.csv"),
        "chosen": json.loads((out / "chosen_params.json").read_text()),
        "curves": json.loads((out / "kdist_curves.json").read_text()),
    }


def fig_datasets(ds_by_name, res):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.3))
    for ax, (name, ds) in zip(axes, ds_by_name.items()):
        _hotspot_scatter(ax, ds, None, f"{HOTSPOT_NAMES[name]}  (n={len(ds.data)}, d={ds.data.shape[1]})")
    return fig


def fig_qualitative(ds_by_name, res):
    main = res["main"]
    fig, axes = plt.subplots(3, 5, figsize=(9.6, 5.6))
    for r, (name, ds) in enumerate(ds_by_name.items()):
        for c, method in enumerate(HOTSPOT_METHODS):
            row = main[(main.dataset == name) & (main.method == method)].iloc[0]
            p = json.loads(row["params"])
            labels = _fit(method, p, ds.data)
            if method.startswith("K-Means"):
                title = f"{method}  k={p['n_clusters']}"
            elif method == "DBSCAN":
                title = f"DBSCAN  eps={_eps_text(ds, p['eps'])}, mPts={p['min_samples']}"
            else:
                title = f"HDBSCAN  mcs={p['min_cluster_size']}"
            title += f"\n{row.n_clusters} cl, {100 * row.noise_fraction:.0f}% noise, S={row.silhouette:.2f}"
            if "ari" in row and np.isfinite(row.ari):
                title += f", ARI={row.ari:.2f}"
            _hotspot_scatter(axes[r, c], ds, labels, title)
        _hotspot_scatter(axes[r, 4], ds, ds.target, "Ground truth" if ds.target is not None else "No ground truth available")
        axes[r, 0].set_ylabel(f"{HOTSPOT_NAMES[name]}\n" + axes[r, 0].get_ylabel(), fontsize=7)
    return fig


def fig_k_sweep(ds_by_name, res):
    sweep, chosen = res["sweep"], res["chosen"]["k"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.3))
    for ax, name in zip(axes, ds_by_name):
        s = sweep[(sweep.dataset == name) & (sweep.method == "K-Means")].copy()
        s["k"] = s["params"].map(lambda p: json.loads(p)["n_clusters"])
        s = s.sort_values("k")
        ax.plot(s.k, s.inertia / s.inertia.max(), "o-", ms=2.5, lw=1, label="inertia (norm.)", color="#333")
        ax.plot(s.k, s.silhouette, "s-", ms=2.5, lw=1, label="silhouette", color="#c0392b")
        if "ari" in s and s.ari.notna().any():
            ax.plot(s.k, s.ari, "^-", ms=2.5, lw=1, label="ARI", color="#2980b9")
        ax.axvline(chosen[name]["elbow"], ls="--", lw=0.8, c="#333", alpha=0.6)
        ax.axvline(chosen[name]["selected"], ls=":", lw=1.1, c="#c0392b")
        ax.set_title(f"{HOTSPOT_NAMES[name]}: elbow k={chosen[name]['elbow']}, selected k={chosen[name]['selected']}", fontsize=7.5)
        ax.set_xlabel("k")
        ax.legend(fontsize=5.5, loc="best")
    return fig


def fig_k_distance(ds_by_name, res):
    from sc4020.tuning import knee

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.2))
    for ax, (name, ds) in zip(axes, ds_by_name.items()):
        curve = np.array(res["curves"][name])
        i, e = knee(curve)
        ax.plot(curve, lw=1.1, color="#333")
        ax.axhline(e, ls="--", lw=0.9, c="#c0392b")
        ax.plot([i], [e], "o", ms=4, c="#c0392b")
        ax.set_title(f"{HOTSPOT_NAMES[name]}: knee eps={_eps_text(ds, e)}\n(min_samples={res['chosen']['kdist'][name]['min_samples']})", fontsize=7.5)
        ax.set_xlabel("points sorted")
        ax.set_ylabel("k-th NN distance")
    return fig


def fig_dbscan_grid(ds_by_name, res):
    sweep = res["sweep"]
    fig, axes = plt.subplots(2, 3, figsize=(7.8, 4.0))
    for c, (name, ds) in enumerate(ds_by_name.items()):
        s = sweep[(sweep.dataset == name) & (sweep.method == "DBSCAN")].copy()
        s["eps"] = s["params"].map(lambda p: round(json.loads(p)["eps"], 4))
        s["min_samples"] = s["params"].map(lambda p: json.loads(p)["min_samples"])
        for r, value in enumerate(["n_clusters", "noise_fraction"]):
            pivot = s.pivot_table(index="min_samples", columns="eps", values=value)
            ax = axes[r, c]
            norm = matplotlib.colors.LogNorm() if value == "n_clusters" and np.nanmin(pivot.values) > 0 else None
            im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", norm=norm)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{v:.3g}" for v in pivot.columns], rotation=90, fontsize=4.5)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=6)
            ax.set_title(f"{HOTSPOT_NAMES[name]}: {value}", fontsize=7)
            ax.set_xlabel("eps", fontsize=6)
            ax.set_ylabel("min_samples", fontsize=6)
            ax.grid(False)
            plt.colorbar(im, ax=ax, fraction=0.04)
    return fig


def fig_init(ds_by_name, res):
    a = res["ablation"][res["ablation"].ablation == "init"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.2))
    for ax, name in zip(axes, ds_by_name):
        s = a[a.dataset == name]
        xs = np.arange(len(s))
        norm = s.inertia_mean.max()
        ax.bar(xs - 0.17, s.inertia_mean / norm, 0.33, yerr=s.inertia_std / norm, capsize=3, label="inertia (norm.)", color="#34495e")
        ax2 = ax.twinx()
        ax2.bar(xs + 0.17, s.iter_mean, 0.33, yerr=s.iter_std, capsize=3, label="iterations", color="#e67e22")
        ax2.grid(False)
        ax.set_xticks(xs)
        ax.set_xticklabels(s.method, fontsize=6)
        ax.set_title(f"{HOTSPOT_NAMES[name]} (k={int(s.k.iloc[0])}, 30 seeds)", fontsize=7.5)
        ax.set_ylabel("inertia (norm.)", fontsize=6.5)
        ax2.set_ylabel("iterations", fontsize=6.5)
    return fig


def fig_init_d3d4(ds_by_name, res):
    """One measure per panel, no second axis; colours are colour-blind-separable categorical slots."""
    seeds = res["seeds"]
    colours = {"K-Means (random init)": "#2a78d6", "K-Means": "#eb6834"}
    names = {"K-Means (random init)": "random", "K-Means": "K-Means++"}
    panels = [("uber_pickups", "inertia", "D3 Uber: gap to best inertia (%)"),
              ("ecoli", "inertia", "D4 Ecoli: gap to best inertia (%)"),
              ("ecoli", "ari", "D4 Ecoli: ARI against the sites")]
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.3))
    for ax, (name, column, title) in zip(axes, panels):
        s = seeds[seeds.dataset == name]
        values = 100 * (s.inertia / s.inertia.min() - 1) if column == "inertia" else s.ari
        for x, method in enumerate(colours):
            v = values[s.method == method].to_numpy()
            ax.scatter(x + rng.uniform(-0.15, 0.15, len(v)), v, s=10, color=colours[method], alpha=0.75, linewidths=0)
            ax.hlines(v.mean(), x - 0.28, x + 0.28, color="#222222", lw=1.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([names[m] for m in colours])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(f"{title}\nK={int(s.k.iloc[0])}, 30 seeds, 1 start", fontsize=7)
        ax.grid(axis="x", visible=False)
    return fig


def fig_ablations(ds_by_name, res):
    abl = res["ablation"]
    d, sc = abl[abl.ablation == "dimensionality"], abl[abl.ablation == "scaling"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.2))
    axes[0].plot(d.dims, d.kmeans_ari, "o-", ms=3, label="K-Means++", color="#2980b9")
    axes[0].plot(d.dims, d.dbscan_ari, "s-", ms=3, label="DBSCAN", color="#c0392b")
    axes[0].set(xlabel="PCA dimensions", ylabel="ARI")
    axes[0].set_title("Ecoli: ARI vs dimensionality", fontsize=7.5)
    axes[0].legend(fontsize=6)
    axes[1].plot(d.dims, d.viable_eps_frac, "o-", ms=3, color="#c0392b")
    axes[1].set(xlabel="PCA dimensions", ylabel="fraction of eps grid viable")
    axes[1].set_title("Ecoli: width of usable eps region", fontsize=7.5)
    sc.pivot_table(index="variant", columns="method", values="ari").plot(
        kind="bar", ax=axes[2], rot=0, width=0.7, color=["#c0392b", "#2980b9"], legend=True
    )
    axes[2].set(ylabel="ARI", xlabel="")
    axes[2].set_title("Ecoli: effect of feature scaling", fontsize=7.5)
    axes[2].legend(fontsize=6)
    return fig


def fig_metric_disagreement(ds_by_name, res):
    e = res["main"][res["main"].dataset == "ecoli"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
    x, w = np.arange(len(e)), 0.38
    axes[0].bar(x - w / 2, e.silhouette, w, label="Silhouette (internal)", color="#7f8c8d")
    axes[0].bar(x + w / 2, e.ari, w, label="ARI (external)", color="#c0392b")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(e.method, rotation=20, fontsize=6.5)
    axes[0].legend(fontsize=6)
    axes[0].set_title("Ecoli: internal and external indices disagree", fontsize=7.5)
    r_int = e.silhouette.rank(ascending=False).to_numpy()
    r_ext = e.ari.rank(ascending=False).to_numpy()
    axes[1].scatter(r_int, r_ext, s=40, c="#c0392b", zorder=3)
    for xi, yi, m in zip(r_int, r_ext, e.method):
        axes[1].annotate(m, (xi, yi), fontsize=6, xytext=(3, 3), textcoords="offset points")
    axes[1].plot([1, len(e)], [1, len(e)], ls="--", lw=0.8, c="#333")
    axes[1].set(xlabel="rank by Silhouette (1 = best)", ylabel="rank by ARI (1 = best)")
    axes[1].set_title(f"Rank correlation = {np.corrcoef(r_int, r_ext)[0, 1]:.2f}", fontsize=7.5)
    return fig


def fig_runtime(ds_by_name, res):
    fig, ax = plt.subplots(figsize=(4.0, 2.2))
    pivot = res["main"].pivot_table(index="dataset", columns="method", values="fit_seconds")
    pivot = pivot[HOTSPOT_METHODS].loc[list(ds_by_name)].rename(index=HOTSPOT_NAMES)
    pivot.plot(kind="bar", ax=ax, rot=0, logy=True, width=0.78, color=["#95a5a6", "#34495e", "#c0392b", "#e67e22"])
    ax.set(ylabel="fit time (s, log scale)", xlabel="")
    ax.legend(fontsize=6)
    ax.set_title("Runtime at the selected operating point", fontsize=8)
    return fig


def fig_operating_point(ds_by_name, res):
    """Heuristic operating points (elbow, k-distance knee) against the shared rule's DBSCAN."""
    chosen, main = res["chosen"], res["main"]
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 4.2))
    for r, name in enumerate(["earthquakes_1965_2016", "uber_pickups"]):
        ds = ds_by_name[name]
        k_elbow = chosen["k"][name]["elbow"]
        labels = _fit("K-Means", {"n_clusters": k_elbow}, ds.data)
        m = evaluate(ds.data, labels)
        _hotspot_scatter(axes[r, 0], ds, labels, f"K-Means++ at elbow k={k_elbow}\n{m['n_clusters']} cl, S={m['silhouette']:.2f}")
        kd = chosen["kdist"][name]
        labels = _fit("DBSCAN", {"eps": kd["knee"], "min_samples": kd["min_samples"]}, ds.data)
        m = evaluate(ds.data, labels)
        _hotspot_scatter(axes[r, 1], ds, labels, f"DBSCAN at k-distance knee\neps={_eps_text(ds, kd['knee'])}, {m['n_clusters']} cl, "
                         f"{100 * m['noise_fraction']:.0f}% noise, S={m['silhouette']:.2f}")
        p = json.loads(main[(main.dataset == name) & (main.method == "DBSCAN")].iloc[0]["params"])
        labels = _fit("DBSCAN", p, ds.data)
        m = evaluate(ds.data, labels)
        _hotspot_scatter(axes[r, 2], ds, labels, f"DBSCAN by the shared rule\neps={_eps_text(ds, p['eps'])}, {m['n_clusters']} cl, "
                         f"{100 * m['noise_fraction']:.0f}% noise, S={m['silhouette']:.2f}")
        axes[r, 0].set_ylabel(f"{HOTSPOT_NAMES[name]}\n" + axes[r, 0].get_ylabel(), fontsize=7)
    return fig


HOTSPOT_FIGURES = {
    "f1_datasets": fig_datasets,
    "f2_qualitative": fig_qualitative,
    "f3_ksweep": fig_k_sweep,
    "f4_kdistance": fig_k_distance,
    "f5_dbscan_grid": fig_dbscan_grid,
    "f6_init": fig_init,
    "f6_init_d3d4": fig_init_d3d4,
    "f7_ablations": fig_ablations,
    "f8_metric_disagreement": fig_metric_disagreement,
    "f9_runtime": fig_runtime,
    "f10_operating_point": fig_operating_point,
}


def hotspot_figures():
    from sc4020.experiments.hotspot import load_all

    res = load_hotspot_results()
    ds_by_name = {ds.name: ds for ds in load_all()}
    with plt.rc_context(_HOTSPOT_STYLE):
        for name, draw in HOTSPOT_FIGURES.items():
            fig = draw(ds_by_name, res)
            fig.tight_layout()
            print(save(fig, HOTSPOT, name))
    e = res["main"][res["main"].dataset == "ecoli"]
    r = np.corrcoef(e.silhouette.rank(ascending=False), e.ari.rank(ascending=False))[0, 1]
    print(f"Ecoli rank correlation silhouette vs ARI: {r:.3f}")


def comparison_figures():
    out = results_dir("comparison")
    for csv, prefix in [("team_results.csv", "team"), ("showcase_results.csv", "showcase")]:
        if (out / csv).exists():
            df = pd.read_csv(out / csv)
            for metric in ["ari", "silhouette", "noise_fraction"]:
                if metric in df and df[metric].notna().any():
                    print(save(plot_metrics_bar(df.dropna(subset=[metric]), metric), "comparison", f"{prefix}_{metric}"))
            print(save(plot_metrics_bar(df, "fit_seconds", log_scale=True), "comparison", f"{prefix}_fit_seconds"))


def dataset_figures():
    out = results_dir("datasets")
    frames = [pd.read_csv(p) for p in sorted(out.glob("*_results.csv"))]
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    for metric, log in [("ari", False), ("silhouette", False), ("davies_bouldin", False), ("fit_seconds", True)]:
        if metric in df and df[metric].notna().any():
            print(save(plot_metrics_bar(df.dropna(subset=[metric]), metric, log_scale=log), "datasets", f"summary_{metric}"))


TOPICS = {"hotspot": hotspot_figures, "comparison": comparison_figures, "datasets": dataset_figures}


def main(argv=None):
    topics = (argv if argv is not None else sys.argv[1:]) or list(TOPICS)
    unknown = set(topics) - set(TOPICS)
    if unknown:
        sys.exit(f"Unknown topics {sorted(unknown)}; choose from {list(TOPICS)}")
    for topic in topics:
        print(f"== {topic}", flush=True)
        TOPICS[topic]()


if __name__ == "__main__":
    main()
