"""Every visualisation and output table in the project.

Experiments and notebooks compute; this module draws and tabulates. Sections:
  style and saving     one dpi, one noise colour, save(fig, topic, name), show_and_save for notebooks
  tables               build_results_table, score_labels
  generic plots        cluster grids, maps, k selection, k-distance, display coordinates

Figures that need per-point labels (cluster grids, maps) are drawn by the notebook that computes the
labels, with the functions below.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.decomposition import PCA

from sc4020.metrics import NOISE_LABEL, evaluate
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


def show_and_save(fig, topic: str, name: str) -> Path:
    """Show the figure in the notebook, then write results/<topic>/figures/<name>.png and close it."""
    display(fig)
    return save(fig, topic, name)


def _colours(n: int):
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20" if n <= 20 else "gist_ncar")
    return (
        cmap(np.linspace(0, 0.95, max(n, 1)))
        if n > 20
        else [cmap(i) for i in range(max(n, 1))]
    )


def scatter_labels(
    ax, xy, labels, title="", class_names=None, size=12, legend_max=12, alpha=0.75
):
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
            ax.scatter(
                xy[noise, 0],
                xy[noise, 1],
                c=NOISE_COLOR,
                marker="x",
                s=size,
                alpha=0.6,
                label="noise",
            )
        for colour, c in zip(colours, clusters):
            name = (
                class_names[c]
                if class_names is not None and c < len(class_names)
                else str(c)
            )
            members = labels == c
            ax.scatter(
                xy[members, 0],
                xy[members, 1],
                color=colour,
                s=size,
                lw=0,
                alpha=alpha,
                label=name,
            )
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


def build_results_table(
    dataset, X, results, y_true=None, silhouette_sample=5000
) -> pd.DataFrame:
    """One row per method: fit time, the methods' parameters and every metric in metrics.evaluate."""
    rows = []
    for name, r in results.items():
        row = {"dataset": dataset, "method": name, "fit_seconds": r.fit_seconds}
        row.update({f"param_{k}": v for k, v in r.params.items()})
        row.update(evaluate(X, r.labels, y_true, silhouette_sample=silhouette_sample))
        rows.append(row)
    return pd.DataFrame(rows)


def score_labels(table: pd.DataFrame) -> dict:
    """Short score per method for figure titles: ARI when labelled, else the silhouette."""
    column = "ari" if "ari" in table and table["ari"].notna().any() else "silhouette"
    short = "ARI" if column == "ari" else "S"
    return {m: f"{short} {v:.3f}" for m, v in zip(table["method"], table[column])}


# --- generic plots ----------------------------------------------------------------------------


def plot_cluster_grid(
    X, results, y_true=None, class_names=None, title="", scores=None, ncols=3
):
    """Ground truth (if any) then one panel per method, on the data or its first two PCs.

    `results` maps a method name to a ClusterResult (or an array of labels); `scores` maps a method
    name to text appended to its panel title (e.g. score_labels(table))."""
    xy, xlabel, ylabel = project_2d(X)
    panels = [("Ground truth", y_true, class_names)] if y_true is not None else []
    panels += [(name, getattr(r, "labels", r), None) for name, r in results.items()]
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False
    )
    axes = axes.flatten()
    for ax, (name, labels, names) in zip(axes, panels):
        suffix = f" ({scores[name]})" if scores and name in scores else ""
        scatter_labels(ax, xy, labels, name + suffix, class_names=names)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    for ax in axes[len(panels) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_spatial_grid(coords_deg, results, title="", reference_title="Points"):
    """All points first, then one map per method (longitude on x, latitude on y)."""
    coords_deg = np.asarray(coords_deg)
    xy = coords_deg[:, ::-1]
    fig, axes = plt.subplots(
        1, len(results) + 1, figsize=(6 * (len(results) + 1), 6), squeeze=False
    )
    axes = axes[0]
    axes[0].scatter(xy[:, 0], xy[:, 1], c="black", alpha=0.15, s=2, edgecolors="none")
    axes[0].set_title(reference_title, fontsize=11)
    for ax, (name, r) in zip(axes[1:], results.items()):
        scatter_labels(
            ax, xy, getattr(r, "labels", r), name, size=4, legend_max=7, alpha=0.6
        )
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
        ax.scatter(
            [best_k],
            [scores[list(k_range).index(best_k)]],
            color="red",
            zorder=5,
            label=f"best k={best_k}",
        )
        ax.legend()
    ax.set_xlabel("k")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. k")
    fig.tight_layout()
    return fig


def plot_k_distance(curve, knee_idx, eps_at_knee, min_samples=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve)
    ax.axvline(
        knee_idx, color="red", linestyle="--", label=f"knee (eps={eps_at_knee:.3g})"
    )
    ax.set_xlabel("Points sorted by distance")
    ax.set_ylabel(
        f"distance to neighbour {min_samples}"
        if min_samples
        else "k-th nearest neighbour distance"
    )
    ax.set_title("DBSCAN eps selection (k-distance curve)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_coords(ds):
    """(x, y, x label, y label) for display: longitude/latitude for maps, else two PCs."""
    if "coords_deg" in ds:
        return ds.coords_deg[:, 1], ds.coords_deg[:, 0], "Longitude", "Latitude"
    pca = PCA(n_components=2, random_state=SEED).fit(ds.data)
    Z = pca.transform(ds.data)
    v = pca.explained_variance_ratio_ * 100
    return Z[:, 0], Z[:, 1], f"PC1 ({v[0]:.0f}%)", f"PC2 ({v[1]:.0f}%)"
