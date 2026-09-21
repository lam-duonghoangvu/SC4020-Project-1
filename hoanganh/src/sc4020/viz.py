"""Plotting helpers for the clustering comparison notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sc4020.clustering import ClusterResult

NOISE_COLOR = [0.6, 0.6, 0.6, 0.5]


def _scatter_labels(
    ax: Axes,
    xy: np.ndarray,
    labels: np.ndarray,
    cmap,
    max_legend: int,
    noise_color=NOISE_COLOR,
    **scatter_kw,
) -> None:
    unique = sorted(set(labels))
    colors = cmap(np.linspace(0, 1, max(len(unique), 1)))
    for k in unique:
        mask = labels == k
        color, text = (noise_color, "Noise") if k == -1 else (colors[k % len(colors)], f"C{k}")
        ax.scatter(
            xy[mask, 0], xy[mask, 1], color=color,
            label=text if len(unique) <= max_legend else None, **scatter_kw,
        )


def plot_pca_grid(
    X_2d: np.ndarray,
    results: dict[str, ClusterResult],
    y_true: np.ndarray,
    var_ratio: np.ndarray,
    ari: dict[str, float | str],
    gt_title: str,
    title: str,
    gt_cmap: str = "Set1",
    cmap=plt.cm.tab10,
    edgecolors: str = "k",
    size: int = 40,
    ncols: int = 4,
) -> Figure:
    """Ground truth in the first panel, then one panel per method, on a PCA 2D projection."""
    n_panels = len(results) + 1
    nrows = -(-n_panels // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
    axes = axes.flatten()

    axes[0].scatter(
        X_2d[:, 0], X_2d[:, 1], c=y_true, cmap=gt_cmap, edgecolors=edgecolors,
        s=size, alpha=0.8,
    )
    axes[0].set_title(gt_title, fontsize=11, fontweight="bold", color="darkred")

    for ax, (name, r) in zip(axes[1:], results.items()):
        _scatter_labels(
            ax, X_2d, r.labels, cmap, max_legend=4,
            edgecolors=edgecolors, s=size, alpha=0.75,
        )
        ax.set_title(f"{name} (ARI: {ari[name]})", fontsize=11, fontweight="bold")
        if len(set(r.labels)) <= 4:
            ax.legend(loc="upper right", fontsize=8)

    for ax in axes[:n_panels]:
        ax.set_xlabel(f"PC1 ({var_ratio[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({var_ratio[1]:.1f}%)")
        ax.grid(True, linestyle="--", alpha=0.3)
    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=15, y=0.99)
    fig.tight_layout()
    return fig


def plot_spatial_grid(
    coords_deg: np.ndarray,
    results: dict[str, ClusterResult],
    title: str,
    reference_title: str = "★ Actual Pickup Density",
    cmap=plt.cm.tab20,
) -> Figure:
    """Raw point density in the first panel, then one panel per method (lon on x, lat on y)."""
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(6 * (len(results) + 1), 6))

    axes[0].scatter(
        coords_deg[:, 1], coords_deg[:, 0], c="black", alpha=0.15, s=2, edgecolors="none"
    )
    axes[0].set_title(reference_title, fontsize=11, fontweight="bold", color="darkred")

    xy = coords_deg[:, ::-1]  # (lon, lat)
    for ax, (name, r) in zip(axes[1:], results.items()):
        _scatter_labels(
            ax, xy, r.labels, cmap, max_legend=6,
            noise_color=[0.65, 0.65, 0.65, 0.2], edgecolors="none", s=5, alpha=0.6,
        )
        ax.set_title(name, fontsize=11, fontweight="bold")
        if len(set(r.labels)) <= 6:
            ax.legend(loc="upper left", markerscale=3, fontsize=8)

    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return fig
