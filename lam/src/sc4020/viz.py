"""Shared plotting helpers for the clustering notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sc4020.clustering import ClusterResult

NOISE_COLOR = "#999999"


def _scatter_labels(ax, X_2d: np.ndarray, labels: np.ndarray, title: str, class_names=None):
    labels = np.asarray(labels)
    unique = sorted(set(labels))
    cmap = plt.get_cmap("tab10" if len(unique) <= 10 else "tab20")

    for i, lab in enumerate(unique):
        mask = labels == lab
        if lab == -1:
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1], c=NOISE_COLOR, marker="x", s=15,
                alpha=0.6, label="noise",
            )
        else:
            name = class_names[lab] if class_names is not None and lab < len(class_names) else str(lab)
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1], color=cmap(i % cmap.N), s=15,
                alpha=0.75, label=name,
            )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    if len(unique) <= 12:
        ax.legend(fontsize=7, loc="best", markerscale=1.5, framealpha=0.6)


def plot_cluster_grid(
    X_2d: np.ndarray,
    results: dict[str, ClusterResult],
    y_true: np.ndarray | None = None,
    class_names=None,
    title: str = "",
    ncols: int = 3,
):
    panels = []
    if y_true is not None:
        panels.append(("Ground Truth", y_true))
    panels.extend((name, r.labels) for name, r in results.items())

    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (name, labels) in zip(axes, panels):
        cn = class_names if name == "Ground Truth" else None
        _scatter_labels(ax, X_2d, labels, name, class_names=cn)
    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_k_selection(k_range, scores, best_k=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(k_range), scores, marker="o")
    if best_k is not None:
        idx = list(k_range).index(best_k)
        ax.scatter([best_k], [scores[idx]], color="red", zorder=5, label=f"best k={best_k}")
        ax.legend()
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette score vs. k")
    fig.tight_layout()
    return fig


def plot_k_distance(distances, knee_idx, eps_at_knee):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(distances)
    ax.axvline(knee_idx, color="red", linestyle="--", label=f"knee (eps={eps_at_knee:.3f})")
    ax.set_xlabel("Points sorted by distance")
    ax.set_ylabel("k-th nearest neighbor distance")
    ax.set_title("DBSCAN eps selection (k-distance graph)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_metrics_bar(df: pd.DataFrame, metric_col: str, log_scale: bool = False):
    pivot = df.pivot(index="dataset", columns="method", values=metric_col)
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel(metric_col)
    ax.set_title(f"{metric_col} by dataset and method")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    return ax.figure
