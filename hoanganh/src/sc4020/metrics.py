"""Internal (unsupervised) and external (ground-truth) cluster validity metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score

from sc4020.clustering import ClusterResult

NOISE_LABEL = -1


def evaluate(
    X: np.ndarray,
    labels: np.ndarray,
    y_true: np.ndarray | None = None,
    silhouette_sample_size: int | None = None,
    random_state: int = 42,
) -> dict:
    """Metrics for one labelling. Noise points (-1) are excluded from the
    internal metrics; ARI is computed on all points, noise counted as its own group.
    """
    mask = labels != NOISE_LABEL
    n_clusters = len(set(labels)) - (1 if NOISE_LABEL in labels else 0)
    scorable = n_clusters > 1 and mask.sum() > n_clusters

    row = {
        "n_clusters_found": n_clusters,
        "noise_points": int((~mask).sum()),
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
    }
    if scorable:
        row["silhouette"] = silhouette_score(
            X[mask], labels[mask], sample_size=silhouette_sample_size, random_state=random_state
        )
        row["davies_bouldin"] = davies_bouldin_score(X[mask], labels[mask])
    if y_true is not None:
        row["adjusted_rand_index"] = (
            adjusted_rand_score(y_true, labels) if n_clusters > 0 else np.nan
        )
    return row


def build_results_table(
    dataset: str,
    X: np.ndarray,
    results: dict[str, ClusterResult],
    y_true: np.ndarray | None = None,
    silhouette_sample_size: int | None = None,
) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        row = {"dataset": dataset, "method": name, "fit_seconds": r.fit_seconds}
        row.update(evaluate(X, r.labels, y_true, silhouette_sample_size))
        rows.append(row)
    return pd.DataFrame(rows)
