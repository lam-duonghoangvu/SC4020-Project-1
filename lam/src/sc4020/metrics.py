"""Internal (unsupervised) and external (supervised) cluster validity metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from sc4020.clustering import ClusterResult

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    mask = labels != -1
    n_clusters_found = len(set(labels[mask])) if mask.any() else 0
    noise_ratio = float(np.mean(labels == -1))

    metrics: dict[str, float] = {
        "n_clusters_found": n_clusters_found,
        "noise_ratio": noise_ratio,
    }

    if n_clusters_found < 2 or mask.sum() < 2:
        metrics.update(
            {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
        )
        return metrics

    X_eval, labels_eval = X[mask], labels[mask]
    try:
        metrics["silhouette"] = float(silhouette_score(X_eval, labels_eval))
    except ValueError:
        metrics["silhouette"] = np.nan
    try:
        metrics["davies_bouldin"] = float(davies_bouldin_score(X_eval, labels_eval))
    except ValueError:
        metrics["davies_bouldin"] = np.nan
    try:
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_eval, labels_eval))
    except ValueError:
        metrics["calinski_harabasz"] = np.nan

    return metrics


def external_metrics(y_true: np.ndarray | None, labels: np.ndarray) -> dict:
    if y_true is None:
        return {}
    return {
        "adjusted_rand_index": float(adjusted_rand_score(y_true, labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(y_true, labels)),
        "adjusted_mutual_info": float(adjusted_mutual_info_score(y_true, labels)),
        "fowlkes_mallows": float(fowlkes_mallows_score(y_true, labels)),
    }


def build_results_table(
    dataset_name: str,
    X: np.ndarray,
    results: dict[str, ClusterResult],
    y_true: np.ndarray | None,
    save: bool = True,
) -> pd.DataFrame:
    rows = []
    for method_name, result in results.items():
        row = {"dataset": dataset_name, "method": method_name, "fit_seconds": result.fit_seconds}
        row.update({f"param_{k}": v for k, v in result.params.items()})
        row.update(internal_metrics(X, result.labels))
        row.update(external_metrics(y_true, result.labels))
        rows.append(row)

    df = pd.DataFrame(rows)
    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        df.to_csv(RESULTS_DIR / f"{dataset_name}_results.csv", index=False)
    return df
