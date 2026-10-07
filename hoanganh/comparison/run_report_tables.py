"""Shared-column results for the group report: D6 Wine and D7 Breast cancer.

Adds the columns the report's shared table needs and that the earlier runs did
not save: largest-cluster share, NMI and the fake-data ("null") silhouette, at
the settings chosen in run_team_comparison.py (plus Agglomerative and Spectral
from the notebooks). Also draws the D7 success / failure figure.

Data: sklearn.datasets copies of the same UCI tables the notebooks download
from Kaggle (same rows, same order), so no download is needed.

    python comparison/run_report_tables.py      # from the hoanganh folder

Outputs: comparison/results/report_d6_d7.csv, ../report/figures/bc_cases.png
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
SEED = 42          # model seed, as in the notebooks
NULL_SEED = 0      # seed of the column shuffle
MIN_SAMPLES = 10   # DBSCAN setting chosen in team_results.csv for both datasets


def k_distance_knee(X, min_samples):
    """Same rule as sc4020.tuning.k_distance_knee."""
    d, _ = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)
    curve = np.sort(d[:, -1])
    x = np.linspace(0.0, 1.0, len(curve))
    y = (curve - curve[0]) / max(curve[-1] - curve[0], 1e-12)
    return float(curve[int(np.argmax(x - y))])


def models(k, gmm_cov, eps):
    return {
        f"K-Means (k={k})": KMeans(n_clusters=k, n_init=10, random_state=SEED),
        f"Agglomerative (Ward, k={k})": AgglomerativeClustering(n_clusters=k, linkage="ward"),
        f"Spectral (k={k})": SpectralClustering(n_clusters=k, affinity="nearest_neighbors", random_state=SEED),
        f"GMM ({k}, {gmm_cov})": GaussianMixture(n_components=k, covariance_type=gmm_cov, n_init=3, random_state=SEED),
        f"DBSCAN ({eps:.2f}, {MIN_SAMPLES})": DBSCAN(eps=eps, min_samples=MIN_SAMPLES),
        "HDBSCAN (5)": HDBSCAN(min_cluster_size=5),
    }


def shuffled(X, seed=NULL_SEED):
    """Each column permuted on its own: same marginals, no joint structure."""
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])


def score(X, labels):
    mask = labels != -1
    ids, counts = np.unique(labels[mask], return_counts=True)
    out = {"clusters": len(ids), "noise_fraction": float((~mask).mean()),
           "largest_fraction": float(counts.max() / len(X)) if len(ids) else 0.0,
           "silhouette": np.nan, "davies_bouldin": np.nan}
    if len(ids) >= 2:
        out["silhouette"] = float(silhouette_score(X[mask], labels[mask]))
        out["davies_bouldin"] = float(davies_bouldin_score(X[mask], labels[mask]))
    return out


def run(name, X, y, k, gmm_cov, eps_factor):
    # eps chosen in team_results.csv: a multiple of the k-distance knee
    eps = eps_factor * k_distance_knee(X, MIN_SAMPLES)
    Xnull = shuffled(X)
    rows, labels_by_method = [], {}
    for method, model in models(k, gmm_cov, eps).items():
        t = time.perf_counter()
        labels = model.fit_predict(X)
        secs = time.perf_counter() - t
        null = score(Xnull, models(k, gmm_cov, eps)[method].fit_predict(Xnull))
        rows.append({"dataset": name, "method": method, **score(X, labels),
                     "ari": adjusted_rand_score(y, labels),
                     "nmi": normalized_mutual_info_score(y, labels),
                     "fit_seconds": secs,
                     "null_silhouette": null["silhouette"],
                     "null_clusters": null["clusters"],
                     "null_noise_fraction": null["noise_fraction"]})
        labels_by_method[method] = labels
    return rows, labels_by_method


def case_figure(X, y, labels_by_method, rows, path):
    """D7: ground truth, K-Means, GMM (success) and DBSCAN (failure) on 2 PCs."""
    xy = PCA(n_components=2).fit_transform(X)
    ari = {r["method"]: r for r in rows}
    pick = [m for m in labels_by_method if m.startswith(("K-Means", "GMM", "DBSCAN"))]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharex=True, sharey=True)
    colours = np.array(["#3b6fb6", "#d1602f"])
    axes[0].scatter(*xy.T, c=colours[y], s=9, linewidths=0)
    axes[0].set_title("Diagnosis (blue: benign, orange: malignant)", fontsize=9)
    for ax, m in zip(axes[1:], pick):
        lab = labels_by_method[m]
        noise = lab == -1
        # colour each cluster by the class most of its points belong to
        col = np.empty(len(lab), dtype=object)
        for c in np.unique(lab[~noise]):
            col[lab == c] = colours[int(round(y[lab == c].mean()))]
        ax.scatter(*xy[noise].T, c="#c8c8c8", s=9, linewidths=0)
        ax.scatter(*xy[~noise].T, c=list(col[~noise]), s=9, linewidths=0)
        r = ari[m]
        ax.set_title(f"{m}\nARI {r['ari']:.3f}, silhouette {r['silhouette']:.3f}, "
                     f"noise {100 * r['noise_fraction']:.1f}%", fontsize=9)
    for ax in axes:
        ax.set_xlabel("PC1"); ax.tick_params(labelsize=7)
    axes[0].set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    wine, bc = load_wine(), load_breast_cancer()
    Xw = StandardScaler().fit_transform(wine.data)
    Xb = StandardScaler().fit_transform(bc.data)
    yb = 1 - bc.target  # 0 benign, 1 malignant, as in the notebooks
    rows_w, _ = run("D6 wine", Xw, wine.target, 3, "diag", 0.75)
    rows_b, lab_b = run("D7 breast_cancer", Xb, yb, 2, "full", 0.5)
    out = HERE / "results" / "report_d6_d7.csv"
    pd.DataFrame(rows_w + rows_b).to_csv(out, index=False)
    fig_path = HERE.parent.parent / "report" / "figures" / "bc_cases.png"
    case_figure(Xb, yb, lab_b, rows_b, fig_path)
    print(pd.DataFrame(rows_w + rows_b).round(3).to_string())
