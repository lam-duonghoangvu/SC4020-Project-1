"""Label-free parameter selection, one rule for every dataset (report, Section 3.2 "Selection rule").

  K-Means  k by the fake-data check (the number of classes when it is known; the label-free choice
           is still recorded): among the k whose seeds agree (minimum pairwise ARI >= MIN_SEED_ARI),
           the largest gap between the real silhouette and the mean silhouette on shuffled columns.
           The silhouette alone can pick a k that shuffled data score just as well (2023
           earthquakes: k=2, 0.544 vs 0.529).
  GMM      k and covariance type with the lowest BIC (k fixed to the number of classes when known).
  DBSCAN   min_samples from the dimension, eps at multiples of the k-distance knee.
  HDBSCAN  min_cluster_size up to n/10, min_samples in {default, 5, 10}.
  Density settings are kept only if they give at least 2 clusters and label at most `max_noise`
  (50%) of points as noise; among those, the highest silhouette wins. The noise cap matters
  because the silhouette ignores noise: without it, it prefers settings that discard most of the
  data. `max_clusters` and `max_largest_share` optionally cap the cluster count and the largest
  cluster's share (silhouette can favour one giant cluster plus a tiny distant one). When no
  setting passes, the least-noise setting with 2+ clusters is kept and marked `valid=False`.

Fake data keeps each feature's values but has no joint structure: `shuffle_columns_independently`
for tables, `uniform_box_null` for coordinates. `null_gap` scores any chosen setting against it.

Labels never choose a setting. When `y_true` is given, every tried setting also records its ARI so
the best-ARI setting can be reported next to the rule's choice (`Selection.best_ari`).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

from sc4020 import methods
from sc4020.metrics import (
    NOISE_LABEL,
    cluster_shares,
    evaluate,
    pairwise_ari,
    silhouette,
)

K_RANGE = range(2, 11)
COVARIANCE_TYPES = ("full", "tied", "diag", "spherical")
EPS_FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
MIN_CLUSTER_SIZES = (5, 10, 15, 20, 30, 50, 100, 200, 400)
HDBSCAN_MIN_SAMPLES = (None, 5, 10)  # None: same as min_cluster_size
MAX_NOISE = 0.5
N_NULLS = 3
N_STABILITY_SEEDS = 10
MIN_SEED_ARI = 0.95


@dataclass
class Selection:
    """The chosen parameters plus every setting that was tried."""

    params: dict
    score: float
    tried: list[dict] = field(default_factory=list)
    valid: bool = True
    label_free: dict | None = None
    best_ari: dict | None = None
    tuning_seconds: float = np.nan


# --- fake data ----------------------------------------------------------------------------------


def shuffle_columns_independently(X, rng) -> np.ndarray:
    """A fake copy of X with every column shuffled on its own: each feature keeps its values, but
    the combinations across features, where clusters live, are destroyed."""
    X = np.asarray(X)
    return np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])


def uniform_box_null(X, rng) -> np.ndarray:
    """A fake copy of X with points uniform in its bounding box (for coordinates)."""
    X = np.asarray(X, dtype=float)
    return rng.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)


NULLS = {"shuffled": shuffle_columns_independently, "uniform_box": uniform_box_null}


def null_gap(
    X, fit, null="shuffled", n_nulls=N_NULLS, random_state=methods.SEED, y_true=None
) -> dict:
    """Scores of `fit` (X -> labels) on the real data and the mean over `n_nulls` fake copies.

    `gap` is the real silhouette minus the mean fake silhouette; NaN when no fake copy formed 2+
    clusters."""
    rng = np.random.default_rng(random_state)
    fakes = [NULLS[null](X, rng) for _ in range(n_nulls)]
    real = evaluate(X, fit(X), y_true)
    fake_scores = [evaluate(fake, fit(fake)) for fake in fakes]
    fake_silhouettes = [scores["silhouette"] for scores in fake_scores]
    null_silhouette = (
        float(np.nanmean(fake_silhouettes))
        if np.isfinite(fake_silhouettes).any()
        else np.nan
    )
    return {
        "null": null,
        "real_silhouette": real["silhouette"],
        "real_clusters": real["n_clusters"],
        "real_noise": real["noise_fraction"],
        "null_silhouette": null_silhouette,
        "null_clusters_mean": float(np.mean([s["n_clusters"] for s in fake_scores])),
        "null_noise_mean": float(np.mean([s["noise_fraction"] for s in fake_scores])),
        "gap": real["silhouette"] - null_silhouette,
    }


# --- k-distance -------------------------------------------------------------------------------


def k_distance_curve(X, min_samples, metric="euclidean") -> np.ndarray:
    """Sorted distance from each point to its min_samples-th neighbour, counting the point itself.

    A point is a DBSCAN core point when this distance is at most eps, so the curve suggests eps.
    For metric="haversine", X is [latitude, longitude] in radians and distances are in radians."""
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2")
    algorithm = "ball_tree" if metric == "haversine" else "auto"
    distances, _ = (
        NearestNeighbors(n_neighbors=min_samples, metric=metric, algorithm=algorithm)
        .fit(X)
        .kneighbors(X)
    )
    return np.sort(distances[:, -1])


def find_knee(curve) -> tuple[int, float]:
    """Index and value at the knee of a monotone curve: the point furthest from the straight line
    joining its first and last points, with both axes scaled to [0, 1]. Works for the increasing
    k-distance curve and the decreasing inertia curve alike."""
    y = np.asarray(curve, dtype=float)
    x = np.linspace(0.0, 1.0, len(y))
    scaled = (y - y[0]) / (y[-1] - y[0] + 1e-12)
    idx = int(np.argmax(np.abs(scaled - x)))
    return idx, float(y[idx])


def k_at_inertia_elbow(inertias, ks) -> int:
    """The knee rule applied to the K-Means inertia curve."""
    return ks[find_knee(inertias)[0]]


def min_samples_from_dimension(n_features: int) -> int:
    return int(np.clip(2 * n_features, 4, 20))


# --- one tried setting --------------------------------------------------------------------------


def passes_shared_rule(
    n_clusters,
    noise_fraction,
    silhouette,
    largest_cluster_share=0.0,
    max_noise=MAX_NOISE,
    max_clusters=None,
    max_largest_share=None,
) -> bool:
    """The shared rule's filter: at least 2 clusters, at most `max_noise` noise, a finite silhouette,
    and (optionally) at most `max_clusters` clusters and a largest-cluster share of `max_largest_share`."""
    return bool(
        n_clusters >= 2
        and noise_fraction <= max_noise
        and (max_clusters is None or n_clusters <= max_clusters)
        and (max_largest_share is None or largest_cluster_share <= max_largest_share)
        and np.isfinite(silhouette)
    )


def best_setting_by_shared_rule(
    grid, max_noise=MAX_NOISE, max_clusters=None, max_largest_share=None
):
    """The row of a grid of tried settings that the shared rule picks: the highest silhouette among
    rows passing `passes_shared_rule`; None when no row passes. Rows whose `data` column is not "real"
    (runs on null data) are ignored. Needs n_clusters, noise_fraction and silhouette columns."""
    grid = pd.DataFrame(grid)
    if "data" in grid.columns:
        grid = grid[grid["data"] == "real"]
    ok = [
        passes_shared_rule(
            r["n_clusters"],
            r["noise_fraction"],
            r["silhouette"],
            r.get("largest_cluster_share", 0.0),
            max_noise,
            max_clusters,
            max_largest_share,
        )
        for _, r in grid.iterrows()
    ]
    candidates = grid[ok]
    return (
        None if candidates.empty else candidates.loc[candidates["silhouette"].idxmax()]
    )


def _score_trial(
    X,
    labels,
    y_true,
    sample_size,
    max_clusters,
    max_noise,
    max_largest_share,
    random_state,
) -> dict:
    """Coverage, silhouette (clustered points only), ARI if labelled, and whether the rule accepts it."""
    labels = np.asarray(labels)
    row = cluster_shares(labels)
    clustered = labels != NOISE_LABEL
    row["silhouette"] = (
        silhouette(X[clustered], labels[clustered], sample_size, random_state)
        if row["n_clusters"] >= 2 and clustered.sum() > row["n_clusters"]
        else np.nan
    )
    if y_true is not None:
        row["ari"] = float(adjusted_rand_score(y_true, labels))
    row["passes"] = passes_shared_rule(
        row["n_clusters"],
        row["noise_fraction"],
        row["silhouette"],
        row["largest_cluster_share"],
        max_noise,
        max_clusters,
        max_largest_share,
    )
    return row


def _pick_by_shared_rule(tried: list[dict], keys: tuple[str, ...]) -> Selection:
    passed = [t for t in tried if t["passes"]]
    if passed:
        # the rule's filter was applied in _score_trial; the highest silhouette wins (as in best_setting_by_shared_rule)
        best, valid = max(passed, key=lambda t: t["silhouette"]), True
    else:
        fallback = [t for t in tried if t["n_clusters"] >= 2] or tried
        best, valid = min(fallback, key=lambda t: t["noise_fraction"]), False
    best_ari = None
    if tried and "ari" in tried[0]:
        top = max(tried, key=lambda t: t["ari"])
        best_ari = {k: top[k] for k in keys}
    return Selection(
        {k: best[k] for k in keys}, best["silhouette"], tried, valid, best_ari=best_ari
    )


# --- per method -------------------------------------------------------------------------------


def stable_k_with_largest_null_gap(
    ks, null_gaps, min_seed_aris, min_seed_ari=MIN_SEED_ARI
):
    """The k with the largest gap over fake data among seed-stable k, and whether that choice is valid.

    The choice is invalid when no k is stable or the best gap is not positive; the largest gap over
    every k is then returned, so the caller still gets a k but knows the data did not beat the null."""
    ks, null_gaps = np.asarray(ks), np.asarray(null_gaps, dtype=float)
    is_stable = np.asarray(min_seed_aris, dtype=float) >= min_seed_ari
    candidates = np.flatnonzero(is_stable) if is_stable.any() else np.arange(len(ks))
    best = candidates[np.nanargmax(null_gaps[candidates])]
    return int(ks[best]), bool(is_stable.any() and null_gaps[best] > 0)


def select_kmeans_k_by_null_gap(
    X,
    k=None,
    k_range: Sequence[int] = K_RANGE,
    n_nulls=N_NULLS,
    n_seeds=N_STABILITY_SEEDS,
    min_seed_ari=MIN_SEED_ARI,
    to_features=None,
    to_silhouette_features=None,
    silhouette_metric="euclidean",
    sample_size=None,
    random_state=methods.SEED,
    y_true=None,
) -> Selection:
    """k by the fake-data check: the largest silhouette gap over shuffled columns among stable k.

    Each k is fitted on the real data, on `n_nulls` copies with every column shuffled (copy i uses
    seed `random_state + i`), and with `n_seeds` seeds (`random_state + i`); a k is stable when the
    lowest pairwise ARI between seeds is at least `min_seed_ari`. `to_features` maps the raw columns
    to the space K-Means runs in, and `to_silhouette_features` with `silhouette_metric` to the space
    the silhouette is measured in (e.g. unit vectors and radians with "haversine" for coordinates).
    Both are applied after shuffling, so the fake data lie in the same space as the real data.
    Each tried k also records Davies-Bouldin (in the K-Means space) and the cluster sizes. With `k`
    given (the number of classes), that k is used and the label-free choice goes to `label_free`."""
    X = np.asarray(X, dtype=float)
    to_features = to_features or (lambda data: data)
    to_silhouette_features = to_silhouette_features or to_features
    fakes = [
        shuffle_columns_independently(X, np.random.default_rng(random_state + i))
        for i in range(n_nulls)
    ]
    features, silhouette_features = to_features(X), to_silhouette_features(X)
    fake_spaces = [(to_features(fake), to_silhouette_features(fake)) for fake in fakes]
    tried = []
    for n in k_range:
        model = methods.kmeans(n, random_state=random_state)
        labels = model.fit_predict(features)
        real_silhouette = silhouette(
            silhouette_features, labels, sample_size, random_state, silhouette_metric
        )
        null_silhouette = float(
            np.mean(
                [
                    silhouette(
                        fake_silhouette_features,
                        methods.kmeans(n, random_state=random_state).fit_predict(
                            fake_features
                        ),
                        sample_size,
                        random_state,
                        silhouette_metric,
                    )
                    for fake_features, fake_silhouette_features in fake_spaces
                ]
            )
        )
        seed_agreements = pairwise_ari(
            [
                methods.kmeans(n, random_state=random_state + i).fit_predict(features)
                for i in range(n_seeds)
            ]
        )
        sizes = np.bincount(labels)
        row = {
            "n_clusters": n,
            "inertia": float(model.inertia_),
            "silhouette": real_silhouette,
            "null_silhouette": null_silhouette,
            "null_gap": real_silhouette - null_silhouette,
            "davies_bouldin": float(davies_bouldin_score(features, labels)),
            "smallest_cluster": int(sizes.min()),
            "largest_cluster": int(sizes.max()),
            "mean_seed_ari": float(seed_agreements.mean()),
            "min_seed_ari": float(seed_agreements.min()),
        }
        if y_true is not None:
            row["ari"] = float(adjusted_rand_score(y_true, labels))
        tried.append(row)
    chosen_k, is_valid = stable_k_with_largest_null_gap(
        [t["n_clusters"] for t in tried],
        [t["null_gap"] for t in tried],
        [t["min_seed_ari"] for t in tried],
        min_seed_ari,
    )
    used_k = chosen_k if k is None else k
    best = next(t for t in tried if t["n_clusters"] == used_k)
    best_ari = (
        {"n_clusters": max(tried, key=lambda t: t["ari"])["n_clusters"]}
        if y_true is not None
        else None
    )
    return Selection(
        {"n_clusters": used_k},
        best["silhouette"],
        tried,
        valid=is_valid,
        label_free={"n_clusters": chosen_k},
        best_ari=best_ari,
    )


def select_gmm_by_bic(
    X,
    k=None,
    k_range: Sequence[int] = K_RANGE,
    covariance_types=COVARIANCE_TYPES,
    random_state=methods.SEED,
    y_true=None,
) -> Selection:
    """Lowest BIC over (k, covariance type); with `k` fixed only the covariance type is chosen.
    The score stored is -BIC so higher is better."""
    X = np.asarray(X)
    tried = []
    for cov in covariance_types:
        for n in k_range:
            model = methods.gmm(n, covariance_type=cov, random_state=random_state).fit(
                X
            )
            row = {"n_components": n, "covariance_type": cov, "score": -model.bic(X)}
            if y_true is not None:
                row["ari"] = float(adjusted_rand_score(y_true, model.predict(X)))
            tried.append(row)
    keys = ("n_components", "covariance_type")
    label_free = max(tried, key=lambda t: t["score"])
    best = (
        label_free
        if k is None
        else max((t for t in tried if t["n_components"] == k), key=lambda t: t["score"])
    )
    best_ari = (
        {c: max(tried, key=lambda t: t["ari"])[c] for c in keys}
        if y_true is not None
        else None
    )
    return Selection(
        {c: best[c] for c in keys},
        best["score"],
        tried,
        label_free={c: label_free[c] for c in keys},
        best_ari=best_ari,
    )


def select_dbscan_by_shared_rule(
    X,
    min_samples_grid: Sequence[int] | None = None,
    eps_factors: Sequence[float] = EPS_FACTORS,
    max_noise=MAX_NOISE,
    max_clusters=None,
    max_largest_share=None,
    sample_size=None,
    random_state=methods.SEED,
    y_true=None,
) -> Selection:
    """eps at `eps_factors` times the k-distance knee for each min_samples, chosen by the shared rule."""
    X = np.asarray(X)
    if min_samples_grid is None:
        base = min_samples_from_dimension(X.shape[1])
        min_samples_grid = sorted({max(3, base // 2), base, base * 2, base * 4})
    tried = []
    for min_samples in min_samples_grid:
        eps_knee = find_knee(k_distance_curve(X, min_samples))[1]
        for factor in eps_factors:
            eps = eps_knee * factor
            labels = methods.dbscan(eps, min_samples).fit_predict(X)
            row = _score_trial(
                X,
                labels,
                y_true,
                sample_size,
                max_clusters,
                max_noise,
                max_largest_share,
                random_state,
            )
            tried.append(
                {"eps": eps, "min_samples": min_samples, "eps_factor": factor, **row}
            )
            # Clusters only merge as eps grows, so once everything is one cluster every larger
            # eps is too (and those are the memory-hungry settings).
            if row["n_clusters"] == 1 and row["noise_fraction"] < max_noise:
                break
    return _pick_by_shared_rule(tried, ("eps", "min_samples"))


def select_hdbscan_by_shared_rule(
    X,
    min_cluster_sizes: Sequence[int] | None = None,
    min_samples_grid: Sequence[int | None] = HDBSCAN_MIN_SAMPLES,
    max_noise=MAX_NOISE,
    max_clusters=None,
    max_largest_share=None,
    sample_size=None,
    random_state=methods.SEED,
    y_true=None,
) -> Selection:
    """min_cluster_size (default: MIN_CLUSTER_SIZES up to n/10) x min_samples, chosen by the shared rule."""
    X = np.asarray(X)
    if min_cluster_sizes is None:
        cap = max(5, len(X) // 10)
        min_cluster_sizes = [c for c in MIN_CLUSTER_SIZES if c <= cap]
    tried = []
    for mcs in min_cluster_sizes:
        for min_samples in min_samples_grid:
            labels = methods.hdbscan(mcs, min_samples).fit_predict(X)
            row = _score_trial(
                X,
                labels,
                y_true,
                sample_size,
                max_clusters,
                max_noise,
                max_largest_share,
                random_state,
            )
            tried.append({"min_cluster_size": mcs, "min_samples": min_samples, **row})
    return _pick_by_shared_rule(tried, ("min_cluster_size", "min_samples"))


def select_all_methods(
    X,
    k=None,
    k_range: Sequence[int] = K_RANGE,
    sample_size=None,
    max_clusters=None,
    random_state=methods.SEED,
    y_true=None,
    method_names: Sequence[str] = methods.FOUR_METHODS,
) -> dict[str, Selection]:
    """Selections keyed by method name (methods.FOUR_METHODS), ready for methods.build_models.

    `sample_size` estimates the silhouette on a sample for large data; `max_clusters` (e.g.
    max(k_range)) gives the density methods the same cluster budget as K-Means and GMM.
    `method_names` limits the selection to some of the four methods."""
    selectors = {
        "K-Means": lambda: select_kmeans_k_by_null_gap(
            X,
            k,
            k_range,
            sample_size=sample_size,
            random_state=random_state,
            y_true=y_true,
        ),
        "DBSCAN": lambda: select_dbscan_by_shared_rule(
            X,
            max_clusters=max_clusters,
            sample_size=sample_size,
            random_state=random_state,
            y_true=y_true,
        ),
        "HDBSCAN": lambda: select_hdbscan_by_shared_rule(
            X,
            max_clusters=max_clusters,
            sample_size=sample_size,
            random_state=random_state,
            y_true=y_true,
        ),
        "GMM": lambda: select_gmm_by_bic(
            X, k, k_range, random_state=random_state, y_true=y_true
        ),
    }
    unknown = set(method_names) - set(selectors)
    if unknown:
        raise ValueError(
            f"no tuning rule for {sorted(unknown)}; choose from {list(selectors)}"
        )
    return {
        name: select() for name, select in selectors.items() if name in method_names
    }
