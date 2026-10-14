"""Per-dataset clustering runs, tuned without labels: the helpers behind every notebook in experiments/.

A notebook goes one method at a time: `select_method` picks the settings, `fit_method` fits them, and
`score_methods`, `tried_settings` and `save_dataset_run` turn the fits into results/datasets/ tables.

K-Means, DBSCAN, HDBSCAN and GMM are tuned by the shared rule (tuning.select_all_methods); K-Means and
GMM get the true k on labelled data, and the label-free choice is recorded. K-Means (random init) and
Spectral use the K-Means k; Spectral with a dense RBF affinity is skipped above
MAX_POINTS_FOR_DENSE_SPECTRAL points. Labels are used only for ARI and the other external scores.
`max_clusters` caps the cluster count of the density methods. Every fit is repeated on fake data
(every column shuffled) for the null silhouette.
"""

import dataclasses
import math
import time

import numpy as np
import pandas as pd

from sc4020 import generate_figures as gf
from sc4020 import methods, tuning
from sc4020.metrics import dbcv_score, evaluate
from sc4020.paths import results_dir

TOPIC = "datasets"
NULL_SEED = 0  # seed of the column shuffle behind the fake-data silhouette
SEED = methods.SEED

# Silhouette and DBCV grow with n^2, so both are estimated on a sample for large data.
SILHOUETTE_SAMPLE = 5000
DBCV_SAMPLE = 3000

# These methods take the k that K-Means chose instead of tuning their own.
METHODS_USING_KMEANS_K = ("K-Means (random init)", "Spectral")

# A dense RBF affinity matrix needs memory quadratic in the number of points; a nearest-neighbour
# graph is sparse, so Spectral with it runs at any size.
MAX_POINTS_FOR_DENSE_SPECTRAL = 10_000

# Global earthquake catalogues have dozens of natural belts, so every method may use up to 50
# clusters there (a coarse grid keeps GMM's BIC search affordable).
QUAKE_K_RANGE = (2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50)
K_RANGES = {"earthquakes_2023": QUAKE_K_RANGE}

# An RBF kernel for roughly globular data and a nearest-neighbour graph for manifold-shaped data
# (report, Section 2 "Spectral clustering").
SPECTRAL_AFFINITY_BY_DATASET = {"wine": "rbf"}
DEFAULT_SPECTRAL_AFFINITY = "nearest_neighbors"

# Columns score_methods adds to gf.build_results_table, in the order they appear.
RUN_COLUMNS = (
    "n_points",
    "n_features",
    "labelled",
    "params",
    "dbcv",
    "tuning_valid",
    "label_free_choice",
    "best_ari_choice",
    "tuning_seconds",
    "null_silhouette",
    "null_clusters",
    "null_noise_fraction",
)


def _sample_size(n_points: int, cap: int) -> int | None:
    return cap if n_points > cap else None


def silhouette_sample_size(n_points: int) -> int | None:
    """Points to estimate the silhouette on: SILHOUETTE_SAMPLE for large data, None (all) otherwise."""
    return _sample_size(n_points, SILHOUETTE_SAMPLE)


def format_params(params: dict | None) -> str:
    """Parameters as short text, e.g. "eps=0.59, min_samples=6"."""
    if not params:
        return ""
    return ", ".join(
        f"{name}={value:.3g}" if isinstance(value, float) else f"{name}={value}"
        for name, value in params.items()
    )


def _spectral_affinity(ds) -> str:
    return SPECTRAL_AFFINITY_BY_DATASET.get(ds.name, DEFAULT_SPECTRAL_AFFINITY)


def is_too_large(ds, method_name: str) -> bool:
    """Whether `method_name` needs more memory than `ds` allows (see MAX_POINTS_FOR_DENSE_SPECTRAL)."""
    return (
        method_name == "Spectral"
        and _spectral_affinity(ds) != "nearest_neighbors"
        and len(ds.data) > MAX_POINTS_FOR_DENSE_SPECTRAL
    )


def _spectral_params(ds, k: int) -> dict:
    return {"n_clusters": k, "affinity": _spectral_affinity(ds)}


def select_method(
    ds, method_name: str, selections=None, k_range=None, max_clusters=None
) -> tuning.Selection:
    """The Selection for one method: the shared rule for the four tuned methods, fixed settings for Spectral.

    Methods in METHODS_USING_KMEANS_K take the k of `selections["K-Means"]`, tuning K-Means first when
    it is missing."""
    k_range = k_range or K_RANGES.get(ds.name, tuning.K_RANGE)
    start = time.perf_counter()
    if method_name in methods.FOUR_METHODS:
        selection = tuning.select_all_methods(
            ds.data,
            k=ds.n_true_clusters,
            k_range=k_range,
            sample_size=silhouette_sample_size(len(ds.data)),
            max_clusters=max_clusters,
            y_true=ds.target,
            method_names=[method_name],
        )[method_name]
    elif method_name == "K-Means (random init)":
        kmeans_selection = (selections or {}).get("K-Means") or select_method(
            ds, "K-Means", k_range=k_range
        )
        selection = dataclasses.replace(kmeans_selection, tried=[])
    else:
        kmeans_selection = (selections or {}).get("K-Means") or select_method(
            ds, "K-Means", k_range=k_range
        )
        selection = tuning.Selection(
            _spectral_params(ds, kmeans_selection.params["n_clusters"]), np.nan
        )
    selection.tuning_seconds = time.perf_counter() - start
    return selection


def fit_method(
    ds, method_name: str, selection: tuning.Selection
) -> methods.ClusterResult:
    """The fitted, timed result of `method_name` at the settings in `selection`."""
    params = {method_name: selection.params}
    return methods.run_models(methods.build_models(params), ds.data, params=params)[
        method_name
    ]


def score_methods(ds, selections, results) -> pd.DataFrame:
    """One row per fitted method: gf.build_results_table plus RUN_COLUMNS."""
    X, n_points = ds.data, len(ds.data)
    table = gf.build_results_table(
        ds.name,
        X,
        results,
        ds.target,
        silhouette_sample=silhouette_sample_size(n_points),
    )
    names = table["method"]
    table.insert(1, "n_points", n_points)
    table.insert(2, "n_features", X.shape[1])
    table.insert(3, "labelled", ds.target is not None)
    table.insert(
        5, "params", [format_params(selections[name].params) for name in names]
    )
    dbcv_sample = _sample_size(n_points, DBCV_SAMPLE)
    table["dbcv"] = [dbcv_score(X, results[name].labels, dbcv_sample) for name in names]
    table["tuning_valid"] = [selections[name].valid for name in names]
    table["label_free_choice"] = [
        format_params(selections[name].label_free) for name in names
    ]
    table["best_ari_choice"] = [
        format_params(selections[name].best_ari) for name in names
    ]
    table["tuning_seconds"] = [selections[name].tuning_seconds for name in names]
    null_rows = [null_scores(ds, name, selections[name]) for name in names]
    table["null_silhouette"] = [row["silhouette"] for row in null_rows]
    table["null_clusters"] = [row["n_clusters"] for row in null_rows]
    table["null_noise_fraction"] = [row["noise_fraction"] for row in null_rows]
    return table


def null_scores(ds, method_name: str, selection: tuning.Selection) -> dict:
    """metrics.evaluate of `method_name` refitted at `selection` on a copy of the data with every
    column shuffled (the fake-data check)."""
    X_null = tuning.shuffle_columns_independently(
        ds.data, np.random.default_rng(NULL_SEED)
    )
    params = {method_name: selection.params}
    labels = methods.run_models(methods.build_models(params), X_null)[
        method_name
    ].labels
    return evaluate(
        X_null, labels, silhouette_sample=silhouette_sample_size(len(X_null))
    )


def tried_settings(ds, selections) -> pd.DataFrame:
    """Every setting the selection tried, one row per (method, setting)."""
    return pd.DataFrame(
        [
            {"dataset": ds.name, "method": name, **setting}
            for name, selection in selections.items()
            for setting in selection.tried
        ]
    )


def plot_dataset(ds, results, table, title, ncols=None):
    """Clusters of every method: maps for coordinate data, else the data or its first two PCs."""
    if "coords_deg" in ds:
        return gf.plot_spatial_grid(ds.coords_deg, results, title=title)
    if ncols is None:
        n_panels = len(results) + (ds.target is not None)
        ncols = math.ceil(n_panels / math.ceil(n_panels / 5))
    return gf.plot_cluster_grid(
        ds.data,
        results,
        ds.target,
        class_names=ds.get("target_names"),
        title=title,
        scores=gf.score_labels(table),
        ncols=ncols,
    )


def _replace_rows(path, new_rows: pd.DataFrame) -> pd.DataFrame:
    """`new_rows` plus the rows of the CSV at `path` whose method was not run this time."""
    if not path.exists() or new_rows.empty:
        return new_rows
    old_rows = pd.read_csv(path)
    if "method" not in old_rows:
        return new_rows
    kept = old_rows[~old_rows["method"].isin(new_rows["method"])]
    if kept.empty:
        return new_rows
    combined = pd.concat([kept, new_rows], ignore_index=True)
    order = {name: i for i, name in enumerate(methods.FACTORIES)}
    return combined.sort_values(
        "method", key=lambda column: column.map(order), kind="stable"
    )


def save_dataset_run(dataset_name, table, tried) -> None:
    """Write results/datasets/<dataset>_results.csv and _tuning.csv, keeping other methods' rows."""
    out = results_dir(TOPIC)
    results_path = out / f"{dataset_name}_results.csv"
    tuning_path = out / f"{dataset_name}_tuning.csv"
    _replace_rows(results_path, table).to_csv(results_path, index=False)
    _replace_rows(tuning_path, tried).to_csv(tuning_path, index=False)
