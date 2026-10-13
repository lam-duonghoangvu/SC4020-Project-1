"""Per-dataset clustering runs, tuned without labels: the helpers behind experiments/01-06.

A notebook goes one method at a time: `select_method` picks the settings, `fit_method` fits them, and
`score_methods`, `tried_settings` and `save_dataset_run` turn the fits into results/datasets/ tables.
`run_dataset` does all of it for a list of methods in one call.

K-Means, DBSCAN, HDBSCAN and GMM are tuned by the shared rule (tuning.select_all_methods); K-Means and
GMM get the true k on labelled data, and the label-free choice is recorded. K-Means (random init),
Spectral and Agglomerative use the K-Means k, and Affinity Propagation picks its own number of
exemplars. Spectral, Agglomerative and Affinity Propagation need memory quadratic in the number of
points, so they are skipped above MAX_POINTS_FOR_QUADRATIC_METHODS. Labels are used only for ARI and
the other external scores. `max_clusters` caps the cluster count of the density methods.
"""

import dataclasses
import math
import time

import numpy as np
import pandas as pd

from sc4020 import generate_figures as gf
from sc4020 import methods, tuning
from sc4020.datasets import DATASETS
from sc4020.metrics import dbcv_score, evaluate
from sc4020.paths import results_dir

TOPIC = "datasets"
# The report's shared table for Wine (D6) and Breast Cancer (D7), one block of rows per dataset.
SHARED_TABLE_NAME = "report_d6_d7.csv"
NULL_SEED = 0  # seed of the column shuffle behind the fake-data silhouette
SEED = methods.SEED

# Silhouette and DBCV grow with n^2, so both are estimated on a sample for large data.
SILHOUETTE_SAMPLE = 5000
DBCV_SAMPLE = 3000

# These methods take the k that K-Means chose instead of tuning their own.
METHODS_USING_KMEANS_K = ("K-Means (random init)", "Spectral", "Agglomerative")

# Affinity matrices and the Ward merge tree need memory quadratic in the number of points.
QUADRATIC_METHODS = ("Spectral", "Agglomerative", "Affinity Propagation")
MAX_POINTS_FOR_QUADRATIC_METHODS = 10_000

# Global earthquake catalogues have dozens of natural belts, so every method may use up to 50
# clusters there (a coarse grid keeps GMM's BIC search affordable).
QUAKE_K_RANGE = (2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50)
K_RANGES = {"earthquakes_1965_2016": QUAKE_K_RANGE, "earthquakes_2023": QUAKE_K_RANGE}

# An RBF kernel for roughly globular data and a nearest-neighbour graph for manifold-shaped data
# (report, Section 2 "Spectral clustering").
SPECTRAL_AFFINITY_BY_DATASET = {"wine": "rbf", "mall_customers": "rbf"}
DEFAULT_SPECTRAL_AFFINITY = "nearest_neighbors"
AFFINITY_PROPAGATION_PARAMS_BY_DATASET = {
    "breast_cancer": {"damping": 0.9, "preference": -150},
    "wine": {"damping": 0.8},
}
DEFAULT_AFFINITY_PROPAGATION_PARAMS = {"damping": 0.9}

# Columns run_dataset adds to gf.build_results_table, in the order they appear.
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


def is_too_large(ds, method_name: str) -> bool:
    """Whether `method_name` needs more memory than `ds` allows (see QUADRATIC_METHODS)."""
    return (
        method_name in QUADRATIC_METHODS
        and len(ds.data) > MAX_POINTS_FOR_QUADRATIC_METHODS
    )


def _fixed_params(ds, method_name: str, k: int | None) -> dict:
    if method_name == "Spectral":
        affinity = SPECTRAL_AFFINITY_BY_DATASET.get(ds.name, DEFAULT_SPECTRAL_AFFINITY)
        return {"n_clusters": k, "affinity": affinity}
    if method_name == "Agglomerative":
        return {"n_clusters": k, "linkage": "ward"}
    return AFFINITY_PROPAGATION_PARAMS_BY_DATASET.get(
        ds.name, DEFAULT_AFFINITY_PROPAGATION_PARAMS
    )


def select_method(
    ds, method_name: str, selections=None, k_range=None, max_clusters=None
) -> tuning.Selection:
    """The Selection for one method: the shared rule for the four tuned methods, fixed settings for the rest.

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
        k = None
        if method_name in METHODS_USING_KMEANS_K:
            kmeans_selection = (selections or {}).get("K-Means") or select_method(
                ds, "K-Means", k_range=k_range
            )
            k = kmeans_selection.params["n_clusters"]
        selection = tuning.Selection(_fixed_params(ds, method_name, k), np.nan)
    selection.tuning_seconds = time.perf_counter() - start
    return selection


def select_params(
    ds, method_names, k_range=None, max_clusters=None
) -> dict[str, tuning.Selection]:
    """A Selection per method in `method_names`; K-Means is tuned once when another method needs its k."""
    selections = {}
    if any(name in METHODS_USING_KMEANS_K for name in method_names):
        selections["K-Means"] = select_method(ds, "K-Means", k_range=k_range)
    for name in method_names:
        if name not in selections:
            selections[name] = select_method(
                ds, name, selections, k_range, max_clusters
            )
    return {name: selections[name] for name in method_names}


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
    return table


def tried_settings(ds, selections) -> pd.DataFrame:
    """Every setting the selection tried, one row per (method, setting)."""
    return pd.DataFrame(
        [
            {"dataset": ds.name, "method": name, **setting}
            for name, selection in selections.items()
            for setting in selection.tried
        ]
    )


def run_dataset(
    ds, method_names, k_range=None, max_clusters=None
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """(scores, tried settings, chosen parameters, cluster results) of `method_names` on one dataset."""
    too_large = [name for name in method_names if is_too_large(ds, name)]
    if too_large:
        print(
            f"   skipping {', '.join(too_large)} on {ds.name}: {len(ds.data):,} points is above "
            f"{MAX_POINTS_FOR_QUADRATIC_METHODS:,}"
        )
    method_names = [name for name in method_names if name not in too_large]
    selections = select_params(ds, method_names, k_range, max_clusters)
    results = {
        name: fit_method(ds, name, selection) for name, selection in selections.items()
    }
    params = {name: selection.params for name, selection in selections.items()}
    return (
        score_methods(ds, selections, results),
        tried_settings(ds, selections),
        params,
        results,
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


def _shared_table_params(X, k: int) -> dict[str, dict]:
    """The shared rule's four methods plus Agglomerative (Ward) and Spectral (nearest-neighbour graph) at k."""
    params = {
        name: selection.params
        for name, selection in tuning.select_all_methods(X, k=k).items()
    }
    params["Agglomerative"] = {"n_clusters": k, "linkage": "ward"}
    params["Spectral"] = {"n_clusters": k, "affinity": "nearest_neighbors"}
    return params


def shared_table_rows(ds) -> tuple[list[dict], dict[str, np.ndarray]]:
    """(rows, labels_by_method) of the report's shared table for one labelled dataset.

    Each row adds the fake-data columns: the same selection and fit on a copy with every column
    shuffled."""
    X, k = ds.data, ds.n_true_clusters
    X_null = tuning.shuffle_columns_independently(X, np.random.default_rng(NULL_SEED))
    params = _shared_table_params(X, k)
    null_models = methods.build_models(_shared_table_params(X_null, k))
    rows, labels_by_method = [], {}
    for name, model in methods.build_models(params).items():
        labels, seconds = methods.fit_predict_timed(model, X)
        null = evaluate(X_null, null_models[name].fit_predict(X_null))
        rows.append(
            {
                "dataset": ds.name,
                "method": name,
                "params": params[name],
                **evaluate(X, labels, ds.target),
                "fit_seconds": seconds,
                "null_silhouette": null["silhouette"],
                "null_clusters": null["n_clusters"],
                "null_noise_fraction": null["noise_fraction"],
            }
        )
        labels_by_method[name] = labels
    return rows, labels_by_method


def save_shared_table_rows(rows: list[dict]) -> pd.DataFrame:
    """Write `rows` into results/comparison/report_d6_d7.csv, replacing that dataset's old rows."""
    path = results_dir("comparison") / SHARED_TABLE_NAME
    new_rows = pd.DataFrame(rows)
    if path.exists():
        old_rows = pd.read_csv(path)
        kept = old_rows[~old_rows["dataset"].isin(new_rows["dataset"])]
        new_rows = pd.concat([kept, new_rows], ignore_index=True)
    order = {name: i for i, name in enumerate(DATASETS)}
    table = new_rows.sort_values(
        "dataset", key=lambda column: column.map(order), kind="stable"
    )
    table.to_csv(path, index=False)
    return table
