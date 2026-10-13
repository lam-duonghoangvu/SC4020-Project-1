"""Hotspot study on Earthquakes 1965-2016, Uber NYC and Ecoli: parameter sweeps, the main comparison
and four ablations (initialisation, distance metric, feature scaling, dimensionality), plus the
fake-data check.

Usage: python -m sc4020.experiments.hotspot
Writes results/hotspot/{main,sweep,ablation,init_seeds,fake_data_check}.csv, chosen_params.json and
kdist_curves.json. Figures: python -m sc4020.generate_figures hotspot
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from sc4020 import methods, tuning
from sc4020.datasets import load_earthquakes_1965_2016, load_ecoli, load_uber_pickups
from sc4020.datasets.preprocess_dataset import EARTH_RADIUS_KM, chord_to_km
from sc4020.metrics import evaluate, pairwise_ari
from sc4020.paths import results_dir
from sc4020.screening import null_gap

SEED = 42
TOPIC = "hotspot"
K_GRID = list(range(2, 26))
N_SEEDS = 30
N_FAKE = 3
KMEANS_INITS = {"K-Means (random init)": "random", "K-Means": "k-means++"}
# Grids from the report (Section 3.2, "Parameter grids"): eps at 0.1 to 3 times the k-distance knee.
EPS_FACTORS = (0.10, 0.15, 0.22, 0.32, 0.45, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0)


def hdbscan_sizes(n):
    if n > 5000:
        return [10, 15, 25, 40, 60, 100, 150, 250]
    return [10, 15, 25, 40, 60, 100] if n > 1000 else [5, 8, 12, 20, 30]


def load_all():
    return [load_earthquakes_1965_2016(), load_uber_pickups(), load_ecoli()]


def is_spatial(ds) -> bool:
    return "coords_deg" in ds


def eps_in_km(ds, eps):
    """DBSCAN eps in km: unit-vector data uses chord length, Uber is already in km."""
    return float(chord_to_km(eps)) if ds.data.shape[1] == 3 and is_spatial(ds) else float(eps)


def record(store, ds, method, params, labels, seconds, X=None, extra=None):
    row = {
        "dataset": ds.name,
        "method": method,
        "params": json.dumps(params, sort_keys=True),
        "fit_seconds": seconds,
        **evaluate(ds.data if X is None else X, labels, ds.target, seed=SEED),
        **(extra or {}),
    }
    store.append(row)
    return row


def k_sweep(ds, sweep):
    inertias = []
    for k in K_GRID:
        model = methods.kmeans(k, random_state=SEED)
        labels, seconds = methods.fit_predict_timed(model, ds.data)
        inertias.append(float(model.inertia_))
        record(sweep, ds, "K-Means", {"n_clusters": k}, labels, seconds,
               extra={"inertia": float(model.inertia_), "n_iter": int(model.n_iter_)})
    rows = [r for r in sweep if r["dataset"] == ds.name and r["method"] == "K-Means"]
    k_silhouette = K_GRID[int(np.nanargmax([r["silhouette"] for r in rows]))]
    choice = {
        "elbow": int(tuning.elbow_k(inertias, K_GRID)),
        "silhouette": int(k_silhouette),
        # the shared rule: the number of classes when known, else the silhouette's k
        "selected": int(ds.n_true_clusters or k_silhouette),
    }
    if ds.target is not None:
        choice["best_ari"] = int(K_GRID[int(np.nanargmax([r["ari"] for r in rows]))])
    return choice


def density_sweep(ds, sweep, method, selection):
    for t in selection.tried:
        keys = ("eps", "min_samples") if method == "DBSCAN" else ("min_cluster_size", "min_samples")
        params = {k: t[k] for k in keys}
        labels = (methods.dbscan(**params) if method == "DBSCAN" else methods.hdbscan(**params)).fit_predict(ds.data)
        record(sweep, ds, method, params, labels, np.nan)


def main():
    warnings.filterwarnings("ignore")
    out = results_dir(TOPIC)
    datasets = load_all()
    for ds in datasets:
        print(f"{ds.name:22s} n={len(ds.data):6d} d={ds.data.shape[1]}", flush=True)

    sweep, main_rows, ablation, seed_rows, fake_rows = [], [], [], [], []
    chosen = {"k": {}, "dbscan": {}, "hdbscan": {}, "kdist": {}}
    curves = {}

    print("\n[1] K-Means k sweep, [2] DBSCAN and [3] HDBSCAN selection", flush=True)
    for ds in datasets:
        X, y = ds.data, ds.target
        chosen["k"][ds.name] = k_sweep(ds, sweep)
        base_ms = tuning.default_min_samples(X.shape[1])
        curve = tuning.k_distance_curve(X, base_ms)
        curves[ds.name] = curve.tolist()
        chosen["kdist"][ds.name] = {"min_samples": base_ms, "knee": tuning.knee(curve)[1]}
        big = len(X) > 1000
        grids = {
            "dbscan": lambda: tuning.select_dbscan(X, min_samples_grid=[4, 8, 16, 32] if big else [3, 5, 8, 12],
                                                   eps_factors=EPS_FACTORS, y_true=y),
            "hdbscan": lambda: tuning.select_hdbscan(X, min_cluster_sizes=hdbscan_sizes(len(X)), min_samples_grid=(None,), y_true=y),
        }
        for method, key in [("DBSCAN", "dbscan"), ("HDBSCAN", "hdbscan")]:
            selection = grids[key]()
            chosen[key][ds.name] = {"rule": selection.params, "valid": selection.valid, "best_ari": selection.best_ari}
            density_sweep(ds, sweep, method, selection)
        print(f"  {ds.name:22s} k={chosen['k'][ds.name]}  DBSCAN={chosen['dbscan'][ds.name]['rule']}  "
              f"HDBSCAN={chosen['hdbscan'][ds.name]['rule']}", flush=True)

    print("\n[4] main comparison", flush=True)
    for ds in datasets:
        k = chosen["k"][ds.name]["selected"]
        for name, init in KMEANS_INITS.items():
            labels, seconds = methods.fit_predict_timed(methods.kmeans(k, init=init, random_state=SEED), ds.data)
            record(main_rows, ds, name, {"n_clusters": k}, labels, seconds)
        for method, factory, key in [("DBSCAN", methods.dbscan, "dbscan"), ("HDBSCAN", methods.hdbscan, "hdbscan")]:
            params = chosen[key][ds.name]["rule"]
            labels, seconds = methods.fit_predict_timed(factory(**params), ds.data)
            record(main_rows, ds, method, params, labels, seconds)

    print(f"\n[5] ablation: initialisation, {N_SEEDS} seeds, 1 start each", flush=True)
    for ds in datasets:
        k = chosen["k"][ds.name]["selected"]
        for name, init in KMEANS_INITS.items():
            runs = []
            for seed in range(N_SEEDS):
                model = methods.kmeans(k, init=init, n_init=1, random_state=seed)
                labels, seconds = methods.fit_predict_timed(model, ds.data)
                ari = evaluate(ds.data, labels, ds.target)["ari"] if ds.target is not None else np.nan
                runs.append(labels)
                seed_rows.append({"dataset": ds.name, "method": name, "k": k, "seed": seed, "inertia": float(model.inertia_),
                                  "n_iter": int(model.n_iter_), "fit_seconds": seconds, "ari": ari})
            rows = pd.DataFrame([r for r in seed_rows if r["dataset"] == ds.name and r["method"] == name])
            ablation.append({
                "ablation": "init", "dataset": ds.name, "method": name, "k": k,
                "inertia_mean": rows.inertia.mean(), "inertia_std": rows.inertia.std(ddof=0), "inertia_min": rows.inertia.min(),
                "inertia_cv_pct": 100 * rows.inertia.std(ddof=0) / rows.inertia.mean(),
                "iter_mean": rows.n_iter.mean(), "iter_std": rows.n_iter.std(ddof=0), "fit_seconds_mean": rows.fit_seconds.mean(),
                "ari_mean": rows.ari.mean(), "ari_std": rows.ari.std(ddof=0), "ari_min": rows.ari.min(), "ari_max": rows.ari.max(),
                "pairwise_ari_mean": float(pairwise_ari(runs).mean()),
            })

    print("\n[6] ablation: distance metric on coordinates", flush=True)
    for ds in [d for d in datasets if is_spatial(d)]:
        p = chosen["dbscan"][ds.name]["rule"]
        eps_km, ms = eps_in_km(ds, p["eps"]), p["min_samples"]
        native = "euclidean_unit_vectors" if ds.data.shape[1] == 3 else "euclidean_km"
        variants = [
            ("haversine", ds.coords_rad, eps_km / EARTH_RADIUS_KM, "haversine"),
            ("euclidean_degrees", ds.coords_deg, eps_km / 111.32, "euclidean"),
            (native, ds.data, p["eps"], "euclidean"),
        ]
        for variant, Z, eps, metric in variants:
            labels, seconds = methods.fit_predict_timed(methods.dbscan(eps, ms, metric=metric), Z)
            ablation.append({"ablation": "metric", "dataset": ds.name, "variant": variant, "eps": eps, "eps_km": eps_km,
                             "min_samples": ms, "fit_seconds": seconds, **evaluate(ds.data, labels, seed=SEED)})

    ecoli = next(d for d in datasets if d.name == "ecoli")
    k_ecoli = chosen["k"]["ecoli"]["selected"]

    def best_ari_dbscan(X, y, ms=5):
        """DBSCAN at the best-ARI eps on a fine grid around the knee: its best possible showing."""
        sel = tuning.select_dbscan(X, min_samples_grid=[ms], eps_factors=np.linspace(0.10, 2.5, 30), y_true=y)
        params = sel.best_ari
        return params["eps"], evaluate(X, methods.dbscan(**params).fit_predict(X), y, seed=SEED)

    print("\n[7] ablation: feature scaling (Ecoli)", flush=True)
    raw = load_ecoli(scale=False).data
    for variant, X in [("raw", raw), ("standardized", ecoli.data)]:
        labels = methods.kmeans(k_ecoli, random_state=SEED).fit_predict(X)
        ablation.append({"ablation": "scaling", "dataset": "ecoli", "variant": variant, "method": "K-Means",
                         **evaluate(X, labels, ecoli.target, seed=SEED)})
        eps, m = best_ari_dbscan(X, ecoli.target)
        ablation.append({"ablation": "scaling", "dataset": "ecoli", "variant": variant, "method": "DBSCAN", "eps": eps, **m})

    print("\n[8] ablation: dimensionality (Ecoli, PCA)", flush=True)
    for dims in [2, 3, 4, 5, 7]:
        X = PCA(n_components=dims, random_state=SEED).fit_transform(ecoli.data) if dims < 7 else ecoli.data
        grid = np.linspace(0.2, 4, 40)
        sel = tuning.select_dbscan(X, min_samples_grid=[5], eps_factors=grid, y_true=ecoli.target)
        viable = sum(t["n_clusters"] >= 2 and t["noise_fraction"] < 0.5 for t in sel.tried) / len(grid)
        eps, m = best_ari_dbscan(X, ecoli.target)
        km = evaluate(X, methods.kmeans(k_ecoli, random_state=SEED).fit_predict(X), ecoli.target, seed=SEED)
        ablation.append({"ablation": "dimensionality", "dataset": "ecoli", "variant": f"{dims}D", "dims": dims,
                         "eps_knee": tuning.knee(tuning.k_distance_curve(X, 5))[1], "eps_best": eps, "viable_eps_frac": viable,
                         "dbscan_ari": m["ari"], "dbscan_noise": m["noise_fraction"], "dbscan_clusters": m["n_clusters"],
                         "kmeans_ari": km["ari"]})

    print("\n[9] fake-data check", flush=True)
    for ds in [d for d in datasets if d.name != "earthquakes_1965_2016"]:
        density_null = "uniform_box" if is_spatial(ds) else "shuffled"
        k = chosen["k"][ds.name]
        runs = [
            ("K-Means", f"k={k['silhouette']} (silhouette)", "shuffled",
             lambda Z, k=k["silhouette"]: methods.kmeans(k, random_state=SEED).fit_predict(Z)),
            ("K-Means", f"k={k['selected']} (selected)", "shuffled",
             lambda Z, k=k["selected"]: methods.kmeans(k, random_state=SEED).fit_predict(Z)),
            ("DBSCAN", json.dumps(chosen["dbscan"][ds.name]["rule"]), density_null,
             lambda Z, p=chosen["dbscan"][ds.name]["rule"]: methods.dbscan(**p).fit_predict(Z)),
            ("HDBSCAN", json.dumps(chosen["hdbscan"][ds.name]["rule"]), density_null,
             lambda Z, p=chosen["hdbscan"][ds.name]["rule"]: methods.hdbscan(**p).fit_predict(Z)),
        ]
        for method, setting, null, fit in runs:
            fake_rows.append({"dataset": ds.name, "method": method, "setting": setting,
                              **null_gap(ds.data, fit, null=null, n_nulls=N_FAKE, seed=SEED, y_true=ds.target)})

    pd.DataFrame(main_rows).to_csv(out / "main.csv", index=False)
    pd.DataFrame(sweep).to_csv(out / "sweep.csv", index=False)
    pd.DataFrame(ablation).to_csv(out / "ablation.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out / "init_seeds.csv", index=False)
    pd.DataFrame(fake_rows).to_csv(out / "fake_data_check.csv", index=False)
    (out / "chosen_params.json").write_text(json.dumps(chosen, indent=2, default=str))
    (out / "kdist_curves.json").write_text(json.dumps(curves))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(pd.DataFrame(main_rows)[["dataset", "method", "params", "n_clusters", "noise_fraction", "silhouette", "ari"]].round(3))
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
