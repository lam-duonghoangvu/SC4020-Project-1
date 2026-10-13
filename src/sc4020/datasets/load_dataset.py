"""Load every dataset by name with `load_dataset`, the way `sklearn.datasets` does.

    from sc4020.datasets import load_dataset
    wine = load_dataset("wine")             # Bunch: data, target, feature_names, target_names, DESCR
    X, y = load_dataset("wine", return_X_y=True)
    uber = load_dataset("uber_pickups")     # downloads into data/ on first use

Every dataset takes the same keywords:
  data_home            folder holding downloaded files (default: the repo's data/ folder)
  download_if_missing  download a missing file (True), or raise FileNotFoundError (False)
  return_X_y           return (data, target) instead of the Bunch
  as_frame             data as a DataFrame, target as a Series, and `frame` holding both
plus the dataset's own preprocessing options, whose defaults are the settings used in the report.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets as skd
from sklearn.utils import Bunch

from sc4020.datasets import download_dataset as dl
from sc4020.datasets import preprocess_dataset as pp
from sc4020.paths import DATA_DIR

NOISE_LABEL = -1


def n_true_clusters(target) -> int | None:
    """Number of classes in a target, ignoring the noise label -1; None without a target."""
    if target is None:
        return None
    target = np.asarray(target)
    return int(len(np.unique(target[target != NOISE_LABEL])))


def _require(path: Path, name: str, download, data_home, download_if_missing):
    """Download `name` when `path` is missing, or raise if downloading is switched off."""
    if Path(path).exists():
        return
    if not download_if_missing:
        raise FileNotFoundError(
            f"{path} not found; run: python -m sc4020.datasets.download_dataset {name}"
        )
    download(data_home=data_home)


def _home(data_home) -> Path:
    return Path(data_home) if data_home is not None else DATA_DIR


def _finish(bunch: Bunch, return_X_y: bool, as_frame: bool):
    bunch.n_true_clusters = n_true_clusters(bunch.target)
    if as_frame:
        bunch.data = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        if bunch.target is not None:
            bunch.target = pd.Series(bunch.target, name="target")
        bunch.frame = (
            bunch.data
            if bunch.target is None
            else pd.concat([bunch.data, bunch.target], axis=1)
        )
    if return_X_y:
        return bunch.data, bunch.target
    return bunch


def _bunch(name, data, target, feature_names, target_names, descr, **extra) -> Bunch:
    return Bunch(
        name=name,
        data=data,
        target=target,
        feature_names=list(feature_names),
        target_names=None if target_names is None else list(target_names),
        DESCR=descr,
        frame=None,
        **extra,
    )


# --- built into scikit-learn ------------------------------------------------------------------


def _load_wine(
    *,
    data_home=None,
    download_if_missing=True,
    return_X_y=False,
    as_frame=False,
    scale=True,
):
    """UCI Wine: 178 wines x 13 chemical measurements, 3 cultivars. Standardized when `scale`."""
    raw = skd.load_wine()
    bunch = _bunch(
        "wine",
        pp.standardize(raw.data) if scale else raw.data.astype(float),
        raw.target,
        raw.feature_names,
        raw.target_names,
        "UCI Wine (scikit-learn copy): 178 x 13, 3 cultivars. Features standardized.",
        data_raw=raw.data,
    )
    return _finish(bunch, return_X_y, as_frame)


def _make_moons(
    *,
    data_home=None,
    download_if_missing=True,
    return_X_y=False,
    as_frame=False,
    n_samples=1000,
    noise=0.07,
    random_state=42,
    scale=True,
):
    """Two interleaving half-moons: non-convex clusters with known labels."""
    X, y = skd.make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    bunch = _bunch(
        "make_moons",
        pp.standardize(X) if scale else X,
        y,
        ["x1", "x2"],
        ["moon_0", "moon_1"],
        f"scikit-learn make_moons: {n_samples} points, noise {noise}, seed {random_state}. Standardized.",
        data_raw=X,
    )
    return _finish(bunch, return_X_y, as_frame)


# --- Kaggle -----------------------------------------------------------------------------------


def _load_fashion_mnist(
    *,
    data_home=None,
    download_if_missing=True,
    return_X_y=False,
    as_frame=False,
    n_samples=3000,
    n_pca=50,
    random_state=42,
):
    """Fashion-MNIST: stratified sample of train+test, pixels / 255, PCA to `n_pca` components."""
    home = _home(data_home)
    paths = [home / name for name in dl.FASHION_FILES]
    for path in paths:
        _require(
            path,
            "fashion_mnist",
            dl.download_fashion_mnist,
            data_home,
            download_if_missing,
        )
    full = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    X, y, pixels, pca = pp.fashion_mnist_sample(full, n_samples, n_pca, random_state)
    bunch = _bunch(
        "fashion_mnist",
        X,
        y,
        [f"pca_{i}" for i in range(n_pca)],
        [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ],
        f"Kaggle Fashion-MNIST: stratified {n_samples:,} of 70,000 images, pixels / 255, PCA to {n_pca}.",
        pixels=pixels,
        pca_model=pca,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )
    return _finish(bunch, return_X_y, as_frame)


def _load_uber_pickups(
    *,
    data_home=None,
    download_if_missing=True,
    return_X_y=False,
    as_frame=False,
    n_samples=15000,
    random_state=42,
):
    """Uber pickups in NYC, April 2014: `n_samples` pickups inside the city, projected to km."""
    path = _home(data_home) / dl.UBER_FILE
    _require(
        path, "uber_pickups", dl.download_uber_pickups, data_home, download_if_missing
    )
    raw = pd.read_csv(path)
    inside = pp.nyc_pickups(raw)
    sample = inside.sample(n=min(n_samples, len(inside)), random_state=random_state)
    deg = sample[["Lat", "Lon"]].to_numpy(dtype=float)
    bunch = _bunch(
        "uber_pickups",
        pp.project_km(deg[:, 0], deg[:, 1]),
        None,
        ["x_km", "y_km"],
        None,
        f"FiveThirtyEight Uber pickups, April 2014: {len(sample):,} of {len(inside):,} pickups inside NYC "
        f"(seed {random_state}), projected to km. No labels.",
        coords_deg=deg,
        coords_rad=pp.to_radians(deg[:, 0], deg[:, 1]),
        n_rows_raw=len(raw),
        n_rows_in_nyc=len(inside),
    )
    return _finish(bunch, return_X_y, as_frame)


# --- GitHub -----------------------------------------------------------------------------------


def _load_ecoli(
    *,
    data_home=None,
    download_if_missing=True,
    return_X_y=False,
    as_frame=False,
    scale=True,
):
    """UCI Ecoli: 336 proteins x 7 scores, 8 localisation sites (codes in alphabetical order)."""
    path = _home(data_home) / dl.ECOLI_FILE
    _require(path, "ecoli", dl.download_ecoli, data_home, download_if_missing)
    features, y, sites = pp.ecoli_table(path)
    raw = features.to_numpy()
    bunch = _bunch(
        "ecoli",
        pp.standardize(raw) if scale else raw,
        y,
        features.columns,
        sites,
        "UCI Ecoli (Horton & Nakai 1996): 336 x 7, 8 localisation sites. Standardized.",
        data_raw=raw,
    )
    return _finish(bunch, return_X_y, as_frame)


# --- USGS -------------------------------------------------------------------------------------


def _earthquake_bunch(name, events, descr, lat_col="latitude", lon_col="longitude"):
    deg = events[[lat_col, lon_col]].to_numpy(dtype=float)
    return _bunch(
        name,
        pp.to_unit_vectors(deg[:, 0], deg[:, 1]),
        None,
        ["x", "y", "z"],
        None,
        descr,
        coords_deg=deg,
        coords_rad=pp.to_radians(deg[:, 0], deg[:, 1]),
        coords_km=pp.project_km(deg[:, 0], deg[:, 1]),
        events=events,
    )


def _load_earthquakes_2023(
    *,
    data_home=None,
    download_if_missing=True,
    return_X_y=False,
    as_frame=False,
    start=dl.USGS_START,
    end=dl.USGS_END,
    min_magnitude=dl.USGS_MIN_MAGNITUDE,
):
    """USGS earthquakes in [start, end) with magnitude >= `min_magnitude`, as unit vectors.

    Non-earthquake events (explosions, eruptions) are dropped with a warning."""
    path = dl.usgs_catalogue_path(start, end, min_magnitude, data_home)
    _require(
        path,
        "earthquakes_2023",
        lambda data_home: dl.download_earthquakes_2023(
            data_home, start=start, end=end, min_magnitude=min_magnitude
        ),
        data_home,
        download_if_missing,
    )
    events = pp.usgs_earthquakes(pd.read_csv(path))
    bunch = _earthquake_bunch(
        "earthquakes_2023",
        events,
        f"USGS earthquakes {start} to {end}, M >= {min_magnitude}: {len(events):,} events as 3D unit vectors.",
    )
    return _finish(bunch, return_X_y, as_frame)


# --- data.gov.sg ------------------------------------------------------------------------------


def _load_hdb_resale(
    *,
    data_home=None,
    download_if_missing=True,
    return_X_y=False,
    as_frame=False,
    n_samples=20_000,
    random_state=0,
    jitter=True,
    scale=True,
):
    """HDB resale sales from Jan 2017: floor area, remaining lease, storey and index-adjusted price.

    `n_samples` sales are drawn with `random_state` (None keeps every sale). With `jitter`, tied
    values are spread within half a recording step (preprocess_dataset.jitter_ties) using the same
    seed; with `scale`, the four features are standardized."""
    try:
        resale_path = dl.latest_datagov("resale", data_home)
        index_path = dl.latest_datagov("price_index", data_home)
    except FileNotFoundError:
        if not download_if_missing:
            raise
        resale_path, index_path = dl.download_hdb_resale(data_home)
    sales = pp.check_resale(pd.read_csv(resale_path))
    price_index = pp.price_index_series(pd.read_csv(index_path))
    features = pp.hdb_features(sales, price_index)
    if n_samples is not None:
        features = features.sample(n=n_samples, random_state=random_state)
    values = (
        pp.jitter_ties(features, np.random.default_rng(random_state))
        if jitter
        else features
    )
    bunch = _bunch(
        "hdb_resale",
        pp.standardize(values) if scale else values.to_numpy(dtype=float),
        None,
        features.columns,
        None,
        f"data.gov.sg HDB resale prices ({resale_path.name}) adjusted by the resale price index "
        f"({index_path.name}): {len(features):,} sales x 4 features"
        f"{', ties jittered' if jitter else ''}{', standardized' if scale else ''}. No labels.",
        features_raw=features,
        sales=sales.loc[features.index],
        all_sales=sales,
        price_index=price_index,
    )
    return _finish(bunch, return_X_y, as_frame)


# Reported datasets first, in the order of the report's comparison table.
DATASETS = {
    "wine": _load_wine,
    "uber_pickups": _load_uber_pickups,
    "make_moons": _make_moons,
    "fashion_mnist": _load_fashion_mnist,
    "ecoli": _load_ecoli,
    "earthquakes_2023": _load_earthquakes_2023,
    "hdb_resale": _load_hdb_resale,
}


def load_dataset(name: str, **kwargs):
    """Load a dataset by its name, e.g. load_dataset("earthquakes_2023")."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {list(DATASETS)}")
    return DATASETS[name](**kwargs)
