"""Loaders for every dataset the team uses, with each owner's preprocessing.

Downloads go to hoanganh/data/team/ (gitignored); Kaggle datasets come from the
kagglehub cache. Each loader returns a `sc4020.data.Dataset` whose `extra`
holds the owner and, for map data, (Lat, Lon) in degrees for plotting.

One deliberate change from the owners' code: global earthquake catalogues are
clustered as 3D unit vectors (as in weiyew/sc4020-earthquake), so all four
methods share one distance with no break at +/-180 longitude. Uber pickups
are projected to km (as in SC4020-Ky) so both axes use the same unit.
"""

from __future__ import annotations

import glob
import os
import sys
import warnings
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HOANGANH = Path(__file__).resolve().parents[1]
REPO = HOANGANH.parent
sys.path.insert(0, str(HOANGANH / "src"))
sys.path.insert(0, str(REPO / "weiyew" / "sc4020-earthquake"))
sys.path.insert(0, str(REPO / "weiyew" / "sc4020-hdb"))

from hdb import features as hdb_features  # noqa: E402
from hdb.datagov import download_dataset  # noqa: E402
from quake import usgs  # noqa: E402

from sc4020 import data  # noqa: E402
from sc4020.data import Dataset  # noqa: E402

DATA_DIR = HOANGANH / "data" / "team"
EARTH_RADIUS_KM = 6371.0088

EARTHQUAKES_1965_URL = "https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv"
ECOLI_URL = (
    "https://raw.githubusercontent.com/deric/clustering-benchmark/master/"
    "src/main/resources/datasets/real-world/ecoli.arff"
)
MALL_SLUG = "vjchoudhary7/customer-segmentation-tutorial-in-python"
FASHION_SLUG = "zalando-research/fashionmnist"
FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def _download(url: str, name: str) -> Path:
    path = DATA_DIR / name
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        import requests

        response = requests.get(url, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def _latest(pattern: str) -> Path | None:
    files = sorted(DATA_DIR.glob(pattern))
    return files[-1] if files else None


def _unit_vectors(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _with_owner(ds: Dataset, owner: str, name: str | None = None) -> Dataset:
    ds.extra = {**ds.extra, "owner": owner}
    if name:
        ds.name = name
    return ds


# --- hoanganh -------------------------------------------------------------


def load_wine() -> Dataset:
    return _with_owner(data.load_wine_data(), "hoanganh, lam")


def load_breast_cancer() -> Dataset:
    return _with_owner(data.load_breast_cancer_data(), "hoanganh")


def load_uber(n_samples: int = 15000) -> Dataset:
    """April 2014 NYC pickups (hoanganh's sample), projected to km."""
    ds = data.load_uber_pickups(n_samples=n_samples)
    lat, lon = ds.X[:, 0], ds.X[:, 1]
    lat0 = np.radians(lat.mean())
    X = np.column_stack([
        EARTH_RADIUS_KM * np.radians(lon) * np.cos(lat0),
        EARTH_RADIUS_KM * np.radians(lat),
    ])
    return Dataset("uber_nyc", X, None, ["x_km", "y_km"], None,
                   {"coords_deg": ds.X, "owner": "hoanganh, Ky"})


# --- lam ------------------------------------------------------------------


def load_moons() -> Dataset:
    return _with_owner(data.load_moons_data(), "lam, hoanganh")


def load_mall_customers() -> Dataset:
    path = glob.glob(os.path.join(kagglehub.dataset_download(MALL_SLUG), "*.csv"))[0]
    df = pd.read_csv(path)
    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    X = StandardScaler().fit_transform(df[features].to_numpy(dtype=float))
    return Dataset("mall_customers", X, None, features, None, {"owner": "lam"})


def load_fashion_mnist(n_samples: int = 3000, n_pca: int = 50, random_state: int = 42) -> Dataset:
    """lam's setup: stratified 3,000-image sample of train+test, pixels / 255, PCA to 50."""
    folder = kagglehub.dataset_download(FASHION_SLUG)
    full = pd.concat(
        [pd.read_csv(os.path.join(folder, f"fashion-mnist_{part}.csv")) for part in ("train", "test")],
        ignore_index=True,
    )
    y_full = full["label"].to_numpy()
    pixels_full = full.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0
    _, pixels, _, y = train_test_split(
        pixels_full, y_full, test_size=n_samples / len(full), stratify=y_full, random_state=random_state
    )
    X = PCA(n_components=n_pca, random_state=random_state).fit_transform(pixels).astype(float)
    return Dataset("fashion_mnist", X, y, [f"pca_{i}" for i in range(n_pca)], FASHION_CLASSES,
                   {"owner": "lam"})


# --- Ky -------------------------------------------------------------------


def load_earthquakes_1965_2016(min_mag: float = 5.5) -> Dataset:
    """Ky's catalogue: global M>=5.5 earthquakes, 1965-2016."""
    df = pd.read_csv(_download(EARTHQUAKES_1965_URL, "earthquakes_1965_2016.csv"))
    df = df[df["Magnitude"] >= min_mag].dropna(subset=["Latitude", "Longitude"])
    coords = df[["Latitude", "Longitude"]].to_numpy(dtype=float)
    return Dataset("earthquakes_1965_2016", _unit_vectors(*coords.T), None, ["x", "y", "z"], None,
                   {"coords_deg": coords, "owner": "Ky"})


def load_ecoli() -> Dataset:
    """Ky's UCI Ecoli: 336 proteins, 7 features (standardized), 8 localisation sites."""
    raw, _ = loadarff(_download(ECOLI_URL, "ecoli.arff"))
    df = pd.DataFrame(raw)
    label = df.columns[-1]
    y = df[label].astype(str).astype("category").cat.codes.to_numpy()
    features = df.drop(columns=[label])
    X = StandardScaler().fit_transform(features.to_numpy(dtype=float))
    return Dataset("ecoli", X, y, list(features.columns), None, {"owner": "Ky"})


# --- weiyew ---------------------------------------------------------------


def load_earthquakes_2023() -> Dataset:
    """weiyew's catalogue: USGS M>=4.5 earthquakes in 2023."""
    path = usgs.catalogue_path(raw_dir=DATA_DIR)
    if not path.exists():
        usgs.download_events(usgs.DEFAULT_START, usgs.DEFAULT_END, usgs.DEFAULT_MIN_MAGNITUDE, raw_dir=DATA_DIR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # drops non-earthquake events with a warning
        events = usgs.load_earthquakes(path)
    coords = events[["latitude", "longitude"]].to_numpy(dtype=float)
    return Dataset("earthquakes_2023", _unit_vectors(*coords.T), None, ["x", "y", "z"], None,
                   {"coords_deg": coords, "owner": "weiyew"})


def load_hdb_resale(sample_size: int = 20_000, seed: int = 0) -> Dataset:
    """weiyew's setup: index-adjusted resale sales, 4 features, 20,000-sale sample
    (seed 0), ties jittered, standardized.

    Jittered data is used for all four methods: 78% of sales share their exact
    floor area, lease and storey with another sale, which makes density methods
    find one cluster per storey band (weiyew EDA section 10); K-Means barely moves.
    """
    for name in ("resale", "price_index"):
        if _latest(f"{name}_*.csv") is None:
            download_dataset(name, raw_dir=DATA_DIR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # drops sales without a price index value
        feats = hdb_features.screening_features(
            hdb_features.load_resale(_latest("resale_*.csv")),
            hdb_features.load_price_index(_latest("price_index_*.csv")),
        )
    sample = feats.sample(n=sample_size, random_state=seed)
    X = hdb_features.standardize(hdb_features.jitter_ties(sample, np.random.default_rng(seed)))
    return Dataset("hdb_resale", X, None, list(sample.columns), None, {"owner": "weiyew"})


# dataset name -> loader, in the order they are reported
LOADERS = {
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "uber_nyc": load_uber,
    "make_moons": load_moons,
    "mall_customers": load_mall_customers,
    "fashion_mnist": load_fashion_mnist,
    "earthquakes_1965_2016": load_earthquakes_1965_2016,
    "ecoli": load_ecoli,
    "earthquakes_2023": load_earthquakes_2023,
    "hdb_resale": load_hdb_resale,
}
