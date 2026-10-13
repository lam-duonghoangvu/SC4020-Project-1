"""Dataset loaders for the clustering comparison notebooks.

Each loader downloads its CSV through kagglehub and returns a `Dataset`.
`X` is the matrix the clustering algorithms are fitted on (standardized for the
tabular datasets, raw lat/lon degrees for Uber pickups).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import kagglehub
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, load_wine, make_blobs, make_moons
from sklearn.preprocessing import StandardScaler

WINE_SLUG = "harrywang/wine-dataset-for-clustering"
BREAST_CANCER_SLUG = "uciml/breast-cancer-wisconsin-data"
UBER_SLUG = "fivethirtyeight/uber-pickups-in-new-york-city"

# Bounding box that keeps only pickups inside New York City.
NYC_BOUNDS = {"lat": (40.57, 40.92), "lon": (-74.15, -73.70)}


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray | None
    feature_names: list[str]
    class_names: list[str] | None
    extra: dict = field(default_factory=dict)

    @property
    def n_true_clusters(self) -> int | None:
        if self.y is None:
            return None
        return int(len(np.unique(self.y[self.y != -1])))  # -1 marks background noise


def _csv_paths(slug: str) -> list[str]:
    return sorted(glob.glob(os.path.join(kagglehub.dataset_download(slug), "*.csv")))


def load_wine_data() -> Dataset:
    df = pd.read_csv(_csv_paths(WINE_SLUG)[0])
    raw = load_wine()  # ground-truth cultivars (3 classes)
    X = StandardScaler().fit_transform(df)
    return Dataset(
        name="wine",
        X=X,
        y=raw.target,
        feature_names=list(df.columns),
        class_names=list(raw.target_names),
    )


def load_breast_cancer_data() -> Dataset:
    df = pd.read_csv(_csv_paths(BREAST_CANCER_SLUG)[0])
    df = df.drop(columns=[c for c in ["id", "Unnamed: 32"] if c in df.columns])

    y = df["diagnosis"].map({"B": 0, "M": 1}).to_numpy()
    features = df.drop(columns=["diagnosis"])
    X = StandardScaler().fit_transform(features)
    return Dataset(
        name="breast_cancer",
        X=X,
        y=y,
        feature_names=list(features.columns),
        class_names=["Benign", "Malignant"],
    )


def load_uber_pickups(n_samples: int = 15000, random_state: int = 42) -> Dataset:
    """April 2014 Uber pickups inside NYC, subsampled to `n_samples` points.

    `X` holds (Lat, Lon) in degrees; `extra["coords_rad"]` holds the same points
    in radians for the haversine-based DBSCAN.
    """
    apr_file = next(p for p in _csv_paths(UBER_SLUG) if "apr14" in p.lower())
    df = pd.read_csv(apr_file)
    df.columns = df.columns.str.strip()

    (lat_lo, lat_hi), (lon_lo, lon_hi) = NYC_BOUNDS["lat"], NYC_BOUNDS["lon"]
    in_nyc = df["Lat"].between(lat_lo, lat_hi) & df["Lon"].between(lon_lo, lon_hi)
    df_clean = df[in_nyc].dropna(subset=["Lat", "Lon"])

    sample = df_clean.sample(n=n_samples, random_state=random_state)
    coords_deg = sample[["Lat", "Lon"]].to_numpy()
    return Dataset(
        name="uber_pickups",
        X=coords_deg,
        y=None,
        feature_names=["Lat", "Lon"],
        class_names=None,
        extra={
            "coords_rad": np.radians(coords_deg),
            "n_rows_raw": len(df),
            "n_rows_in_nyc": len(df_clean),
        },
    )


def load_moons_data(
    n_samples: int = 1000, noise: float = 0.07, random_state: int = 42
) -> Dataset:
    """Two interleaving half-moons: non-convex clusters with known labels."""
    raw_X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return Dataset(
        name="make_moons",
        X=StandardScaler().fit_transform(raw_X),
        y=y,
        feature_names=["x1", "x2"],
        class_names=["moon_0", "moon_1"],
    )


def load_digits_data() -> Dataset:
    """8x8 handwritten digits (1,797 images, 64 pixels, 10 classes).

    Pixels are kept on their shared 0-16 intensity scale: standardizing would
    blow up the near-constant border pixels into noise dimensions.
    """
    raw = load_digits()
    return Dataset(
        name="digits",
        X=raw.data.astype(float),
        y=raw.target,
        feature_names=list(raw.feature_names),
        class_names=[str(c) for c in raw.target_names],
    )


def load_varied_density_data(n_noise: int = 120, random_state: int = 0) -> Dataset:
    """Two tight, dense blobs next to each other, one wide sparse blob, and
    uniform background noise (label -1).

    The dense pair sit closer together than the spread of the sparse blob, so no
    single density threshold separates the pair and still keeps the sparse blob.
    """
    blobs, y = make_blobs(
        n_samples=[400, 400, 400],
        centers=[[0.0, 0.0], [3.0, 0.0], [10.0, 8.0]],
        cluster_std=[0.3, 0.3, 2.0],
        random_state=random_state,
    )
    noise = np.random.default_rng(random_state).uniform(-6, 16, size=(n_noise, 2))
    return Dataset(
        name="varied_density",
        X=StandardScaler().fit_transform(np.vstack([blobs, noise])),
        y=np.concatenate([y, np.full(n_noise, -1)]),
        feature_names=["x1", "x2"],
        class_names=["dense_0", "dense_1", "sparse"],
        extra={"n_noise": n_noise},
    )
