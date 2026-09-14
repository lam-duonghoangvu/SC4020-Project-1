"""Dataset loaders for the clustering comparison notebooks.

Each loader returns a `Dataset` with an already-scaled feature matrix ready
to be fed directly into the clustering algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, make_moons
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray | None
    feature_names: list[str]
    class_names: list[str] | None
    extra: dict | None = None

    @property
    def n_true_clusters(self) -> int | None:
        if self.y is None:
            return None
        return int(len(np.unique(self.y)))


def load_wine_data() -> Dataset:
    raw = load_wine()
    X = StandardScaler().fit_transform(raw.data)
    return Dataset(
        name="wine",
        X=X,
        y=raw.target,
        feature_names=list(raw.feature_names),
        class_names=list(raw.target_names),
    )


def load_mall_customers() -> Dataset:
    df = pd.read_csv(DATA_DIR / "Mall_Customers.csv")
    feature_names = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    X = StandardScaler().fit_transform(df[feature_names].to_numpy())
    return Dataset(
        name="mall_customers",
        X=X,
        y=None,
        feature_names=feature_names,
        class_names=None,
        extra={"gender": df["Gender"].to_numpy()},
    )


def load_fashion_mnist(
    n_samples: int = 3000, n_pca: int = 50, random_state: int = 42
) -> Dataset:
    train = pd.read_csv(DATA_DIR / "fashion-mnist_train.csv")
    test = pd.read_csv(DATA_DIR / "fashion-mnist_test.csv")
    full = pd.concat([train, test], ignore_index=True)

    y_full = full["label"].to_numpy()
    pixels_full = full.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0

    frac = n_samples / len(full)
    _, pixels, _, y = train_test_split(
        pixels_full,
        y_full,
        test_size=frac,
        stratify=y_full,
        random_state=random_state,
    )

    pca = PCA(n_components=n_pca, random_state=random_state)
    X = pca.fit_transform(pixels)

    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]
    return Dataset(
        name="fashion_mnist",
        X=X,
        y=y,
        feature_names=[f"pca_{i}" for i in range(n_pca)],
        class_names=class_names,
        extra={"pixels": pixels, "pca_model": pca, "explained_variance_ratio": pca.explained_variance_ratio_},
    )


def load_make_moons(
    n_samples: int = 1000, noise: float = 0.07, random_state: int = 42
) -> Dataset:
    raw_X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X = StandardScaler().fit_transform(raw_X)
    return Dataset(
        name="make_moons",
        X=X,
        y=y,
        feature_names=["x1", "x2"],
        class_names=["moon_0", "moon_1"],
    )
