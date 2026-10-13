"""Every dataset in the project, loaded by name like `sklearn.datasets`.

    from sc4020.datasets import load_wine, load_dataset
    X, y = load_wine(return_X_y=True)
    quakes = load_dataset("earthquakes_2023")

load_dataset.py builds the Bunch, download_dataset.py fetches missing files into data/, and
preprocess_dataset.py holds the cleaning and feature steps they share.
"""

from sc4020.datasets.load_dataset import (
    DATASETS,
    load_breast_cancer,
    load_chicago_theft,
    load_dataset,
    load_digits,
    load_earthquakes_1965_2016,
    load_earthquakes_2023,
    load_ecoli,
    load_fashion_mnist,
    load_hdb_resale,
    load_mall_customers,
    load_ridgecrest_2019,
    load_uber_pickups,
    load_wine,
    make_moons,
    make_varied_density,
    n_true_clusters,
)

__all__ = [
    "DATASETS",
    "load_breast_cancer",
    "load_chicago_theft",
    "load_dataset",
    "load_digits",
    "load_earthquakes_1965_2016",
    "load_earthquakes_2023",
    "load_ecoli",
    "load_fashion_mnist",
    "load_hdb_resale",
    "load_mall_customers",
    "load_ridgecrest_2019",
    "load_uber_pickups",
    "load_wine",
    "make_moons",
    "make_varied_density",
    "n_true_clusters",
]
