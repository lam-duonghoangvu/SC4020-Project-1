"""Every dataset in the project, loaded by name like `sklearn.datasets`.

    from sc4020.datasets import load_dataset
    X, y = load_dataset("wine", return_X_y=True)
    quakes = load_dataset("earthquakes_2023")

load_dataset.py builds the Bunch, download_dataset.py fetches missing files into data/, and
preprocess_dataset.py holds the cleaning and feature steps they share.
"""

from sc4020.datasets.load_dataset import DATASETS, load_dataset, n_true_clusters

__all__ = ["DATASETS", "load_dataset", "n_true_clusters"]
