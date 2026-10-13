# Data

No manual download is needed. All datasets are fetched on first use through
[`kagglehub`](https://github.com/Kaggle/kagglehub) and cached locally by it
(a Kaggle account / API token may be required, see the kagglehub docs).

| Notebook | Dataset | Kaggle slug |
| --- | --- | --- |
| `01_wine.ipynb` | [Wine dataset for clustering](https://www.kaggle.com/datasets/harrywang/wine-dataset-for-clustering) | `harrywang/wine-dataset-for-clustering` |
| `02_breast_cancer.ipynb` | [Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) | `uciml/breast-cancer-wisconsin-data` |
| `03_uber_pickups.ipynb` | [Uber pickups in New York City](https://www.kaggle.com/datasets/fivethirtyeight/uber-pickups-in-new-york-city) | `fivethirtyeight/uber-pickups-in-new-york-city` |

The Wine ground-truth labels come from `sklearn.datasets.load_wine()`.
