# SC4020 Project 1

Clustering technical review: K-Means / K-Means++, DBSCAN, HDBSCAN and GMM compared across the
team's datasets. The report is in `report/`.

## Layout

One Python project (`pyproject.toml`) with one package, `sc4020`:

| Path                             | Contents                                                                                                    |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `src/sc4020/datasets/`           | every dataset by name: `load_dataset.py`, `download_dataset.py`, `preprocess_dataset.py`                    |
| `src/sc4020/methods.py`          | the clustering methods and a timed runner                                                                   |
| `src/sc4020/tuning.py`           | the shared selection rule (k-distance knee, K-Means, GMM, DBSCAN, HDBSCAN), K by the gap over fake data     |
| `src/sc4020/metrics.py`          | validity scores and DBCV                                                                                    |
| `src/sc4020/generate_figures.py` | every figure and output table                                                                               |
| `src/sc4020/run.py`              | the helpers behind every notebook: select, fit and score one method at a time, with the fake-data columns   |
| `experiments/`                   | one clustering notebook per dataset (`NN_<source>_<dataset>_clustering.ipynb`), all on the same template    |
| `results/datasets/`              | written by the notebooks and gitignored: `<dataset>_results.csv`, `<dataset>_tuning.csv` and `figures/`     |
| `data/`                          | `README.md` lists every dataset and how it is downloaded; downloads are gitignored                          |
| `tests/`                         | unit tests                                                                                                  |
| `submission/`                    | builds the code zip: `python submission/make_code_zip.py --group XX`                                        |

## Setup

    uv sync
    uv run pytest

Every experiment is a notebook in `experiments/`, one per dataset, numbered by the report's dataset order. The
notebooks share one template: dataset loading, then K-Means, K-Means++, DBSCAN, HDBSCAN, GMM and Spectral,
then metrics (with the silhouette on fake data) and cluster plots. Each runs on its own; figures are shown in
the notebook and saved to `results/datasets/figures/`:

    uv run jupyter nbconvert --to notebook --execute --inplace experiments/01_scikit_learn_make_moons_clustering.ipynb

| Notebook                                      | Dataset                     |
| --------------------------------------------- | --------------------------- |
| `01_scikit_learn_make_moons_clustering.ipynb` | make_moons (D1)             |
| `02_usgs_earthquakes_2023_clustering.ipynb`   | Earthquakes 2023 (D2)       |
| `03_kaggle_uber_pickups_clustering.ipynb`     | Uber NYC pickups (D3)       |
| `04_uci_ecoli_clustering.ipynb`               | Ecoli (D4)                  |
| `05_kaggle_fashion_mnist_clustering.ipynb`    | Fashion-MNIST (D5)          |
| `06_scikit_learn_wine_clustering.ipynb`       | Wine (D6)                   |
| `07_data_gov_hdb_resale_clustering.ipynb`     | HDB resale (D8)             |

The command for each part of the report is in `submission/README_code.md`.
