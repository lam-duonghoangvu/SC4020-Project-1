# SC4020 Project 1

Clustering technical review: K-Means / K-Means++, DBSCAN, HDBSCAN and GMM compared across the
team's datasets. The report is in `report/`.

## Layout

One Python project (`pyproject.toml`) with one package, `sc4020`:

| Path                                     | Contents                                                                                                           |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `src/sc4020/datasets/`                   | every dataset by name: `load_dataset.py`, `download_dataset.py`, `preprocess_dataset.py`                           |
| `src/sc4020/methods.py`                  | the clustering methods and a timed runner                                                                          |
| `src/sc4020/tuning.py`                   | the shared selection rule (k-distance knee, K-Means, GMM, DBSCAN, HDBSCAN), K by the gap over fake data            |
| `src/sc4020/metrics.py`                  | validity scores, DBCV, distances on the sphere                                                                     |
| `src/sc4020/clusters.py`                 | cluster renumbering and centres                                                                                    |
| `src/sc4020/generate_figures.py`         | every figure and output table                                                                                      |
| `src/sc4020/run.py`                      | per-dataset helpers behind notebooks 01-06: select, fit and score one method at a time (see `docs/datasets.md`)   |
| `experiments/`                           | every experiment as a numbered notebook (`NN_<source>_<dataset>_<step>.ipynb`); run them in order                 |
| `results/<topic>/`                       | result CSVs and figures cited in the report: `comparison`, `datasets`, `hotspot`, `earthquakes_2023`, `hdb_resale` |
| `data/`                                  | `README.md` lists every dataset and how it is downloaded; downloads are gitignored                                 |
| `tests/`                                 | unit tests                                                                                                         |
| `docs/`                                  | notes per topic, and the hotspot study's individual draft report                                                   |
| `submission/`                            | builds the code zip: `python submission/make_code_zip.py --group XX`                                               |

## Setup

    uv sync
    uv run pytest

Every experiment is a notebook in `experiments/`, numbered in the order to run them; each also runs on its own
once the notebooks it reads from have run (only 12 needs 09). Figures are shown in the notebook and saved to
`results/<topic>/figures/`:

    uv run jupyter nbconvert --to notebook --execute --inplace experiments/01_scikit_learn_wine_clustering.ipynb

| Notebook | What it does |
| --- | --- |
| `01_scikit_learn_wine_clustering.ipynb` | Wine (D6): every method, metrics, cluster plots, and the report's shared-table rows |
| `02_scikit_learn_breast_cancer_clustering.ipynb` | Breast Cancer (D7): the same, plus `report/figures/bc_cases.png` |
| `03_scikit_learn_make_moons_clustering.ipynb` | make_moons (D1) |
| `04_kaggle_mall_customers_clustering.ipynb` | Mall Customers |
| `05_kaggle_fashion_mnist_clustering.ipynb` | Fashion-MNIST (D5) |
| `06_kaggle_uber_pickups_clustering.ipynb` | Uber NYC pickups (D3) |
| `07_usgs_earthquakes_2023_eda.ipynb` | Earthquakes 2023 (D2): exploratory analysis behind each preprocessing choice |
| `08_usgs_earthquakes_2023_clustering.ipynb` | Earthquakes 2023: parameter choice, initialisation, cluster maps, comparison |
| `09_usgs_earthquakes_2023_ablation.ipynb` | Earthquakes 2023: ablations, success and failure cases, scalability, cross-dataset table |
| `10_data_gov_hdb_resale_eda.ipynb` | HDB resale (D8): exploratory analysis |
| `11_data_gov_hdb_resale_clustering.ipynb` | HDB resale: parameter choice and comparison |
| `12_data_gov_hdb_resale_ablation.ipynb` | HDB resale: largest-cluster cap, cases, feature ablation, stability, scalability (needs 09) |
| `13_multi_source_hotspot_clustering.ipynb` | Earthquakes 1965-2016, Uber, Ecoli: sweeps, main comparison, ablations A-D, fake-data check, shared-table rows, case statistics, figures |
| `14_multi_source_hotspot_study.ipynb` | earlier exploratory hotspot study on Ridgecrest 2019, Chicago thefts, Ecoli and make_moons |

The command for each part of the report is in `submission/README_code.md`; topic notes are in `docs/`.
