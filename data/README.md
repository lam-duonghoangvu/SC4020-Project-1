# Data

Every dataset in the project loads by name through `sc4020.datasets`.

```python
from sc4020.datasets import load_dataset

X, y = load_dataset("wine", return_X_y=True)
hdb = load_dataset("hdb_resale")  # Bunch: data, target, feature_names, DESCR, ...
quakes = load_dataset("earthquakes_2023", return_X_y=True)  # other keywords pass through
```

Download datasets manually:

```bash
uv run python -m sc4020.datasets.download_dataset all
uv run python -m sc4020.datasets.download_dataset earthquakes_2023 hdb_resale   # or some of them
```

## Summary

| Dataset                                                                    | `load_dataset` name          | Source                                                                                              | File(s) in `data/`                                  |
| -------------------------------------------------------------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| Wine                                                                       | `wine`                       | scikit-learn copy of the UCI table                                                                  | none                                                |
| Breast Cancer Wisconsin                                                    | `breast_cancer`              | scikit-learn copy of the UCI table                                                                  | none                                                |
| Digits                                                                     | `digits`                     | scikit-learn copy of the UCI table                                                                  | none                                                |
| make_moons                                                                 | `make_moons`                 | scikit-learn `make_moons`                                                                           | none                                                |
| Varied density blobs                                                       | `varied_density`             | scikit-learn `make_blobs` + uniform noise                                                           | none                                                |
| Mall Customers                                                             | `mall_customers`             | Kaggle `vjchoudhary7/customer-segmentation-tutorial-in-python`                                      | `Mall_Customers.csv`                                |
| Fashion-MNIST                                                              | `fashion_mnist`              | Kaggle `zalando-research/fashionmnist`                                                              | `fashion-mnist_train.csv`, `fashion-mnist_test.csv` |
| Uber NYC pickups, Apr 2014                                                 | `uber_pickups`               | Kaggle `fivethirtyeight/uber-pickups-in-new-york-city`                                              | `uber-raw-data-apr14.csv`                           |
| Earthquakes 1965-2016 (M >= 5.5)                                           | `earthquakes_1965_2016`      | plotly sample datasets, `earthquakes-23k.csv` (GitHub)                                              | `earthquakes.csv`                                   |
| Ecoli (protein localization)                                               | `ecoli`                      | UCI, ARFF copy from the `deric/clustering-benchmark` repository (GitHub)                            | `ecoli.arff`                                        |
| Earthquakes 2023 (M >= 4.5)                                                | `earthquakes_2023`           | USGS FDSN event API                                                                                 | `usgs_2023-01-01_2024-01-01_m4.5.csv`               |
| HDB resale prices (Jan 2017 onwards) and resale price index (1Q2009 = 100) | `hdb_resale`                 | data.gov.sg, datasets `d_8b84c4ee58e3cfc0ece0d773c8ca6abc` and `d_14f63e595975691e7c24a27ae4c07c79` | `resale_<date>.csv`, `price_index_<date>.csv`       |
| Ridgecrest 2019 earthquakes (M >= 2)                                       | `ridgecrest_2019`            | USGS FDSN event API                                                                                 | `ridgecrest_2019.csv`                               |
| Chicago thefts, Jul 2023                                                   | `chicago_theft`              | City of Chicago data portal, dataset `ijzp-q8t2`                                                    | `chicago_theft_2023_07.csv`                         |

The last two are used only by `experiments/14_multi_source_hotspot_study.ipynb`. The docstring of each private
loader (`_load_*` / `_make_*` in `load_dataset.py`) and the Bunch's `DESCR` give the preprocessing (scaling, sampling, coordinate projection) and its defaults.

## Expected folder after all downloads

```
data/
├── README.md
├── Mall_Customers.csv
├── fashion-mnist_train.csv
├── fashion-mnist_test.csv
├── uber-raw-data-apr14.csv
├── earthquakes.csv
├── ecoli.arff
├── usgs_2023-01-01_2024-01-01_m4.5.csv
├── resale_<date>.csv
├── price_index_<date>.csv
├── ridgecrest_2019.csv
└── chicago_theft_2023_07.csv
```
