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

| Dataset                                                                    | `load_dataset` name | Source                                                                                              | File(s) in `data/`                                  |
| -------------------------------------------------------------------------- | ------------------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| Wine                                                                       | `wine`              | scikit-learn copy of the UCI table                                                                  | none                                                |
| make_moons                                                                 | `make_moons`        | scikit-learn `make_moons`                                                                           | none                                                |
| Fashion-MNIST                                                              | `fashion_mnist`     | Kaggle `zalando-research/fashionmnist`                                                              | `fashion-mnist_train.csv`, `fashion-mnist_test.csv` |
| Uber NYC pickups, Apr 2014                                                 | `uber_pickups`      | Kaggle `fivethirtyeight/uber-pickups-in-new-york-city`                                              | `uber-raw-data-apr14.csv`                           |
| Ecoli (protein localization)                                               | `ecoli`             | UCI, ARFF copy from the `deric/clustering-benchmark` repository (GitHub)                            | `ecoli.arff`                                        |
| Earthquakes 2023 (M >= 4.5)                                                | `earthquakes_2023`  | USGS FDSN event API                                                                                 | `usgs_2023-01-01_2024-01-01_m4.5.csv`               |
| HDB resale prices (Jan 2017 onwards) and resale price index (1Q2009 = 100) | `hdb_resale`        | data.gov.sg, datasets `d_8b84c4ee58e3cfc0ece0d773c8ca6abc` and `d_14f63e595975691e7c24a27ae4c07c79` | `resale_<date>.csv`, `price_index_<date>.csv`       |

The docstring of each private loader (`_load_*` / `_make_*` in `load_dataset.py`) and the Bunch's `DESCR` give the preprocessing (scaling, sampling, coordinate projection) and its defaults.

## Expected folder after all downloads

```
data/
├── README.md
├── fashion-mnist_train.csv
├── fashion-mnist_test.csv
├── uber-raw-data-apr14.csv
├── ecoli.arff
├── usgs_2023-01-01_2024-01-01_m4.5.csv
├── resale_<date>.csv
└── price_index_<date>.csv
```
