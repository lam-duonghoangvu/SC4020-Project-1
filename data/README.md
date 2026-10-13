# Data

Every dataset in the project loads by name through `sc4020.datasets`, the way `sklearn.datasets`
works:

```python
from sc4020.datasets import load_wine, load_hdb_resale, load_dataset

X, y = load_wine(return_X_y=True)
hdb = load_hdb_resale()                 # Bunch: data, target, feature_names, DESCR, ...
quakes = load_dataset("earthquakes_2023")
```

Datasets built into scikit-learn need nothing. Every other loader downloads its file into this folder
the first time it is called and reads the local copy afterwards. Downloaded files are never
committed: only this README is tracked. To fetch everything ahead of time (from the repo root):

```bash
uv run python -m sc4020.datasets.download_dataset all
uv run python -m sc4020.datasets.download_dataset earthquakes_2023 hdb_resale   # or some of them
```

Pass `download_if_missing=False` to a loader to get a `FileNotFoundError` naming the command instead
of a download, and `data_home=...` to keep the files elsewhere.

## Summary

| Dataset | Loader | Source | File(s) in `data/` |
|---|---|---|---|
| Wine | `load_wine` | scikit-learn copy of the UCI table | none (built in) |
| Breast Cancer Wisconsin | `load_breast_cancer` | scikit-learn copy of the UCI table | none (built in) |
| Digits | `load_digits` | scikit-learn copy of the UCI table | none (built in) |
| make_moons | `make_moons` | scikit-learn `make_moons` | none (generated) |
| Varied density blobs | `make_varied_density` | scikit-learn `make_blobs` + uniform noise | none (generated) |
| Mall Customers | `load_mall_customers` | Kaggle `vjchoudhary7/customer-segmentation-tutorial-in-python` | `Mall_Customers.csv` |
| Fashion-MNIST | `load_fashion_mnist` | Kaggle `zalando-research/fashionmnist` | `fashion-mnist_train.csv`, `fashion-mnist_test.csv` |
| Uber NYC pickups, Apr 2014 | `load_uber_pickups` | Kaggle `fivethirtyeight/uber-pickups-in-new-york-city` | `uber-raw-data-apr14.csv` |
| Earthquakes 1965-2016 (M >= 5.5) | `load_earthquakes_1965_2016` | plotly sample datasets, `earthquakes-23k.csv` (GitHub) | `earthquakes.csv` |
| Ecoli (protein localization) | `load_ecoli` | UCI, ARFF copy from the `deric/clustering-benchmark` repository (GitHub) | `ecoli.arff` |
| Earthquakes 2023 (M >= 4.5) | `load_earthquakes_2023` | USGS FDSN event API | `usgs_2023-01-01_2024-01-01_m4.5.csv` |
| HDB resale prices (Jan 2017 onwards) and resale price index (1Q2009 = 100) | `load_hdb_resale` | data.gov.sg, datasets `d_8b84c4ee58e3cfc0ece0d773c8ca6abc` and `d_14f63e595975691e7c24a27ae4c07c79` | `resale_<date>.csv`, `price_index_<date>.csv` |
| Ridgecrest 2019 earthquakes (M >= 2) | `load_ridgecrest_2019` | USGS FDSN event API | `ridgecrest_2019.csv` |
| Chicago thefts, Jul 2023 | `load_chicago_theft` | City of Chicago data portal, dataset `ijzp-q8t2` | `chicago_theft_2023_07.csv` |

The last two are used only by `notebooks/hotspot_study.ipynb`. Each loader's docstring and the
Bunch's `DESCR` give the preprocessing (scaling, sampling, coordinate projection) and its defaults.

## Built into scikit-learn

No download and no files. Wine, Breast Cancer Wisconsin and Digits are copies of the UCI tables
that ship inside the `scikit-learn` package; make_moons and the varied-density blobs are generated
with a fixed seed. The `sc4020` loaders wrap them with the project's preprocessing (Wine and Breast
Cancer standardized, Breast Cancer coded 0 benign / 1 malignant, Digits on its 0-16 pixel scale,
make_moons with noise 0.07 and seed 42).

## Kaggle datasets

Mall Customers, Fashion-MNIST and Uber need a Kaggle account and an API token
(`~/.kaggle/kaggle.json`, or the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables). The
loaders download through `kagglehub` and copy the files into this folder. The Fashion-MNIST archive
also holds the raw `*-ubyte` files; only the two CSVs are used.

Without kagglehub, the Kaggle CLI (`pip install kaggle`) gives the same files:

```bash
kaggle datasets download -d vjchoudhary7/customer-segmentation-tutorial-in-python -p data --unzip
kaggle datasets download -d zalando-research/fashionmnist -p data --unzip
kaggle datasets download -d fivethirtyeight/uber-pickups-in-new-york-city -f uber-raw-data-apr14.csv -p data --unzip
```

## Earthquakes 1965-2016 and Ecoli

Plain files on GitHub, no account needed. By hand:

```bash
curl -o data/earthquakes.csv https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv
curl -o data/ecoli.arff https://raw.githubusercontent.com/deric/clustering-benchmark/master/src/main/resources/datasets/real-world/ecoli.arff
```

`earthquakes.csv` has 23,412 events; the loader keeps magnitude 5.5 and above. `ecoli.arff` has 336
proteins, 7 features and 8 localization sites.

## Earthquakes 2023 and Ridgecrest 2019 (USGS)

No account or API key needed. `load_earthquakes_2023(start=..., end=..., min_magnitude=...)` saves
`data/usgs_<start>_<end>_m<magnitude>.csv`; the defaults are 2023 and magnitude 4.5. The USGS API
refuses queries that match more than 20,000 events; narrow the dates or raise the magnitude if that
happens. Non-earthquake events (explosions, eruptions) are dropped with a warning.

## HDB resale prices and price index (data.gov.sg)

No account or API key needed. Each download saves a dated copy, `data/resale_<YYYY-MM-DD>.csv` and
`data/price_index_<YYYY-MM-DD>.csv`. The loader reads the latest one, so results stay fixed until
you download again (`python -m sc4020.datasets.download_dataset hdb_resale --force`). The downloader
retries when data.gov.sg rate-limits it.

## Chicago thefts (City of Chicago)

No account needed. The Socrata API query keeps thefts in July 2023 (at most 6,000 rows).

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
