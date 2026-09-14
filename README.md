# SC4020 Project 1: HDB resale flats

Clustering of HDB resale transactions from data.gov.sg, the proposed fourth dataset for the
clustering review. The plan and stage dates are in `../PLAN_HDB.md`; the stage protocol is in
`../AGENTS.md`.

## Data

| Dataset | data.gov.sg ID | Use |
|---|---|---|
| Resale flat prices based on registration date from Jan-2017 onwards | `d_8b84c4ee58e3cfc0ece0d773c8ca6abc` | One row per transaction, the clustering data |
| HDB Resale Price Index (1Q2009 = 100), Quarterly | `d_14f63e595975691e7c24a27ae4c07c79` | Adjusts prices to the latest quarter's price level |

No API key is needed. Downloads are saved as `data/raw/<name>_<date>.csv`; the analysis reads the
latest file, so results stay fixed until a new download is made.

## Setup

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m hdb.datagov
.venv/bin/python -m pytest
```

## Layout

| Path | Contents |
|---|---|
| `hdb/datagov.py` | Download from data.gov.sg (initiate, poll, save), with retries when rate limited |
| `hdb/features.py` | Loaders, remaining lease and storey parsing, index-adjusted price, `jitter_ties`, standardization |
| `hdb/screening.py` | Stage 0 shuffled-null tests and the discrete-value check |
| `hdb/methods.py` | K-means (random, K-means++), DBSCAN and HDBSCAN factories |
| `hdb/metrics.py` | `evaluate` (silhouette, Davies-Bouldin, ARI, NMI), k-distance and knee, seed stability, density setting selection |
| `hdb/paths.py` | Data, results and figures folders |
| `experiments/screening.py` | Stage 0: writes `results/screening*.csv` and `figures/screening.png` |
| `notebooks/eda.ipynb` | Stage 1 EDA; decisions in section 17 |
| `notebooks/clustering.ipynb` | Stage 3: parameter choice and comparison for all four methods |
| `tests/` | `pytest` tests for every module |

`data/`, `results/` and `figures/` are not committed.
