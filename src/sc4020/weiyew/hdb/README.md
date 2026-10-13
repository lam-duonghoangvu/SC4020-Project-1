# SC4020 Project 1: HDB resale flats

Clustering of HDB resale transactions from data.gov.sg, the proposed fourth dataset for the
clustering review. The plan and stage dates are in `../PLAN_HDB.md`; the stage protocol is in
`../AGENTS.md`.

## Data

| Dataset | data.gov.sg ID | Use |
|---|---|---|
| Resale flat prices based on registration date from Jan-2017 onwards | `d_8b84c4ee58e3cfc0ece0d773c8ca6abc` | One row per transaction, the clustering data |
| HDB Resale Price Index (1Q2009 = 100), Quarterly | `d_14f63e595975691e7c24a27ae4c07c79` | Adjusts prices to the latest quarter's price level |

No API key is needed. Downloads are saved as `data/<name>_<date>.csv`; the analysis reads the
latest file, so results stay fixed until a new download is made.

## Setup

From the repo root:

```
uv sync
uv run python -m sc4020.weiyew.hdb.datagov
uv run pytest tests/weiyew/hdb
```

## Layout

Paths are from the repo root; `hdb/` is `src/sc4020/weiyew/hdb/`.

| Path | Contents |
|---|---|
| `hdb/datagov.py` | Download from data.gov.sg (initiate, poll, save), with retries when rate limited |
| `hdb/features.py` | Loaders, remaining lease and storey parsing, index-adjusted price, `jitter_ties`, standardization |
| `hdb/screening.py` | Stage 0 shuffled-null tests and the discrete-value check |
| `hdb/methods.py` | K-means (random, K-means++), DBSCAN and HDBSCAN factories |
| `hdb/metrics.py` | `evaluate` (silhouette, Davies-Bouldin, ARI, NMI), k-distance and knee, seed stability, density setting selection |
| `hdb/paths.py` | Data, results and figures folders |
| `hdb/experiments/screening.py` | Stage 0: writes `results/weiyew/hdb/screening*.csv` and `results/weiyew/hdb/figures/screening.png` |
| `notebooks/weiyew/hdb/eda.ipynb` | Stage 1 EDA; decisions in section 17 |
| `notebooks/weiyew/hdb/clustering.ipynb` | Stage 3: parameter choice and comparison for all four methods |
| `hdb/experiments/scalability.py` | Stage 4: runtime against number of sales, up to all sales; writes `results/weiyew/hdb/scalability.csv` (about 5 minutes) |
| `notebooks/weiyew/hdb/ablation.ipynb` | Stage 4: largest-cluster cap, success and failure cases, feature ablation, subsample stability, scalability, cross-dataset table (run `sc4020.weiyew.hdb.experiments.scalability` first) |
| `tests/weiyew/hdb/` | `pytest` tests for every module |

Downloads in `data/` and figures in `results/weiyew/hdb/figures/` are not committed.
