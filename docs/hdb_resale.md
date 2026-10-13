# HDB resale flats

Clustering of HDB resale transactions from data.gov.sg.

## Data

| Dataset | data.gov.sg ID | Use |
|---|---|---|
| Resale flat prices based on registration date from Jan-2017 onwards | `d_8b84c4ee58e3cfc0ece0d773c8ca6abc` | One row per transaction, the clustering data |
| HDB Resale Price Index (1Q2009 = 100), Quarterly | `d_14f63e595975691e7c24a27ae4c07c79` | Adjusts prices to the latest quarter's price level |

No API key is needed. Downloads are saved as `data/<name>_<date>.csv`; the loader reads the latest
file, so results stay fixed until a new download is made.

```bash
uv run python -m sc4020.datasets.download_dataset hdb_resale    # or let load_dataset("hdb_resale") download on first use
```

## Where things are

| Path | Contents |
|---|---|
| `sc4020.datasets.load_dataset("hdb_resale")` | sample (`n_samples`, `random_state`), optional `jitter` and `scale`; `features_raw`, `sales`, `all_sales`, `price_index` on the Bunch |
| `sc4020.datasets.download_dataset` | data.gov.sg download (initiate, poll, save) with retries when rate limited; `latest_datagov` |
| `sc4020.datasets.preprocess_dataset` | remaining lease and storey parsing, index-adjusted price, `hdb_features`, `jitter_ties`, `standardize` |
| `sc4020.metrics` | `evaluate` (silhouette, Davies-Bouldin, ARI, NMI, coverage), `pairwise_ari`, `cluster_shares`, `single_value_share` (the discrete-value check) |
| `sc4020.tuning` | `k_distance_curve`, `find_knee`, `best_setting_by_shared_rule` (the shared rule, with the optional largest-cluster cap), `select_kmeans_k_by_null_gap` (the K sweep against shuffled columns), `shuffle_columns_independently` |
| `experiments/10_data_gov_hdb_resale_eda.ipynb` | Stage 1 EDA; decisions in section 17 |
| `experiments/11_data_gov_hdb_resale_clustering.ipynb` | Stage 3: parameter choice and comparison for all four methods |
| `experiments/12_data_gov_hdb_resale_ablation.ipynb` | Stage 4: largest-cluster cap, success and failure cases, feature ablation, subsample stability, scalability (runtime against number of sales, up to all sales -> `results/hdb_resale/scalability.csv`, about 5 minutes), cross-dataset table (reads notebook 09's `cross_dataset_comparison.csv`) |

Downloads in `data/` and figures in `results/hdb_resale/figures/` are not committed.
