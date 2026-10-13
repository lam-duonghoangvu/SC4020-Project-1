# SC4020 Project 1: Earthquake locations

The density-structure dataset for the clustering review: global earthquakes of magnitude 4.5 and
above from the USGS catalogue, clustered on latitude and longitude. It sits next to
`../sc4020-clustering` (synthetic and LTA datasets) and follows the protocol in `../AGENTS.md`.

## Layout

Paths are from the repo root.

```
src/sc4020/weiyew/quake/
  paths.py      data/weiyew/quake/raw/, results/weiyew/quake/, results/weiyew/quake/figures/
  usgs.py       download the USGS catalogue into data/raw/; load_earthquakes
  screening.py  Stage 0 tests: shuffled-null silhouette, HDBSCAN vs a null
  features.py   coordinates as radians (haversine) or 3D unit vectors (K-means); great-circle distances
  eda.py        helpers for the EDA notebook (region names, magnitude thresholds, grid-cell counts)
  methods.py    the clustering methods: K-means on unit vectors, DBSCAN (eps in km) and HDBSCAN on haversine
  metrics.py    evaluate (silhouette on great-circle distance), k-distances in km, seed agreement, setting selection
  clusters.py   renumber clusters by size; centre, spread and top regions per cluster
notebooks/weiyew/quake/
  eda.ipynb         exploratory analysis that justifies each preprocessing choice (Stage 1)
  clustering.ipynb  parameter choice, initialization ablation, cluster maps and comparison of the four methods (Stage 3)
  ablation.ipynb    one-change ablations, time and magnitude robustness, success and failure cases, scalability,
                    comparison with the other datasets (Stage 4)
src/sc4020/weiyew/quake/experiments/
  screening.py  Stage 0 screening of the earthquake locations
tests/weiyew/quake/          pytest suite
data/weiyew/quake/raw/       downloaded catalogue (gitignored, never submitted)
results/weiyew/quake/        CSVs written by experiments (per-point label files gitignored)
results/weiyew/quake/figures/  plots written by experiments (gitignored)
```

## Setup

From the repo root:

```bash
uv sync
```

## Getting the data

No API key is needed.

```bash
uv run python -m sc4020.weiyew.quake.usgs                                                   # 2023, magnitude 4.5+
uv run python -m sc4020.weiyew.quake.usgs --start 2023-01-01 --end 2024-01-01 --min-magnitude 4.5
```

The file lands in `data/weiyew/quake/raw/usgs_<start>_<end>_m<magnitude>.csv`. The USGS API refuses queries matching
more than 20,000 events; narrow the dates or raise the magnitude if that happens.

## Running

```bash
uv run pytest tests/weiyew/quake
uv run python -m sc4020.weiyew.quake.experiments.screening
```

## Submission

`git archive` includes only tracked files, so data, results and figures are excluded automatically.
