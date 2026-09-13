# SC4020 Project 1: Earthquake locations

The density-structure dataset for the clustering review: global earthquakes of magnitude 4.5 and
above from the USGS catalogue, clustered on latitude and longitude. It sits next to
`../sc4020-clustering` (synthetic and LTA datasets) and follows the protocol in `../AGENTS.md`.

## Layout

```
quake/
  paths.py      data/raw/, results/, figures/ for this project
  usgs.py       download the USGS catalogue into data/raw/; load_earthquakes
  screening.py  Stage 0 tests: shuffled-null silhouette, HDBSCAN vs a null
  features.py   coordinates as radians (haversine) or 3D unit vectors (K-means); great-circle distances
  eda.py        helpers for the EDA notebook (region names, magnitude thresholds, grid-cell counts)
  methods.py    the clustering methods: K-means on unit vectors, DBSCAN (eps in km) and HDBSCAN on haversine
  metrics.py    evaluate (silhouette on great-circle distance), k-distances in km, seed agreement, setting selection
  clusters.py   renumber clusters by size; centre, spread and top regions per cluster
notebooks/
  eda.ipynb         exploratory analysis that justifies each preprocessing choice (Stage 1)
  clustering.ipynb  parameter choice, initialization ablation, cluster maps and comparison of the four methods (Stage 3)
  ablation.ipynb    one-change ablations, time and magnitude robustness, success and failure cases, scalability,
                    comparison with the other datasets (Stage 4)
experiments/
  screening.py  Stage 0 screening of the earthquake locations
tests/          pytest suite
data/raw/       downloaded catalogue (gitignored, never submitted)
results/        CSVs written by experiments (gitignored)
figures/        plots written by experiments (gitignored)
```

## Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Getting the data

No API key is needed.

```bash
python -m quake.usgs                                                   # 2023, magnitude 4.5+
python -m quake.usgs --start 2023-01-01 --end 2024-01-01 --min-magnitude 4.5
```

The file lands in `data/raw/usgs_<start>_<end>_m<magnitude>.csv`. The USGS API refuses queries matching
more than 20,000 events; narrow the dates or raise the magnitude if that happens.

## Running

```bash
python -m pytest
python -m experiments.screening
```

## Submission

`git archive` includes only tracked files, so data, results and figures are excluded automatically.
