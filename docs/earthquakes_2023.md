# Earthquake locations, 2023

The density-structure dataset for the clustering review: global earthquakes of magnitude 4.5 and
above from the USGS catalogue, clustered on latitude and longitude.

## Where things are

| Path | Contents |
|---|---|
| `sc4020.datasets.load_dataset("earthquakes_2023")` | downloads the USGS catalogue into `data/` on first use; drops non-earthquake events; unit vectors, degrees and radians |
| `sc4020.datasets.preprocess_dataset` | `to_radians`, `to_unit_vectors`, `region_from_place` |
| `sc4020.metrics` | `evaluate` (silhouette with `metric="haversine"`, Davies-Bouldin on unit vectors), `haversine_km`, `nearest_neighbour_km`, `pairwise_ari` |
| `sc4020.tuning` | `k_distance_curve` (with `metric="haversine"`), `best_setting_by_shared_rule` (the shared rule), `select_kmeans_k_by_null_gap` (K by the gap over shuffled coordinates among seed-stable K), `shuffle_columns_independently` and `uniform_box_null` (fake data) |
| `sc4020.clusters` | renumber clusters by size, centre on the sphere, the 180 degree line check |
| `sc4020.generate_figures` | `cluster_summary`, `magnitude_threshold_table`, `grid_cell_counts` |
| `experiments/07_usgs_earthquakes_2023_eda.ipynb` | exploratory analysis that justifies each preprocessing choice (Stage 1) |
| `experiments/08_usgs_earthquakes_2023_clustering.ipynb` | parameter choice, cluster maps and comparison of the four methods (Stage 3) |
| `experiments/09_usgs_earthquakes_2023_cases.ipynb` | success and failure cases, scalability, comparison with the other datasets |
| `results/earthquakes_2023/` | CSVs written by the notebooks, and their figures in `figures/` (per-point label files and figures are gitignored) |

In these notebooks K-Means runs on 3D unit vectors and DBSCAN and HDBSCAN on radians with the
haversine distance, so eps is a distance in km on the Earth's surface.

## Getting the data

No API key is needed. The loader downloads on first use; to fetch it ahead of time:

```bash
uv run python -m sc4020.datasets.download_dataset earthquakes_2023
```

The file lands in `data/usgs_<start>_<end>_m<magnitude>.csv`. The USGS API refuses queries matching
more than 20,000 events; narrow the dates or raise the magnitude if that happens
(`load_dataset("earthquakes_2023", start=..., end=..., min_magnitude=...)`).

## Running

```bash
uv run jupyter nbconvert --to notebook --execute --inplace experiments/0[789]_usgs_earthquakes_2023_*.ipynb
```

The cases notebook reads `../sc4020-clustering/results` (synthetic and LTA datasets) if that
separate project sits next to this repo; it is not part of this repository.
