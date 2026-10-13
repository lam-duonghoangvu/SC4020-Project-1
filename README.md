# SC4020 Project 1

Clustering technical review: K-Means / K-Means++, DBSCAN, HDBSCAN and GMM compared across the
team's datasets. The report is in `report/`.

## Layout

One Python project (`pyproject.toml`) with one package, `sc4020`:

| Path | Contents |
|---|---|
| `src/sc4020/datasets/` | every dataset by name, like `sklearn.datasets`: `load_dataset.py`, `download_dataset.py`, `preprocess_dataset.py` |
| `src/sc4020/methods.py` | the clustering methods and a timed runner |
| `src/sc4020/tuning.py` | the shared selection rule (k-distance knee, K-Means, GMM, DBSCAN, HDBSCAN) |
| `src/sc4020/metrics.py`, `dbcv.py` | validity scores, DBCV, distances on the sphere |
| `src/sc4020/screening.py`, `clusters.py` | fake-data checks; cluster renumbering and centres |
| `src/sc4020/generate_figures.py` | every figure and output table |
| `src/sc4020/experiments/` | the scripts behind the report's tables |
| `notebooks/` | one notebook per dataset and topic |
| `results/<topic>/` | result CSVs and figures cited in the report: `comparison`, `datasets`, `hotspot`, `earthquakes_2023`, `hdb_resale` |
| `data/` | `README.md` lists every dataset and how it is downloaded; downloads are gitignored |
| `tests/` | unit tests |
| `docs/` | notes per topic, and the hotspot study's individual draft report |
| `submission/` | builds the code zip: `python submission/make_code_zip.py --group XX` |

## Setup

    uv sync
    uv run pytest

Run scripts as modules from the repo root, e.g. `uv run python -m sc4020.experiments.team_comparison`.
The command for each part of the report is in `submission/README_code.md`; topic notes are in `docs/`.
