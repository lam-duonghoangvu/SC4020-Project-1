# SC4020 Project 1: Clustering technical review (source code)

Code for the experiments in `group_XX_report.pdf`: K-Means / K-Means++, DBSCAN, HDBSCAN and
GMM, compared on 10 datasets. Datasets and model outputs are not included. Each loader downloads
its public dataset on first use, or the member README says where to place the file.

## Setup

Python 3.13 (Python 3.12 is enough for `weiyew/`).

    pip install -r requirements.txt

On Windows, set `PYTHONUTF8=1` before running.

## Layout

| Folder | Owner | Contents |
|---|---|---|
| `hoanganh/` | Hoang Anh | `src/sc4020/` (clustering, metrics, DBCV, tuning, plots); `notebooks/` Wine, Breast Cancer, Uber NYC; `comparison/` the shared 4-method x 10-dataset comparison |
| `lam/` | Lam | `src/sc4020/` and `notebooks/` for Wine, Mall Customers, Fashion-MNIST, make_moons; `00_summary.ipynb` |
| `SC4020-Ky/` | Ky | `src/` hotspot study on Earthquakes 1965-2016, Uber NYC, Ecoli: `run.py` (experiments), `figures.py` (figures) |
| `weiyew/sc4020-earthquake/` | Wei Yew | USGS 2023 earthquakes: `quake/` package, `experiments/`, `notebooks/`, `tests/` |
| `weiyew/sc4020-hdb/` | Wei Yew | HDB resale flats: `hdb/` package, `experiments/`, `notebooks/`, `tests/` |

Every folder has a `results/` with the CSVs cited in the report. Each member folder also has
its own README with more detail.

## Reproduce

Run each command from the folder named in it.

| Report part | Folder | Command |
|---|---|---|
| Cross-dataset comparison (3.3), runtime (3.6) | `hoanganh/` | `python comparison/run_team_comparison.py`, then `python comparison/run_report_tables.py` |
| Per-dataset analyses: Wine, Breast Cancer, Uber | `hoanganh/` | `notebooks/01_*.ipynb` to `03_*.ipynb` |
| make_moons, Fashion-MNIST, Mall Customers | `lam/` | `notebooks/00_summary.ipynb` to `04_*.ipynb` (data placement: `lam/data/README.md`) |
| Earthquakes 1965-2016, Uber, Ecoli: k sweep, k-distance, ablations | `SC4020-Ky/src/` | `python run.py && python figures.py` (data placement: `SC4020-Ky/README.md`) |
| Earthquakes 2023: screening, clustering, ablations | `weiyew/sc4020-earthquake/` | `python -m quake.usgs`, `python -m experiments.screening`, then the notebooks |
| HDB resale: screening, ablations, scalability | `weiyew/sc4020-hdb/` | `python -m hdb.datagov`, `python -m experiments.screening`, `python -m experiments.scalability`, then the notebooks |

Unit tests: `python -m pytest` in either `weiyew/` project.

## Datasets (not included)

| Dataset | Source |
|---|---|
| Wine, Breast Cancer Wisconsin, Uber NYC pickups (Apr 2014), Mall Customers, Fashion-MNIST | Kaggle, fetched with `kagglehub` or placed by hand |
| make_moons, Wine labels | generated or loaded by scikit-learn |
| Earthquakes 1965-2016 (M >= 5.5) | plotly sample datasets |
| Ecoli | UCI Machine Learning Repository |
| Earthquakes 2023 (M >= 4.5) | USGS FDSN event API |
| HDB resale prices, HDB resale price index | data.gov.sg API |
