# SC4020 Project 1

Clustering technical review: K-Means / K-Means++, DBSCAN, HDBSCAN and GMM compared across the
team's datasets. The report is in `report/`.

## Layout

One Python project (`pyproject.toml`), with each member's work split the same way:

| Folder | Contents |
|---|---|
| `src/sc4020/<member>/` | code: `lam`, `hoanganh`, `ky`, `weiyew/quake`, `weiyew/hdb` |
| `notebooks/<member>/` | notebooks |
| `results/<member>/` | result CSVs (and figures) cited in the report |
| `data/<member>/` | dataset instructions; downloaded data is gitignored |
| `tests/weiyew/` | unit tests |
| `docs/ky/` | Ky's individual draft report |
| `submission/` | builds the code zip: `python submission/make_code_zip.py --group XX` |

## Setup

    uv sync
    uv run pytest

Run scripts as modules from the repo root, e.g. `uv run python -m sc4020.ky.run`. Run commands for
each part are in `submission/README_code.md` and the member READMEs in `src/sc4020/<member>/`.
