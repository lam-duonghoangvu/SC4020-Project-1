"""Project paths: downloaded data under data/weiyew/hdb/raw/, experiment outputs under
results/weiyew/hdb/ and results/weiyew/hdb/figures/."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = PROJECT_ROOT / "data" / "weiyew" / "hdb" / "raw"


def results_dir():
    path = PROJECT_ROOT / "results" / "weiyew" / "hdb"
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir():
    path = PROJECT_ROOT / "results" / "weiyew" / "hdb" / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path
