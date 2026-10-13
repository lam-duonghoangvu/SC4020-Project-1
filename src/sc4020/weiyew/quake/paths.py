"""Project paths: downloaded data under data/weiyew/quake/raw/, experiment outputs under
results/weiyew/quake/ and results/weiyew/quake/figures/."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = PROJECT_ROOT / "data" / "weiyew" / "quake" / "raw"


def results_dir():
    path = PROJECT_ROOT / "results" / "weiyew" / "quake"
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir():
    path = PROJECT_ROOT / "results" / "weiyew" / "quake" / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path
