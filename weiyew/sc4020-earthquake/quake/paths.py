"""Project paths: downloaded data under data/raw/, experiment outputs under results/ and figures/."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def results_dir():
    path = PROJECT_ROOT / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir():
    path = PROJECT_ROOT / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path
