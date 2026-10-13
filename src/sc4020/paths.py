"""Repository folders: downloaded data and experiment results."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data"
RESULTS_DIR = REPO / "results"


def results_dir(topic: str) -> Path:
    """results/<topic>/, created if missing."""
    path = RESULTS_DIR / topic
    path.mkdir(parents=True, exist_ok=True)
    return path
