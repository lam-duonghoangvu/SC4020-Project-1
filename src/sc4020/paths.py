"""Repository folders: downloaded data, experiment results and the report's figures."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data"
RESULTS_DIR = REPO / "results"
REPORT_FIGURES_DIR = REPO / "report" / "figures"


def results_dir(topic: str) -> Path:
    """results/<topic>/, created if missing."""
    path = RESULTS_DIR / topic
    path.mkdir(parents=True, exist_ok=True)
    return path
