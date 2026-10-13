"""Download non-builtin dataset into data/.

Usage:
    python -m sc4020.datasets.download_dataset all
    python -m sc4020.datasets.download_dataset earthquakes_2023 hdb_resale

Each `download_<name>(data_home=None, force=False)` saves the raw file(s) under `data_home`
(default: the repo's data/ folder) and returns the path(s). An existing file is kept unless
`force=True`. The loaders in load_dataset.py call these on first use, so running this module is
only needed to fetch everything ahead of time.

Sources (details in data/README.md):
  Kaggle         Fashion-MNIST, Uber NYC pickups (needs a Kaggle API token)
  GitHub         Ecoli (ARFF copy of the UCI table)
  USGS           Earthquakes 2023
  data.gov.sg    HDB resale prices and price index (no key; rate limited, so calls are retried)
"""

import argparse
import datetime
import glob
import os
import shutil
import sys
import time
from pathlib import Path

import requests

from sc4020.paths import DATA_DIR

TIMEOUT_SECONDS = 120


def _home(data_home) -> Path:
    path = Path(data_home) if data_home is not None else DATA_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fetch_url(
    url: str, target: Path, params: dict | None = None, get=requests.get
) -> Path:
    response = get(url, params=params, timeout=TIMEOUT_SECONDS)
    if not response.ok:
        raise RuntimeError(
            f"{url} returned HTTP {response.status_code}: {response.text.strip()[:500]}"
        )
    target.write_bytes(response.content)
    return target


def _fetch_kaggle(slug: str, names: list[str], data_home, force: bool) -> list[Path]:
    """Download a Kaggle dataset with kagglehub and copy the named files into data_home."""
    home = _home(data_home)
    targets = [home / name for name in names]
    if not force and all(t.exists() for t in targets):
        return targets
    import kagglehub

    folder = kagglehub.dataset_download(slug)
    found = {
        os.path.basename(p): p
        for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
    }
    for name, target in zip(names, targets):
        if name not in found:
            raise FileNotFoundError(
                f"{name} not in Kaggle dataset {slug}; files: {sorted(found)}"
            )
        shutil.copy(found[name], target)
    return targets


# --- Kaggle ---------------------------------------------------------------------------------

FASHION_SLUG = "zalando-research/fashionmnist"
UBER_SLUG = "fivethirtyeight/uber-pickups-in-new-york-city"

FASHION_FILES = ["fashion-mnist_train.csv", "fashion-mnist_test.csv"]
UBER_FILE = "uber-raw-data-apr14.csv"


def download_fashion_mnist(data_home=None, force=False) -> list[Path]:
    return _fetch_kaggle(FASHION_SLUG, FASHION_FILES, data_home, force)


def download_uber_pickups(data_home=None, force=False) -> Path:
    return _fetch_kaggle(UBER_SLUG, [UBER_FILE], data_home, force)[0]


# --- GitHub ---------------------------------------------------------------------------------

ECOLI_URL = (
    "https://raw.githubusercontent.com/deric/clustering-benchmark/master/"
    "src/main/resources/datasets/real-world/ecoli.arff"
)
ECOLI_FILE = "ecoli.arff"


def download_ecoli(data_home=None, force=False, get=requests.get) -> Path:
    target = _home(data_home) / ECOLI_FILE
    return (
        target
        if target.exists() and not force
        else _fetch_url(ECOLI_URL, target, get=get)
    )


# --- USGS -----------------------------------------------------------------------------------
# The event API refuses queries matching more than 20,000 events (HTTP 400); its message is
# passed on so the query can be narrowed.

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_START = "2023-01-01"
USGS_END = "2024-01-01"
USGS_MIN_MAGNITUDE = 4.5


def usgs_query_params(start, end, min_magnitude):
    """Query for every event with magnitude >= min_magnitude in [start, end), oldest first."""
    return {
        "format": "csv",
        "starttime": start,
        "endtime": end,
        "minmagnitude": min_magnitude,
        "orderby": "time-asc",
    }


def usgs_catalogue_path(
    start=USGS_START, end=USGS_END, min_magnitude=USGS_MIN_MAGNITUDE, data_home=None
) -> Path:
    return _home(data_home) / f"usgs_{start}_{end}_m{min_magnitude}.csv"


def download_earthquakes_2023(
    data_home=None,
    force=False,
    start=USGS_START,
    end=USGS_END,
    min_magnitude=USGS_MIN_MAGNITUDE,
    get=requests.get,
) -> Path:
    target = usgs_catalogue_path(start, end, min_magnitude, data_home)
    if target.exists() and not force:
        return target
    return _fetch_url(
        USGS_URL, target, params=usgs_query_params(start, end, min_magnitude), get=get
    )


# --- data.gov.sg ----------------------------------------------------------------------------
# Two steps: initiate a download, then poll until the API returns a file URL. Without a key the
# API allows few requests and answers code 24 (TOO_MANY_REQUESTS), so those calls are retried.
# Each download is saved as <name>_<YYYY-MM-DD>.csv and the loader reads the latest one, so
# results stay fixed until a new download is made.

DATAGOV_URL = "https://api-open.data.gov.sg/v1/public/api/datasets"
DATAGOV_IDS = {
    "resale": "d_8b84c4ee58e3cfc0ece0d773c8ca6abc",
    "price_index": "d_14f63e595975691e7c24a27ae4c07c79",
}
RATE_LIMITED = 24
RETRY_SECONDS = 12
MAX_ATTEMPTS = 10


def _datagov_data(url, get, sleep, max_attempts):
    """The `data` field of a successful API response, retrying while rate limited."""
    for _ in range(max_attempts):
        body = get(url, timeout=TIMEOUT_SECONDS).json()
        if body.get("code") == 0:
            return body["data"]
        if body.get("code") != RATE_LIMITED:
            raise RuntimeError(
                f"data.gov.sg error for {url}: {body.get('errorMsg') or body}"
            )
        sleep(RETRY_SECONDS)
    raise RuntimeError(
        f"data.gov.sg still rate limited after {max_attempts} attempts: {url}"
    )


def download_datagov(
    name,
    data_home=None,
    today=None,
    get=requests.get,
    sleep=time.sleep,
    max_attempts=MAX_ATTEMPTS,
) -> Path:
    """Save one data.gov.sg dataset as <data_home>/<name>_<today>.csv and return the path."""
    dataset_id = DATAGOV_IDS[name]
    _datagov_data(
        f"{DATAGOV_URL}/{dataset_id}/initiate-download", get, sleep, max_attempts
    )
    for _ in range(max_attempts):
        poll = _datagov_data(
            f"{DATAGOV_URL}/{dataset_id}/poll-download", get, sleep, max_attempts
        )
        if poll.get("url"):
            break
        sleep(RETRY_SECONDS)
    else:
        raise RuntimeError(
            f"Download of {name} not ready after {max_attempts} polls; last status {poll.get('status')}"
        )

    response = get(poll["url"], timeout=TIMEOUT_SECONDS)
    if not response.ok:
        raise RuntimeError(
            f"File download for {name} returned HTTP {response.status_code}"
        )
    target = (
        _home(data_home) / f"{name}_{today or datetime.date.today().isoformat()}.csv"
    )
    target.write_bytes(response.content)
    return target


def latest_datagov(name, data_home=None) -> Path:
    """Most recent saved download of a data.gov.sg dataset."""
    home = Path(data_home) if data_home is not None else DATA_DIR
    files = sorted(home.glob(f"{name}_????-??-??.csv"))
    if not files:
        raise FileNotFoundError(
            f"No {name} download in {home}; run: python -m sc4020.datasets.download_dataset hdb_resale"
        )
    return files[-1]


def download_hdb_resale(data_home=None, force=False, **kwargs) -> list[Path]:
    """Resale prices and the resale price index; keeps the latest saved copies unless `force`."""
    paths = []
    for name in DATAGOV_IDS:
        try:
            if force:
                raise FileNotFoundError
            paths.append(latest_datagov(name, data_home))
        except FileNotFoundError:
            paths.append(download_datagov(name, data_home, **kwargs))
    return paths


DOWNLOADERS = {
    "fashion_mnist": download_fashion_mnist,
    "uber_pickups": download_uber_pickups,
    "ecoli": download_ecoli,
    "earthquakes_2023": download_earthquakes_2023,
    "hdb_resale": download_hdb_resale,
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "names", nargs="+", help=f"'all' or any of: {', '.join(DOWNLOADERS)}"
    )
    parser.add_argument(
        "--force", action="store_true", help="download again even if the file exists"
    )
    args = parser.parse_args(argv)
    names = list(DOWNLOADERS) if args.names == ["all"] else args.names
    unknown = set(names) - set(DOWNLOADERS)
    if unknown:
        parser.error(
            f"unknown datasets {sorted(unknown)}; choose from {list(DOWNLOADERS)}"
        )

    failed = 0
    for name in names:
        try:
            print(f"{name}: {DOWNLOADERS[name](force=args.force)}", flush=True)
        except (RuntimeError, OSError, requests.RequestException) as err:
            print(f"{name}: download failed: {err}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
