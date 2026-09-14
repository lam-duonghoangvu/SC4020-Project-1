"""Download HDB datasets from data.gov.sg into data/raw/, one dated CSV per download.

Usage:
    python -m hdb.datagov            # both datasets
    python -m hdb.datagov resale     # one dataset

No API key is needed. The download API works in two steps: initiate a download, then poll until
it returns a file URL. Without a key the API allows few requests and answers code 24
(TOO_MANY_REQUESTS), so rate-limited calls are retried after a pause.

The resale dataset is updated monthly. Each download is saved as <name>_<YYYY-MM-DD>.csv and the
analysis reads the latest saved file, so results do not change unless a new download is made.
"""

import argparse
import datetime
import sys
import time

import requests

from hdb.paths import RAW_DIR

API_URL = "https://api-open.data.gov.sg/v1/public/api/datasets"
DATASETS = {
    "resale": "d_8b84c4ee58e3cfc0ece0d773c8ca6abc",
    "price_index": "d_14f63e595975691e7c24a27ae4c07c79",
}
RATE_LIMITED = 24
RETRY_SECONDS = 12
MAX_ATTEMPTS = 10
TIMEOUT_SECONDS = 120


def _api_data(url, get, sleep, max_attempts):
    """Return the `data` field of a successful API response, retrying while rate limited."""
    for _ in range(max_attempts):
        body = get(url, timeout=TIMEOUT_SECONDS).json()
        if body.get("code") == 0:
            return body["data"]
        if body.get("code") != RATE_LIMITED:
            raise RuntimeError(f"data.gov.sg error for {url}: {body.get('errorMsg') or body}")
        sleep(RETRY_SECONDS)
    raise RuntimeError(f"data.gov.sg still rate limited after {max_attempts} attempts: {url}")


def download_dataset(name, raw_dir=RAW_DIR, today=None, get=requests.get, sleep=time.sleep, max_attempts=MAX_ATTEMPTS):
    """Save one dataset as raw_dir/<name>_<today>.csv and return the path."""
    dataset_id = DATASETS[name]
    _api_data(f"{API_URL}/{dataset_id}/initiate-download", get, sleep, max_attempts)

    for _ in range(max_attempts):
        poll = _api_data(f"{API_URL}/{dataset_id}/poll-download", get, sleep, max_attempts)
        if poll.get("url"):
            break
        sleep(RETRY_SECONDS)
    else:
        raise RuntimeError(f"Download of {name} not ready after {max_attempts} polls; last status {poll.get('status')}")

    response = get(poll["url"], timeout=TIMEOUT_SECONDS)
    if not response.ok:
        raise RuntimeError(f"File download for {name} returned HTTP {response.status_code}")

    target = raw_dir / f"{name}_{today or datetime.date.today().isoformat()}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    return target


def latest_download(name, raw_dir=RAW_DIR):
    """Path of the most recent saved download of a dataset."""
    files = sorted(raw_dir.glob(f"{name}_????-??-??.csv"))
    if not files:
        raise FileNotFoundError(f"No {name} download in {raw_dir}; run: python -m hdb.datagov {name}")
    return files[-1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help=f"Datasets to download: {', '.join(DATASETS)} (default: all)")
    args = parser.parse_args(argv)
    unknown = set(args.names) - set(DATASETS)
    if unknown:
        parser.error(f"unknown datasets {sorted(unknown)}; choose from {list(DATASETS)}")

    for name in args.names or DATASETS:
        try:
            target = download_dataset(name)
        except (RuntimeError, requests.RequestException) as err:
            print(f"Download failed: {err}", file=sys.stderr)
            return 1
        print(f"Saved {name} to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
