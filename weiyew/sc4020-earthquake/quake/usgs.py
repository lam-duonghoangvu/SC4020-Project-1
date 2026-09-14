"""Download earthquake events from the USGS catalogue into data/raw/.

Usage:
    python -m quake.usgs
    python -m quake.usgs --start 2023-01-01 --end 2024-01-01 --min-magnitude 4.5

No API key is needed. The USGS event API refuses queries matching more than 20,000 events
(HTTP 400); the error message from the API is passed on so the query can be narrowed.
"""

import argparse
import sys
import warnings

import pandas as pd
import requests

from quake.paths import RAW_DIR

BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1"
TIMEOUT_SECONDS = 120

DEFAULT_START = "2023-01-01"
DEFAULT_END = "2024-01-01"
DEFAULT_MIN_MAGNITUDE = 4.5

REQUIRED_COLUMNS = ["time", "latitude", "longitude", "depth", "mag", "type", "place", "id"]


def query_params(start, end, min_magnitude):
    """Query for every event with magnitude >= min_magnitude in [start, end), oldest first."""
    return {
        "format": "csv",
        "starttime": start,
        "endtime": end,
        "minmagnitude": min_magnitude,
        "orderby": "time-asc",
    }


def catalogue_path(start=DEFAULT_START, end=DEFAULT_END, min_magnitude=DEFAULT_MIN_MAGNITUDE, raw_dir=RAW_DIR):
    return raw_dir / f"usgs_{start}_{end}_m{min_magnitude}.csv"


def download_events(start, end, min_magnitude, raw_dir=RAW_DIR, get=requests.get):
    """Save the USGS catalogue for one query as CSV and return the file path."""
    response = get(f"{BASE_URL}/query", params=query_params(start, end, min_magnitude), timeout=TIMEOUT_SECONDS)
    if not response.ok:
        raise RuntimeError(f"USGS returned HTTP {response.status_code}: {response.text.strip()[:500]}")

    target = catalogue_path(start, end, min_magnitude, raw_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(response.text)
    return target


def load_earthquakes(path=None):
    """Load a downloaded catalogue, keeping only events whose type is 'earthquake'.

    The catalogue also lists other seismic events (e.g. volcanic eruptions, explosions);
    they are dropped with a warning that names how many of each type.
    """
    events = pd.read_csv(path or catalogue_path())
    missing = set(REQUIRED_COLUMNS) - set(events.columns)
    if missing:
        raise ValueError(f"Earthquake catalogue is missing columns {sorted(missing)}; got {list(events.columns)}")

    out_of_range = ~events["latitude"].between(-90, 90) | ~events["longitude"].between(-180, 180)
    if out_of_range.any():
        raise ValueError(f"{int(out_of_range.sum())} events have coordinates outside the valid range")

    other = events["type"] != "earthquake"
    if other.any():
        warnings.warn(
            f"Dropped {int(other.sum())} events that are not earthquakes: "
            f"{events.loc[other, 'type'].value_counts().to_dict()}",
            stacklevel=2,
        )
    return events[~other].reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", default=DEFAULT_END, help="End date YYYY-MM-DD (exclusive)")
    parser.add_argument("--min-magnitude", type=float, default=DEFAULT_MIN_MAGNITUDE)
    args = parser.parse_args(argv)

    try:
        target = download_events(args.start, args.end, args.min_magnitude)
    except (RuntimeError, requests.RequestException) as err:
        print(f"Download failed: {err}", file=sys.stderr)
        return 1
    print(f"Saved {len(pd.read_csv(target))} events to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
