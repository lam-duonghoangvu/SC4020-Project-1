"""Load the resale transactions and price index, and build numeric features per transaction."""

import re
import warnings

import numpy as np
import pandas as pd

from hdb.datagov import latest_download

RESALE_COLUMNS = [
    "month",
    "town",
    "flat_type",
    "block",
    "street_name",
    "storey_range",
    "floor_area_sqm",
    "flat_model",
    "lease_commence_date",
    "remaining_lease",
    "resale_price",
]
SCREENING_FEATURES = ["floor_area_sqm", "remaining_lease_years", "storey_mid", "real_price"]
TIE_WIDTHS = {"floor_area_sqm": 0.5, "remaining_lease_years": 0.5 / 12, "storey_mid": 1.5}

_LEASE = re.compile(r"^(\d+) years?(?: (\d+) months?)?$")
_STOREY = re.compile(r"^(\d+) TO (\d+)$")


def load_resale(path=None):
    resale = pd.read_csv(path or latest_download("resale"))
    missing = set(RESALE_COLUMNS) - set(resale.columns)
    if missing:
        raise ValueError(f"Resale file is missing columns {sorted(missing)}; got {list(resale.columns)}")
    return resale


def load_price_index(path=None):
    """Price index as a Series indexed by quarter label, e.g. '2017-Q1'."""
    index = pd.read_csv(path or latest_download("price_index"))
    missing = {"quarter", "index"} - set(index.columns)
    if missing:
        raise ValueError(f"Price index file is missing columns {sorted(missing)}; got {list(index.columns)}")
    return index.set_index("quarter")["index"].astype(float)


def remaining_lease_years(text):
    """'61 years 04 months' -> 61.333; '61 years' -> 61.0."""
    match = _LEASE.match(str(text).strip())
    if not match:
        raise ValueError(f"Unrecognised remaining lease: {text!r}")
    return int(match.group(1)) + int(match.group(2) or 0) / 12


def storey_midpoint(text):
    """'10 TO 12' -> 11.0."""
    match = _STOREY.match(str(text).strip())
    if not match:
        raise ValueError(f"Unrecognised storey range: {text!r}")
    return (int(match.group(1)) + int(match.group(2))) / 2


def sale_quarter(month):
    """'2017-05' -> '2017-Q2'."""
    year, month_number = str(month).split("-")
    return f"{year}-Q{(int(month_number) - 1) // 3 + 1}"


def real_prices(resale, price_index):
    """Resale price in the price level of the latest index quarter.

    price * (latest index / index of the sale quarter). Sales in quarters without an index value
    (the index is published after the quarter ends) are NaN.
    """
    quarter_index = resale["month"].map(sale_quarter).map(price_index)
    return resale["resale_price"] * price_index.iloc[-1] / quarter_index


def screening_features(resale, price_index):
    """One row per transaction with SCREENING_FEATURES; drops sales without an index value, with a warning."""
    features = pd.DataFrame(
        {
            "floor_area_sqm": resale["floor_area_sqm"].astype(float),
            "remaining_lease_years": resale["remaining_lease"].map(remaining_lease_years),
            "storey_mid": resale["storey_range"].map(storey_midpoint),
            "real_price": real_prices(resale, price_index),
        },
        index=resale.index,
    )
    no_index = features["real_price"].isna()
    if no_index.any():
        months = sorted(resale.loc[no_index, "month"].unique())
        warnings.warn(
            f"Dropped {int(no_index.sum())} sales without a price index value (months {months[0]} to {months[-1]})",
            stacklevel=2,
        )
    return features[~no_index]


def jitter_ties(features, rng, widths=None):
    """Spread each value uniformly within ± its width, so sales with identical values no longer coincide.

    The default widths are half the step between neighbouring values: storey midpoints are 3 apart,
    floor area is recorded in whole m², remaining lease in whole months. Price is continuous and is
    not changed. Used for density methods, which otherwise treat repeated values as dense points
    (notebooks/eda.ipynb, section 10).
    """
    widths = TIE_WIDTHS if widths is None else widths
    scale = np.array([widths.get(column, 0.0) for column in features.columns])
    return features + rng.uniform(-1, 1, size=features.shape) * scale


def standardize(X):
    """Scale each column to mean 0 and standard deviation 1."""
    X = np.asarray(X, dtype=float)
    return (X - X.mean(axis=0)) / X.std(axis=0)
