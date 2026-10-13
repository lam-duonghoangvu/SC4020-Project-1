"""Turn raw tables into the matrices the clustering methods run on.

Pure functions: no file access (load_dataset.py reads the files) and no global state.

Coordinates. Latitude and longitude in degrees are not a flat space: longitude wraps at +/-180
and a degree of longitude shrinks towards the poles. Three representations are used:
  unit vectors  points on the unit sphere (x, y, z). Euclidean distance between two unit vectors
                is the chord length, which grows with great-circle distance, so every method can
                use it, worldwide. Used for the global earthquake catalogues.
  km            equirectangular projection around the mean latitude. Accurate over a city, so it
                is used for Uber, where it keeps eps in kilometres.
  radians       [latitude, longitude] in radians, for methods run with metric="haversine".
"""

import re
import warnings

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EARTH_RADIUS_KM = 6371.0088  # IUGG mean Earth radius


# --- general ----------------------------------------------------------------------------------


def standardize(X) -> np.ndarray:
    """Scale each column to mean 0 and standard deviation 1 (constant columns are left at 0)."""
    return StandardScaler().fit_transform(np.asarray(X, dtype=float))


# --- coordinates ------------------------------------------------------------------------------


def to_radians(lat, lon) -> np.ndarray:
    return np.radians(
        np.column_stack([np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)])
    )


def to_unit_vectors(lat, lon) -> np.ndarray:
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    return np.column_stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )


def project_km(lat, lon) -> np.ndarray:
    """Equirectangular projection to km around the mean latitude: (x east, y north)."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0 = np.radians(lat.mean())
    return np.column_stack(
        [
            EARTH_RADIUS_KM * np.radians(lon) * np.cos(lat0),
            EARTH_RADIUS_KM * np.radians(lat),
        ]
    )


# "82 km WNW of Hihifo, Tonga" and "south of the Fiji Islands" start with a position relative to a place.
_RELATIVE_POSITION = (
    r"^(?:\d+(?:\.\d+)? km [NSEW]{1,3} of |(?:north|south|east|west)\w* of )"
)


def region_from_place(place: pd.Series) -> pd.Series:
    """Region name from USGS place strings: the text after the last comma, without a leading position.

    "82 km WNW of Hihifo, Tonga" -> "Tonga"; "south of the Fiji Islands" -> "the Fiji Islands";
    "Gulf of Alaska" stays "Gulf of Alaska".
    """
    return place.str.split(", ").str[-1].str.replace(_RELATIVE_POSITION, "", regex=True)


# --- Fashion-MNIST ----------------------------------------------------------------------------


def fashion_mnist_sample(
    full: pd.DataFrame, n_samples: int = 3000, n_pca: int = 50, random_state: int = 42
):
    """Stratified `n_samples` images from train+test, pixels / 255, then PCA to `n_pca` components.

    Returns (X_pca, y, pixels, pca)."""
    y_full = full["label"].to_numpy()
    pixels_full = full.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0
    _, pixels, _, y = train_test_split(
        pixels_full,
        y_full,
        test_size=n_samples / len(full),
        stratify=y_full,
        random_state=random_state,
    )
    pca = PCA(n_components=n_pca, random_state=random_state)
    return pca.fit_transform(pixels).astype(float), y, pixels, pca


# --- Uber -------------------------------------------------------------------------------------

# Bounding box that keeps only pickups inside New York City.
NYC_BOUNDS = {"lat": (40.57, 40.92), "lon": (-74.15, -73.70)}


def nyc_pickups(raw: pd.DataFrame) -> pd.DataFrame:
    """Pickups inside NYC_BOUNDS with both coordinates, columns stripped of spaces and quotes."""
    df = raw.rename(columns=lambda c: c.strip().strip('"'))
    (lat_lo, lat_hi), (lon_lo, lon_hi) = NYC_BOUNDS["lat"], NYC_BOUNDS["lon"]
    inside = df["Lat"].between(lat_lo, lat_hi) & df["Lon"].between(lon_lo, lon_hi)
    return df[inside].dropna(subset=["Lat", "Lon"])


# --- earthquakes ------------------------------------------------------------------------------

USGS_COLUMNS = ["time", "latitude", "longitude", "depth", "mag", "type", "place", "id"]


def usgs_earthquakes(events: pd.DataFrame) -> pd.DataFrame:
    """Check a USGS catalogue and keep only events whose type is 'earthquake'.

    The catalogue also lists other seismic events (e.g. volcanic eruptions, explosions); they are
    dropped with a warning that names how many of each type."""
    missing = set(USGS_COLUMNS) - set(events.columns)
    if missing:
        raise ValueError(
            f"Earthquake catalogue is missing columns {sorted(missing)}; got {list(events.columns)}"
        )
    out_of_range = ~events["latitude"].between(-90, 90) | ~events["longitude"].between(
        -180, 180
    )
    if out_of_range.any():
        raise ValueError(
            f"{int(out_of_range.sum())} events have coordinates outside the valid range"
        )
    other = events["type"] != "earthquake"
    if other.any():
        warnings.warn(
            f"Dropped {int(other.sum())} events that are not earthquakes: "
            f"{events.loc[other, 'type'].value_counts().to_dict()}",
            stacklevel=2,
        )
    return events[~other].reset_index(drop=True)


# --- Ecoli ------------------------------------------------------------------------------------


def ecoli_table(path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Features, integer site codes (alphabetical) and site names from the Ecoli ARFF file."""
    raw, _ = loadarff(path)
    df = pd.DataFrame(raw)
    label = df.columns[-1]
    sites = df[label].astype(str).astype("category")
    return (
        df.drop(columns=[label]).astype(float),
        sites.cat.codes.to_numpy(),
        list(sites.cat.categories),
    )


# --- HDB resale -------------------------------------------------------------------------------

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
HDB_FEATURES = ["floor_area_sqm", "remaining_lease_years", "storey_mid", "real_price"]
TIE_WIDTHS = {
    "floor_area_sqm": 0.5,
    "remaining_lease_years": 0.5 / 12,
    "storey_mid": 1.5,
}

_LEASE = re.compile(r"^(\d+) years?(?: (\d+) months?)?$")
_STOREY = re.compile(r"^(\d+) TO (\d+)$")


def check_resale(resale: pd.DataFrame) -> pd.DataFrame:
    missing = set(RESALE_COLUMNS) - set(resale.columns)
    if missing:
        raise ValueError(
            f"Resale file is missing columns {sorted(missing)}; got {list(resale.columns)}"
        )
    return resale


def price_index_series(index: pd.DataFrame) -> pd.Series:
    """Price index as a Series indexed by quarter label, e.g. '2017-Q1'."""
    missing = {"quarter", "index"} - set(index.columns)
    if missing:
        raise ValueError(
            f"Price index file is missing columns {sorted(missing)}; got {list(index.columns)}"
        )
    return index.set_index("quarter")["index"].astype(float)


def remaining_lease_years(text) -> float:
    """'61 years 04 months' -> 61.333; '61 years' -> 61.0."""
    match = _LEASE.match(str(text).strip())
    if not match:
        raise ValueError(f"Unrecognised remaining lease: {text!r}")
    return int(match.group(1)) + int(match.group(2) or 0) / 12


def storey_midpoint(text) -> float:
    """'10 TO 12' -> 11.0."""
    match = _STOREY.match(str(text).strip())
    if not match:
        raise ValueError(f"Unrecognised storey range: {text!r}")
    return (int(match.group(1)) + int(match.group(2))) / 2


def sale_quarter(month) -> str:
    """'2017-05' -> '2017-Q2'."""
    year, month_number = str(month).split("-")
    return f"{year}-Q{(int(month_number) - 1) // 3 + 1}"


def real_prices(resale: pd.DataFrame, price_index: pd.Series) -> pd.Series:
    """Resale price in the price level of the latest index quarter.

    price * (latest index / index of the sale quarter). Sales in quarters without an index value
    (the index is published after the quarter ends) are NaN."""
    quarter_index = resale["month"].map(sale_quarter).map(price_index)
    return resale["resale_price"] * price_index.iloc[-1] / quarter_index


def hdb_features(resale: pd.DataFrame, price_index: pd.Series) -> pd.DataFrame:
    """One row per sale with HDB_FEATURES; drops sales without an index value, with a warning."""
    features = pd.DataFrame(
        {
            "floor_area_sqm": resale["floor_area_sqm"].astype(float),
            "remaining_lease_years": resale["remaining_lease"].map(
                remaining_lease_years
            ),
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


def jitter_ties(features: pd.DataFrame, rng, widths=None) -> pd.DataFrame:
    """Spread each value uniformly within ± its width, so sales with identical values no longer coincide.

    The default widths are half the step between neighbouring values: storey midpoints are 3 apart,
    floor area is recorded in whole m², remaining lease in whole months. Price is continuous and is
    not changed. Without this, density methods treat repeated values as dense points and find one
    cluster per storey band."""
    widths = TIE_WIDTHS if widths is None else widths
    scale = np.array([widths.get(column, 0.0) for column in features.columns])
    return features + rng.uniform(-1, 1, size=features.shape) * scale
