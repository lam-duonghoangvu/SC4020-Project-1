"""Dataset loaders for the clustering study.

Three datasets, two application domains:
  EQ    - global earthquake epicentres  (spatial, no ground truth)
  UBER  - NYC Uber pickups              (spatial, no ground truth)
  ECOLI - protein localisation sites    (non-spatial, ground truth available)
"""
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff

R_EARTH_KM = 6371.0088
DATA = __file__.rsplit("/", 2)[0] + "/data"
SEED = 42


def _project_km(lat, lon):
    """Equirectangular projection to local kilometres.

    Distances are approximately Euclidean within a region, so that a
    centroid-based method and a density-based method can be compared on
    the same distance scale. Reference latitude is the data mean.
    """
    lat0 = np.radians(np.mean(lat))
    x = R_EARTH_KM * np.radians(lon) * np.cos(lat0)
    y = R_EARTH_KM * np.radians(lat)
    return np.column_stack([x, y])


def load_earthquakes(min_mag=5.5):
    df = pd.read_csv(f"{DATA}/earthquakes.csv")
    df = df[df["Magnitude"] >= min_mag].dropna(subset=["Latitude", "Longitude"])
    lat = df["Latitude"].to_numpy()
    lon = df["Longitude"].to_numpy()
    return {
        "name": "Earthquakes",
        "X": _project_km(lat, lon),          # km, for KMeans / metrics
        "X_deg": np.column_stack([lat, lon]),  # raw degrees, for the metric ablation
        "X_rad": np.radians(np.column_stack([lat, lon])),  # for haversine DBSCAN
        "y": None,
        "spatial": True,
        "n": len(df),
        "d": 2,
    }


def load_uber(n_sample=6000, seed=SEED):
    df = pd.read_csv(f"{DATA}/uber_apr14.csv")
    df.columns = [c.strip().strip('"') for c in df.columns]
    df = df.rename(columns={"Lat": "lat", "Lon": "lon"})
    # NYC bounding box - drops a small number of GPS artefacts
    df = df[(df.lat.between(40.50, 40.95)) & (df.lon.between(-74.10, -73.70))]
    df = df.sample(n=min(n_sample, len(df)), random_state=seed)
    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    return {
        "name": "UberNYC",
        "X": _project_km(lat, lon),
        "X_deg": np.column_stack([lat, lon]),
        "X_rad": np.radians(np.column_stack([lat, lon])),
        "y": None,
        "spatial": True,
        "n": len(df),
        "d": 2,
    }


def load_ecoli(scale=True):
    raw, _ = loadarff(f"{DATA}/ecoli.arff")
    df = pd.DataFrame(raw)
    ycol = df.columns[-1]
    y = df[ycol].astype(str).astype("category").cat.codes.to_numpy()
    X = df.drop(columns=[ycol]).to_numpy(dtype=float)
    Xr = X.copy()
    if scale:
        X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return {
        "name": "Ecoli",
        "X": X,
        "X_raw": Xr,
        "y": y,
        "spatial": False,
        "n": X.shape[0],
        "d": X.shape[1],
    }


def load_all():
    return [load_earthquakes(), load_uber(), load_ecoli()]


if __name__ == "__main__":
    for ds in load_all():
        k = len(np.unique(ds["y"])) if ds["y"] is not None else "-"
        print(f"{ds['name']:12s} n={ds['n']:6d} d={ds['d']}  classes={k}")
