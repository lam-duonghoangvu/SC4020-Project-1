"""Coordinate features for clustering earthquake epicentres.

Latitude and longitude in degrees are not a flat space: longitude wraps at +/-180 and a degree of
longitude shrinks towards the poles. Two representations avoid both problems:
  radians: [latitude, longitude] in radians, for methods that accept metric="haversine".
  unit vectors: points on the unit sphere (x, y, z). Euclidean distance between two unit vectors is
      the chord length, which increases with great-circle distance, so K-means can use them.
"""

import numpy as np
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0


def to_radians(lat, lon):
    return np.radians(np.column_stack([np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)]))


def to_unit_vectors(lat, lon):
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between points given in degrees."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def nearest_neighbour_km(lat, lon, k=1):
    """Great-circle distance in km from each point to its k-th nearest other point."""
    X = to_radians(lat, lon)
    distances, _ = BallTree(X, metric="haversine").query(X, k=k + 1)
    return distances[:, k] * EARTH_RADIUS_KM
