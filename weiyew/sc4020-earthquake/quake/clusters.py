"""Describe clusters of earthquakes: where they are, how spread out, and which regions they contain."""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from quake.eda import region_from_place
from quake.features import EARTH_RADIUS_KM, haversine_km, to_radians, to_unit_vectors
from quake.metrics import NOISE_LABEL


def cross_line_pairs(lat, lon, labels, max_km=50):
    """How a clustering treats close neighbours on opposite sides of the 180 degree line.

    Counts pairs of points within `max_km` of each other with one point east of longitude 90 and the
    other west of longitude -90, and how many of those pairs are both clustered and in the same cluster.
    """
    lat, lon, labels = (np.asarray(v) for v in (lat, lon, labels))
    east, west = np.flatnonzero(lon > 90), np.flatnonzero(lon < -90)
    if len(east) == 0 or len(west) == 0:
        return {"cross_line_pairs": 0, "pairs_both_clustered": 0, "pairs_same_cluster": 0}
    tree = BallTree(to_radians(lat[west], lon[west]), metric="haversine")
    neighbours = tree.query_radius(to_radians(lat[east], lon[east]), r=max_km / EARTH_RADIUS_KM)
    pairs = [(e, west[w]) for e, found in zip(east, neighbours) for w in found]
    left = np.array([labels[e] for e, _ in pairs], dtype=int)
    right = np.array([labels[w] for _, w in pairs], dtype=int)
    both = (left != NOISE_LABEL) & (right != NOISE_LABEL)
    return {
        "cross_line_pairs": len(pairs),
        "pairs_both_clustered": int(both.sum()),
        "pairs_same_cluster": int((both & (left == right)).sum()),
    }


def relabel_by_size(labels):
    """Renumber clusters 0, 1, ... from largest to smallest (ties by old label); noise stays NOISE_LABEL."""
    labels = np.asarray(labels)
    clustered = labels[labels != NOISE_LABEL]
    counts = pd.Series(clustered).value_counts()
    order = sorted(counts.index, key=lambda label: (-counts[label], label))
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping.get(label, NOISE_LABEL) for label in labels])


def spherical_centre(lat, lon):
    """Latitude and longitude (degrees) of the normalised mean unit vector: the centre on the sphere."""
    mean = to_unit_vectors(lat, lon).mean(axis=0)
    mean = mean / np.linalg.norm(mean)
    return float(np.degrees(np.arcsin(mean[2]))), float(np.degrees(np.arctan2(mean[1], mean[0])))


def cluster_summary(events, labels, top_regions=3):
    """One row per cluster (noise excluded): size, centre, distance of members to the centre, top regions."""
    labels = np.asarray(labels)
    regions = region_from_place(events["place"]).to_numpy()
    rows = []
    for cluster in sorted(set(labels.tolist()) - {NOISE_LABEL}):
        mask = labels == cluster
        members = events[mask]
        lat, lon = spherical_centre(members["latitude"], members["longitude"])
        to_centre = haversine_km(members["latitude"], members["longitude"], lat, lon)
        common = pd.Series(regions[mask]).value_counts().head(top_regions)
        rows.append(
            {
                "cluster": cluster,
                "events": int(mask.sum()),
                "% of events": 100 * mask.sum() / len(labels),
                "centre latitude": lat,
                "centre longitude": lon,
                "median km to centre": float(np.median(to_centre)),
                "95th percentile km to centre": float(np.percentile(to_centre, 95)),
                "median depth km": float(members["depth"].median()),
                "top regions": ", ".join(f"{region} ({count})" for region, count in common.items()),
            }
        )
    return pd.DataFrame(rows)
