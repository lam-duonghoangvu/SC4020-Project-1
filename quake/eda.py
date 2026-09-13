"""Helpers for notebooks/eda.ipynb."""

import numpy as np
import pandas as pd

from quake.features import nearest_neighbour_km

# "82 km WNW of Hihifo, Tonga" and "south of the Fiji Islands" start with a position relative to a place.
_RELATIVE_POSITION = r"^(?:\d+(?:\.\d+)? km [NSEW]{1,3} of |(?:north|south|east|west)\w* of )"


def region_from_place(place):
    """Region name from USGS place strings: the text after the last comma, without a leading position.

    "82 km WNW of Hihifo, Tonga" -> "Tonga"; "south of the Fiji Islands" -> "the Fiji Islands";
    "Gulf of Alaska" stays "Gulf of Alaska".
    """
    return place.str.split(", ").str[-1].str.replace(_RELATIVE_POSITION, "", regex=True)


def magnitude_threshold_table(events, thresholds=(4.5, 5.0, 5.5, 6.0)):
    """What each minimum magnitude keeps: events, share of events, and median nearest-neighbour distance."""
    rows = []
    for threshold in thresholds:
        kept = events[events["mag"] >= threshold]
        spacing = nearest_neighbour_km(kept["latitude"], kept["longitude"]) if len(kept) > 1 else [np.nan]
        rows.append(
            {
                "min magnitude": threshold,
                "events": len(kept),
                "% kept": 100 * len(kept) / len(events),
                "median nearest neighbour km": float(np.median(spacing)),
            }
        )
    return pd.DataFrame(rows)


def grid_cell_counts(lat, lon, period, cell_degrees=5):
    """Event counts per (latitude cell, longitude cell), one column per period."""
    cells = pd.DataFrame(
        {
            "cell_lat": np.floor(np.asarray(lat, dtype=float) / cell_degrees) * cell_degrees,
            "cell_lon": np.floor(np.asarray(lon, dtype=float) / cell_degrees) * cell_degrees,
            "period": np.asarray(period),
        }
    )
    return cells.groupby(["cell_lat", "cell_lon", "period"]).size().unstack("period", fill_value=0)
