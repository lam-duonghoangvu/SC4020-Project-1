import math

import pandas as pd
import pytest

from quake.eda import grid_cell_counts, magnitude_threshold_table, region_from_place


def test_region_strips_distance_and_direction_but_keeps_names_with_of():
    places = pd.Series(
        [
            "82 km WNW of Hihifo, Tonga",
            "south of the Fiji Islands",
            "southeast of the Loyalty Islands",
            "Pazarcik earthquake, Kahramanmaras earthquake sequence",
            "Gulf of Alaska",
            "7.5 km S of Town, Japan region",
        ]
    )

    assert list(region_from_place(places)) == [
        "Tonga",
        "the Fiji Islands",
        "the Loyalty Islands",
        "Kahramanmaras earthquake sequence",
        "Gulf of Alaska",
        "Japan region",
    ]


def test_threshold_table_counts_events_at_or_above_each_magnitude():
    events = pd.DataFrame({"mag": [4.5, 5.0, 5.5, 6.0], "latitude": [0, 0, 0, 0], "longitude": [0, 1, 3, 6]})

    table = magnitude_threshold_table(events)

    assert list(table["events"]) == [4, 3, 2, 1]
    assert list(table["% kept"]) == pytest.approx([100, 75, 50, 25])
    # Gaps to the nearest point are 1, 1, 2 and 3 degrees of longitude on the equator.
    assert table["median nearest neighbour km"].iloc[0] == pytest.approx(111.195 * 1.5, rel=1e-4)
    assert math.isnan(table["median nearest neighbour km"].iloc[-1])


def test_grid_cell_counts_one_column_per_period_with_zero_fill():
    counts = grid_cell_counts(lat=[1, 2, 7, 1], lon=[1, 4, 1, -1], period=["H1", "H2", "H2", "H1"])

    assert list(counts.columns) == ["H1", "H2"]
    assert counts.loc[(0.0, 0.0)].tolist() == [1, 1]
    assert counts.loc[(5.0, 0.0)].tolist() == [0, 1]
    assert counts.loc[(0.0, -5.0)].tolist() == [1, 0]
