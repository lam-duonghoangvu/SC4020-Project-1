import numpy as np
import pandas as pd
import pytest

from sc4020.datasets.preprocess_dataset import (
    EARTH_RADIUS_KM,
    HDB_FEATURES,
    check_resale,
    chord_to_km,
    hdb_features,
    jitter_ties,
    km_to_chord,
    nyc_pickups,
    price_index_series,
    project_km,
    real_prices,
    region_from_place,
    remaining_lease_years,
    sale_quarter,
    standardize,
    storey_midpoint,
    to_radians,
    to_unit_vectors,
    usgs_earthquakes,
    varied_density_blobs,
)

INDEX = pd.Series({"2017-Q1": 100.0, "2017-Q2": 125.0, "2017-Q3": 200.0})


def resale_rows():
    return pd.DataFrame(
        {
            "month": ["2017-01", "2017-05", "2017-09", "2017-10"],
            "town": ["ANG MO KIO"] * 4,
            "flat_type": ["3 ROOM", "4 ROOM", "5 ROOM", "4 ROOM"],
            "block": ["406", "108", "602", "465"],
            "street_name": ["ANG MO KIO AVE 10"] * 4,
            "storey_range": ["10 TO 12", "01 TO 03", "04 TO 06", "07 TO 09"],
            "floor_area_sqm": [67.0, 92.0, 110.0, 91.0],
            "flat_model": ["New Generation"] * 4,
            "lease_commence_date": [1978, 1980, 1990, 1985],
            "remaining_lease": [
                "61 years 04 months",
                "62 years",
                "72 years 01 month",
                "67 years 11 months",
            ],
            "resale_price": [300000.0, 400000.0, 500000.0, 450000.0],
        }
    )


# --- HDB --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, years",
    [
        ("61 years 04 months", 61 + 4 / 12),
        ("62 years", 62.0),
        ("72 years 01 month", 72 + 1 / 12),
    ],
)
def test_remaining_lease_years(text, years):
    assert remaining_lease_years(text) == pytest.approx(years)


def test_remaining_lease_rejects_other_formats():
    with pytest.raises(ValueError, match="Unrecognised remaining lease"):
        remaining_lease_years("61")


def test_storey_midpoint():
    assert storey_midpoint("10 TO 12") == 11.0
    assert storey_midpoint("01 TO 05") == 3.0
    with pytest.raises(ValueError, match="Unrecognised storey range"):
        storey_midpoint("10-12")


@pytest.mark.parametrize(
    "month, quarter",
    [
        ("2017-01", "2017-Q1"),
        ("2017-03", "2017-Q1"),
        ("2017-04", "2017-Q2"),
        ("2017-12", "2017-Q4"),
    ],
)
def test_sale_quarter(month, quarter):
    assert sale_quarter(month) == quarter


def test_real_prices_use_latest_index_quarter_as_price_level():
    prices = real_prices(resale_rows(), INDEX)

    # 300,000 at index 100 and 400,000 at index 125 in the price level of index 200.
    assert prices.iloc[:3].tolist() == [600000.0, 640000.0, 500000.0]
    assert pd.isna(prices.iloc[3])


def test_hdb_features_drop_sales_without_index_with_warning():
    with pytest.warns(
        UserWarning,
        match=r"Dropped 1 sales without a price index value \(months 2017-10 to 2017-10\)",
    ):
        features = hdb_features(resale_rows(), INDEX)

    assert list(features.columns) == HDB_FEATURES
    assert features.index.tolist() == [0, 1, 2]
    assert features.loc[0].tolist() == pytest.approx(
        [67.0, 61 + 4 / 12, 11.0, 600000.0]
    )


def test_checks_reject_missing_columns():
    bad = pd.DataFrame({"month": ["2017-01"]})

    with pytest.raises(ValueError, match="Resale file is missing columns"):
        check_resale(bad)
    with pytest.raises(ValueError, match="Price index file is missing columns"):
        price_index_series(bad)


def test_price_index_is_indexed_by_quarter():
    index = pd.DataFrame({"quarter": ["1990-Q1", "1990-Q2"], "index": [24.3, 24.4]})

    assert price_index_series(index).to_dict() == {"1990-Q1": 24.3, "1990-Q2": 24.4}


def test_jitter_ties_separates_identical_rows_within_half_step():
    features = pd.DataFrame(
        {
            "floor_area_sqm": [93.0] * 200,
            "remaining_lease_years": [94.75] * 200,
            "storey_mid": [5.0] * 200,
            "real_price": [600000.0] * 200,
        }
    )

    jittered = jitter_ties(features, np.random.default_rng(0))

    assert not jittered.duplicated().any()
    assert (jittered["storey_mid"] - 5.0).abs().max() <= 1.5
    assert (jittered["floor_area_sqm"] - 93.0).abs().max() <= 0.5
    assert (jittered["remaining_lease_years"] - 94.75).abs().max() <= 0.5 / 12
    assert (jittered["real_price"] == 600000.0).all()
    assert (features["storey_mid"] == 5.0).all()


def test_standardize_gives_mean_zero_and_unit_std():
    X = standardize([[1.0, 100.0], [2.0, 300.0], [3.0, 500.0]])

    assert X.mean(axis=0) == pytest.approx([0.0, 0.0])
    assert X.std(axis=0) == pytest.approx([1.0, 1.0])


# --- coordinates ------------------------------------------------------------------------------


def test_unit_vectors_lie_on_sphere_and_meet_across_180_degree_line():
    vectors = to_unit_vectors([0, 45, -90, 0], [179.9, 10, 0, -179.9])

    assert np.linalg.norm(vectors, axis=1) == pytest.approx(np.ones(4))
    assert np.linalg.norm(vectors[0] - vectors[3]) == pytest.approx(
        np.radians(0.2), rel=1e-4
    )


def test_to_radians_keeps_latitude_first():
    assert to_radians([90], [180]) == pytest.approx(np.array([[np.pi / 2, np.pi]]))


def test_project_km_one_degree_of_latitude():
    xy = project_km([0.0, 1.0], [10.0, 10.0])

    assert xy[1, 1] - xy[0, 1] == pytest.approx(np.radians(1) * EARTH_RADIUS_KM)
    assert xy[0, 0] == pytest.approx(xy[1, 0])


def test_chord_and_km_round_trip():
    km = np.array([0.0, 10.0, 1000.0, 5000.0])

    assert chord_to_km(km_to_chord(km)) == pytest.approx(km)
    # the chord between two unit vectors equals the great-circle distance's chord
    v = to_unit_vectors([0, 0], [0, 10])
    assert chord_to_km(np.linalg.norm(v[0] - v[1])) == pytest.approx(
        np.radians(10) * EARTH_RADIUS_KM
    )


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


# --- earthquakes, Uber, synthetic -----------------------------------------------------------------

USGS = pd.DataFrame(
    {
        "time": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "latitude": [12.3, -24.3, 19.4],
        "longitude": [143.3, 179.8, -155.3],
        "depth": [30.1, 516.5, 1.0],
        "mag": [4.6, 4.7, 4.5],
        "type": ["earthquake", "earthquake", "volcanic eruption"],
        "place": [
            "180 km SW of Merizo Village, Guam",
            "south of the Fiji Islands",
            "Hawaii",
        ],
        "id": ["a", "b", "c"],
    }
)


def test_usgs_keeps_only_earthquakes_and_warns_with_counts():
    with pytest.warns(UserWarning, match=r"Dropped 1 events .*'volcanic eruption': 1"):
        events = usgs_earthquakes(USGS)

    assert list(events["id"]) == ["a", "b"]


def test_usgs_rejects_missing_columns_and_bad_coordinates():
    with pytest.raises(ValueError, match="missing columns"):
        usgs_earthquakes(USGS[["latitude", "longitude"]])
    with pytest.raises(ValueError, match="1 events have coordinates outside"):
        usgs_earthquakes(USGS.assign(latitude=[95.0, 0.0, 0.0]))


def test_nyc_pickups_strip_column_names_and_keep_the_city_box():
    raw = pd.DataFrame(
        {'"Lat"': [40.75, 41.5, 40.70, np.nan], "Lon ": [-73.98, -73.98, -80.0, -73.9]}
    )

    kept = nyc_pickups(raw)

    assert list(kept.columns) == ["Lat", "Lon"]
    assert kept["Lat"].tolist() == [40.75]


def test_varied_density_blobs_have_three_clusters_and_noise():
    X, y = varied_density_blobs(n_noise=50, random_state=0)

    assert X.shape == (1250, 2)
    assert sorted(set(y.tolist())) == [-1, 0, 1, 2]
    assert (y == -1).sum() == 50
