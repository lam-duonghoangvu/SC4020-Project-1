import numpy as np
import pandas as pd
import pytest

from hdb.features import (
    SCREENING_FEATURES,
    jitter_ties,
    load_price_index,
    load_resale,
    real_prices,
    remaining_lease_years,
    sale_quarter,
    screening_features,
    standardize,
    storey_midpoint,
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
            "remaining_lease": ["61 years 04 months", "62 years", "72 years 01 month", "67 years 11 months"],
            "resale_price": [300000.0, 400000.0, 500000.0, 450000.0],
        }
    )


@pytest.mark.parametrize(
    "text, years",
    [("61 years 04 months", 61 + 4 / 12), ("62 years", 62.0), ("72 years 01 month", 72 + 1 / 12)],
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


@pytest.mark.parametrize("month, quarter", [("2017-01", "2017-Q1"), ("2017-03", "2017-Q1"), ("2017-04", "2017-Q2"), ("2017-12", "2017-Q4")])
def test_sale_quarter(month, quarter):
    assert sale_quarter(month) == quarter


def test_real_prices_use_latest_index_quarter_as_price_level():
    prices = real_prices(resale_rows(), INDEX)

    # 300,000 at index 100 and 400,000 at index 125 in the price level of index 200.
    assert prices.iloc[:3].tolist() == [600000.0, 640000.0, 500000.0]
    assert pd.isna(prices.iloc[3])


def test_screening_features_drop_sales_without_index_with_warning():
    with pytest.warns(UserWarning, match=r"Dropped 1 sales without a price index value \(months 2017-10 to 2017-10\)"):
        features = screening_features(resale_rows(), INDEX)

    assert list(features.columns) == SCREENING_FEATURES
    assert features.index.tolist() == [0, 1, 2]
    assert features.loc[0].tolist() == pytest.approx([67.0, 61 + 4 / 12, 11.0, 600000.0])


def test_loaders_reject_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"month": ["2017-01"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Resale file is missing columns"):
        load_resale(path)
    with pytest.raises(ValueError, match="Price index file is missing columns"):
        load_price_index(path)


def test_load_price_index_is_indexed_by_quarter(tmp_path):
    path = tmp_path / "price_index.csv"
    path.write_text("quarter,index\n1990-Q1,24.3\n1990-Q2,24.4\n")

    assert load_price_index(path).to_dict() == {"1990-Q1": 24.3, "1990-Q2": 24.4}


def test_jitter_ties_separates_identical_rows_within_half_step():
    features = pd.DataFrame(
        {"floor_area_sqm": [93.0] * 200, "remaining_lease_years": [94.75] * 200, "storey_mid": [5.0] * 200, "real_price": [600000.0] * 200}
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
