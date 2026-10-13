import numpy as np
import pandas as pd
import pytest

from sc4020.datasets import (
    DATASETS,
    load_breast_cancer,
    load_dataset,
    load_digits,
    load_earthquakes_2023,
    load_hdb_resale,
    load_mall_customers,
    load_wine,
    make_moons,
    make_varied_density,
    n_true_clusters,
)


def test_builtin_loaders_return_a_bunch_like_scikit_learn():
    wine = load_wine()

    assert wine.name == "wine"
    assert wine.data.shape == (178, 13)
    assert wine.n_true_clusters == 3
    assert len(wine.feature_names) == 13 and len(wine.target_names) == 3
    assert wine.DESCR
    assert wine.data.mean(axis=0) == pytest.approx(np.zeros(13), abs=1e-9)


def test_return_X_y_and_as_frame():
    X, y = load_wine(return_X_y=True)
    frame = load_wine(as_frame=True).frame

    assert X.shape == (178, 13) and y.shape == (178,)
    assert isinstance(frame, pd.DataFrame) and frame.shape == (178, 14)


def test_breast_cancer_codes_malignant_as_one():
    bc = load_breast_cancer()

    assert bc.target_names == ["benign", "malignant"]
    assert int(bc.target.sum()) == 212  # malignant tumours in the UCI table


def test_digits_keep_pixel_scale_and_moons_are_standardized():
    assert load_digits().data.max() == 16.0
    moons = make_moons()
    assert moons.data.shape == (1000, 2) and moons.data.std(axis=0) == pytest.approx([1.0, 1.0])


def test_varied_density_counts_classes_without_noise():
    ds = make_varied_density()

    assert ds.n_true_clusters == 3
    assert n_true_clusters(ds.target) == 3
    assert n_true_clusters(None) is None


def test_missing_file_without_download_names_the_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_dataset mall_customers"):
        load_mall_customers(data_home=tmp_path, download_if_missing=False)


def test_load_dataset_by_name_and_unknown_name():
    assert load_dataset("wine").name == "wine"
    assert list(DATASETS)[:3] == ["wine", "breast_cancer", "uber_pickups"]
    with pytest.raises(KeyError, match="unknown dataset"):
        load_dataset("iris")


def test_earthquakes_2023_from_a_saved_catalogue(tmp_path):
    (tmp_path / "usgs_2023-01-01_2024-01-01_m4.5.csv").write_text(
        "time,latitude,longitude,depth,mag,type,place,id\n"
        "2023-01-01,0.0,179.9,10,4.6,earthquake,A,a\n"
        "2023-01-02,0.0,-179.9,10,4.7,earthquake,B,b\n"
        "2023-01-03,10.0,0.0,10,4.5,explosion,C,c\n"
    )

    with pytest.warns(UserWarning, match="Dropped 1 events"):
        quakes = load_earthquakes_2023(data_home=tmp_path, download_if_missing=False)

    assert quakes.data.shape == (2, 3)
    assert quakes.coords_deg.tolist() == [[0.0, 179.9], [0.0, -179.9]]
    assert np.linalg.norm(quakes.data[0] - quakes.data[1]) < 0.01  # neighbours across the 180 degree line


def test_hdb_resale_from_saved_downloads(tmp_path):
    rows = 40
    pd.DataFrame(
        {
            "month": ["2017-01", "2017-05"] * (rows // 2),
            "town": ["ANG MO KIO"] * rows,
            "flat_type": ["4 ROOM"] * rows,
            "block": ["1"] * rows,
            "street_name": ["X"] * rows,
            "storey_range": ["01 TO 03", "10 TO 12"] * (rows // 2),
            "floor_area_sqm": np.linspace(60, 120, rows),
            "flat_model": ["Model A"] * rows,
            "lease_commence_date": [1990] * rows,
            "remaining_lease": ["70 years"] * rows,
            "resale_price": np.linspace(3e5, 6e5, rows),
        }
    ).to_csv(tmp_path / "resale_2026-01-01.csv", index=False)
    (tmp_path / "price_index_2026-01-01.csv").write_text("quarter,index\n2017-Q1,100\n2017-Q2,125\n")

    hdb = load_hdb_resale(data_home=tmp_path, download_if_missing=False, n_samples=30)
    everything = load_hdb_resale(data_home=tmp_path, download_if_missing=False, n_samples=None, jitter=False, scale=False)

    assert hdb.data.shape == (30, 4)
    assert hdb.data.mean(axis=0) == pytest.approx(np.zeros(4), abs=1e-9)
    assert len(hdb.sales) == 30 and (hdb.sales.index == hdb.features_raw.index).all()
    assert everything.data.shape == (40, 4)
    assert everything.data[:, 2].tolist()[:2] == [2.0, 11.0]  # storey midpoints, unjittered
