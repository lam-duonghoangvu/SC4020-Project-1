import pandas as pd
import pytest

from quake.usgs import catalogue_path, download_events, load_earthquakes, query_params

CSV = (
    "time,latitude,longitude,depth,mag,magType,type,place,id\n"
    '2023-01-01T00:00:00Z,12.3,143.3,30.1,4.6,mb,earthquake,"180 km SW of Merizo Village, Guam",a\n'
    "2023-01-02T00:00:00Z,-24.3,179.8,516.5,4.7,mb,earthquake,south of the Fiji Islands,b\n"
    "2023-01-03T00:00:00Z,19.4,-155.3,1.0,4.5,ml,volcanic eruption,Hawaii,c\n"
)


class FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code
        self.ok = status_code < 400


def test_query_params_ask_for_csv_in_time_order():
    params = query_params("2023-01-01", "2024-01-01", 4.5)

    assert params == {
        "format": "csv",
        "starttime": "2023-01-01",
        "endtime": "2024-01-01",
        "minmagnitude": 4.5,
        "orderby": "time-asc",
    }


def test_download_saves_response_under_query_named_file(tmp_path):
    urls = []

    def get(url, params, timeout):
        urls.append(url)
        return FakeResponse(CSV)

    target = download_events("2023-01-01", "2024-01-01", 4.5, raw_dir=tmp_path, get=get)

    assert target == tmp_path / "usgs_2023-01-01_2024-01-01_m4.5.csv"
    assert target.read_text() == CSV
    assert urls == ["https://earthquake.usgs.gov/fdsnws/event/1/query"]


def test_download_over_search_limit_raises_with_api_message(tmp_path):
    def get(url, params, timeout):
        body = "Error 400: Bad Request\n\n166725 matching events exceeds search limit of 20000."
        return FakeResponse(body, status_code=400)

    with pytest.raises(RuntimeError, match="exceeds search limit"):
        download_events("2000-01-01", "2024-01-01", 4.5, raw_dir=tmp_path, get=get)
    assert not catalogue_path("2000-01-01", "2024-01-01", 4.5, tmp_path).exists()


def test_load_keeps_only_earthquakes_and_warns_with_counts(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text(CSV)

    with pytest.warns(UserWarning, match=r"Dropped 1 events .*'volcanic eruption': 1"):
        events = load_earthquakes(path)

    assert list(events["id"]) == ["a", "b"]


def test_load_rejects_missing_columns(tmp_path):
    path = tmp_path / "events.csv"
    pd.DataFrame({"latitude": [1.0], "longitude": [2.0]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        load_earthquakes(path)


def test_load_rejects_out_of_range_coordinates(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text(CSV.replace("12.3,143.3", "95.0,143.3"))

    with pytest.raises(ValueError, match="1 events have coordinates outside"):
        load_earthquakes(path)
