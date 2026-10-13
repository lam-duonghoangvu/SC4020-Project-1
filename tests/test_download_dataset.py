import pytest

from sc4020.datasets.download_dataset import (
    DATAGOV_IDS,
    DATAGOV_URL,
    USGS_URL,
    download_datagov,
    download_earthquakes_2023,
    latest_datagov,
    usgs_catalogue_path,
    usgs_query_params,
)

CSV = "time,latitude,longitude,depth,mag,type,place,id\n2023-01-01T00:00:00Z,12.3,143.3,30.1,4.6,earthquake,Guam,a\n"
FILE_URL = "https://s3.example/resale.csv"
DATAGOV_CSV = b"month,town\n2017-01,ANG MO KIO\n"


class FakeResponse:
    def __init__(self, text="", body=None, content=None, status_code=200):
        self.text = text
        self.content = text.encode() if content is None else content
        self._body = body
        self.status_code = status_code
        self.ok = status_code < 400

    def json(self):
        return self._body


# --- USGS -------------------------------------------------------------------------------------


def test_usgs_query_params_ask_for_csv_in_time_order():
    assert usgs_query_params("2023-01-01", "2024-01-01", 4.5) == {
        "format": "csv",
        "starttime": "2023-01-01",
        "endtime": "2024-01-01",
        "minmagnitude": 4.5,
        "orderby": "time-asc",
    }


def test_usgs_download_saves_response_under_query_named_file(tmp_path):
    urls = []

    def get(url, params, timeout):
        urls.append(url)
        return FakeResponse(CSV)

    target = download_earthquakes_2023(tmp_path, get=get)

    assert target == tmp_path / "usgs_2023-01-01_2024-01-01_m4.5.csv"
    assert target.read_text() == CSV
    assert urls == [USGS_URL]


def test_usgs_download_keeps_existing_file_unless_forced(tmp_path):
    usgs_catalogue_path(data_home=tmp_path).write_text("old")
    calls = []

    def get(url, params, timeout):
        calls.append(url)
        return FakeResponse(CSV)

    assert download_earthquakes_2023(tmp_path, get=get).read_text() == "old"
    assert download_earthquakes_2023(tmp_path, force=True, get=get).read_text() == CSV
    assert len(calls) == 1


def test_usgs_download_over_search_limit_raises_with_api_message(tmp_path):
    def get(url, params, timeout):
        return FakeResponse("Error 400: 166725 matching events exceeds search limit of 20000.", status_code=400)

    with pytest.raises(RuntimeError, match="exceeds search limit"):
        download_earthquakes_2023(tmp_path, start="2000-01-01", get=get)
    assert not usgs_catalogue_path("2000-01-01", data_home=tmp_path).exists()


# --- data.gov.sg ------------------------------------------------------------------------------


def fake_api(responses):
    """A `get` that answers each URL from a queue of responses and records the URLs called."""
    calls = []

    def get(url, timeout):
        calls.append(url)
        return responses[url].pop(0)

    return get, calls


def ok(data):
    return FakeResponse(body={"code": 0, "data": data})


RATE_LIMITED = FakeResponse(body={"code": 24, "errorMsg": "Rate limit exceeded"})


def test_datagov_initiates_polls_and_saves_dated_file(tmp_path):
    base = f"{DATAGOV_URL}/{DATAGOV_IDS['resale']}"
    get, calls = fake_api(
        {
            f"{base}/initiate-download": [ok({"message": "initiated"})],
            f"{base}/poll-download": [ok({"status": "DOWNLOAD_IN_PROGRESS"}), ok({"status": "DOWNLOAD_SUCCESS", "url": FILE_URL})],
            FILE_URL: [FakeResponse(content=DATAGOV_CSV)],
        }
    )

    target = download_datagov("resale", tmp_path, today="2026-09-14", get=get, sleep=lambda s: None)

    assert target == tmp_path / "resale_2026-09-14.csv"
    assert target.read_bytes() == DATAGOV_CSV
    assert calls == [f"{base}/initiate-download", f"{base}/poll-download", f"{base}/poll-download", FILE_URL]


def test_datagov_retries_rate_limited_calls(tmp_path):
    base = f"{DATAGOV_URL}/{DATAGOV_IDS['price_index']}"
    pauses = []
    get, _ = fake_api(
        {
            f"{base}/initiate-download": [RATE_LIMITED, RATE_LIMITED, ok({})],
            f"{base}/poll-download": [ok({"url": FILE_URL})],
            FILE_URL: [FakeResponse(content=DATAGOV_CSV)],
        }
    )

    download_datagov("price_index", tmp_path, today="2026-09-14", get=get, sleep=pauses.append)

    assert len(pauses) == 2


def test_datagov_gives_up_when_always_rate_limited(tmp_path):
    base = f"{DATAGOV_URL}/{DATAGOV_IDS['resale']}"
    get, _ = fake_api({f"{base}/initiate-download": [RATE_LIMITED] * 3})

    with pytest.raises(RuntimeError, match="still rate limited after 3 attempts"):
        download_datagov("resale", tmp_path, get=get, sleep=lambda s: None, max_attempts=3)
    assert not list(tmp_path.iterdir())


def test_datagov_raises_api_error_message(tmp_path):
    base = f"{DATAGOV_URL}/{DATAGOV_IDS['resale']}"
    get, _ = fake_api({f"{base}/initiate-download": [FakeResponse(body={"code": 5, "errorMsg": "Dataset not found"})]})

    with pytest.raises(RuntimeError, match="Dataset not found"):
        download_datagov("resale", tmp_path, get=get, sleep=lambda s: None)


def test_latest_datagov_picks_most_recent_date(tmp_path):
    for date in ["2026-08-01", "2026-09-14", "2026-01-31"]:
        (tmp_path / f"resale_{date}.csv").write_text("")
    (tmp_path / "price_index_2026-12-01.csv").write_text("")

    assert latest_datagov("resale", tmp_path) == tmp_path / "resale_2026-09-14.csv"


def test_latest_datagov_without_files_says_how_to_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m sc4020.datasets.download_dataset hdb_resale"):
        latest_datagov("resale", tmp_path)
