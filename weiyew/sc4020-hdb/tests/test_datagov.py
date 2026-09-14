import pytest

from hdb.datagov import API_URL, DATASETS, download_dataset, latest_download

FILE_URL = "https://s3.example/resale.csv"
CSV = b"month,town\n2017-01,ANG MO KIO\n"


class FakeResponse:
    def __init__(self, body=None, content=b"", status_code=200):
        self._body = body
        self.content = content
        self.status_code = status_code
        self.ok = status_code < 400

    def json(self):
        return self._body


def fake_api(responses):
    """A `get` that answers each URL from a queue of responses and records the URLs called."""
    calls = []

    def get(url, timeout):
        calls.append(url)
        return responses[url].pop(0)

    return get, calls


def ok(data):
    return FakeResponse({"code": 0, "data": data})


RATE_LIMITED = FakeResponse({"code": 24, "errorMsg": "Rate limit exceeded"})


def test_download_initiates_polls_and_saves_dated_file(tmp_path):
    base = f"{API_URL}/{DATASETS['resale']}"
    get, calls = fake_api(
        {
            f"{base}/initiate-download": [ok({"message": "initiated"})],
            f"{base}/poll-download": [ok({"status": "DOWNLOAD_IN_PROGRESS"}), ok({"status": "DOWNLOAD_SUCCESS", "url": FILE_URL})],
            FILE_URL: [FakeResponse(content=CSV)],
        }
    )

    target = download_dataset("resale", raw_dir=tmp_path, today="2026-09-14", get=get, sleep=lambda s: None)

    assert target == tmp_path / "resale_2026-09-14.csv"
    assert target.read_bytes() == CSV
    assert calls == [f"{base}/initiate-download", f"{base}/poll-download", f"{base}/poll-download", FILE_URL]


def test_download_retries_rate_limited_calls(tmp_path):
    base = f"{API_URL}/{DATASETS['price_index']}"
    pauses = []
    get, _ = fake_api(
        {
            f"{base}/initiate-download": [RATE_LIMITED, RATE_LIMITED, ok({})],
            f"{base}/poll-download": [ok({"url": FILE_URL})],
            FILE_URL: [FakeResponse(content=CSV)],
        }
    )

    download_dataset("price_index", raw_dir=tmp_path, today="2026-09-14", get=get, sleep=pauses.append)

    assert len(pauses) == 2


def test_download_gives_up_when_always_rate_limited(tmp_path):
    base = f"{API_URL}/{DATASETS['resale']}"
    get, _ = fake_api({f"{base}/initiate-download": [RATE_LIMITED] * 3})

    with pytest.raises(RuntimeError, match="still rate limited after 3 attempts"):
        download_dataset("resale", raw_dir=tmp_path, get=get, sleep=lambda s: None, max_attempts=3)
    assert not list(tmp_path.iterdir())


def test_download_raises_api_error_message(tmp_path):
    base = f"{API_URL}/{DATASETS['resale']}"
    get, _ = fake_api({f"{base}/initiate-download": [FakeResponse({"code": 5, "errorMsg": "Dataset not found"})]})

    with pytest.raises(RuntimeError, match="Dataset not found"):
        download_dataset("resale", raw_dir=tmp_path, get=get, sleep=lambda s: None)


def test_latest_download_picks_most_recent_date(tmp_path):
    for date in ["2026-08-01", "2026-09-14", "2026-01-31"]:
        (tmp_path / f"resale_{date}.csv").write_text("")
    (tmp_path / "price_index_2026-12-01.csv").write_text("")

    assert latest_download("resale", tmp_path) == tmp_path / "resale_2026-09-14.csv"


def test_latest_download_without_files_says_how_to_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m hdb.datagov resale"):
        latest_download("resale", tmp_path)
