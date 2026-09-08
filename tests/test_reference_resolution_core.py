from __future__ import annotations

import json
import urllib.error
import urllib.parse

import pytest

import tabulus.reference_resolution.core as core_module
from tabulus.reference_resolution.core import (
    CoreClient,
    CoreError,
    candidate_from_core_work,
)


class _FakeHeaders(dict):
    def get(
        self,
        key,
        default=None,
    ):
        return super().get(
            key.lower(),
            default,
        )


class _FakeResponse:
    def __init__(
        self,
        payload: dict,
        headers=None,
    ):
        self.payload = json.dumps(
            payload
        ).encode("utf-8")
        self.headers = _FakeHeaders(
            {
                str(key).lower(): value
                for key, value
                in (headers or {}).items()
            }
        )

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ):
        return False

    def read(self):
        return self.payload


def test_candidate_from_core_work_normalizes_live_schema() -> None:
    candidate = candidate_from_core_work(
        {
            "id": 47879950,
            "title": (
                "Atomic Layer Deposition for "
                "Novel Dye-Sensitized Solar Cells"
            ),
            "doi": "10.1149/1.3633681",
            "authors": [
                {"name": "Stefik, M."},
                {"name": "Tetreault, N."},
            ],
            "publishedDate": "2011-09-01",
            "yearPublished": 2025,
            "journals": [],
            "publisher": "Pennington",
            "downloadUrl": "",
            "links": [
                {
                    "type": "display",
                    "url": (
                        "https://core.ac.uk/"
                        "works/47879950"
                    ),
                }
            ],
        }
    )

    assert candidate.source == "core"
    assert candidate.source_id == "47879950"
    assert candidate.doi == "10.1149/1.3633681"
    assert candidate.authors == (
        "Stefik, M.",
        "Tetreault, N.",
    )

    # Prefer explicit publication date when available.
    assert candidate.year == 2011

    assert candidate.venue == ""
    assert candidate.url == (
        "https://core.ac.uk/works/47879950"
    )


def test_core_year_falls_back_to_year_published() -> None:
    candidate = candidate_from_core_work(
        {
            "id": 7,
            "yearPublished": 2020,
        }
    )

    assert candidate.year == 2020


def test_candidate_from_core_work_handles_sparse_record() -> None:
    candidate = candidate_from_core_work(
        {
            "id": 8,
            "title": "Sparse work",
        }
    )

    assert candidate.source_id == "8"
    assert candidate.title == "Sparse work"
    assert candidate.doi == ""
    assert candidate.authors == ()
    assert candidate.year is None
    assert candidate.venue == ""


def test_core_client_requires_api_key() -> None:
    with pytest.raises(
        ValueError,
        match="api_key",
    ):
        CoreClient(api_key="")


def test_core_search_uses_canonical_endpoint_and_bearer_auth(
    monkeypatch,
) -> None:
    seen = {}

    def fake_urlopen(
        request,
        timeout,
    ):
        seen["url"] = (
            request.full_url
        )
        seen["authorization"] = (
            request.get_header(
                "Authorization"
            )
        )
        seen["timeout"] = timeout

        return _FakeResponse(
            {
                "limit": 5,
                "offset": 0,
                "results": [
                    {
                        "id": 101,
                        "doi": "10.1000/a",
                        "title": "Candidate A",
                    }
                ],
                "searchId": "search-1",
                "totalHits": 20,
            },
            headers={
                "x-ratelimit-limit": "500",
                "x-ratelimit-remaining": "499",
                "x-ratelimit-retry-after": (
                    "2026-09-05T12:54:56+0000"
                ),
            },
        )

    monkeypatch.setattr(
        core_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )

    client = CoreClient(
        api_key="secret-key",
        timeout_seconds=12.0,
        limit=5,
    )

    result = client.search_works(
        "atomic layer deposition"
    )

    parsed = urllib.parse.urlparse(
        seen["url"]
    )

    assert parsed.path.endswith(
        "/v3/search/works/"
    )

    query = urllib.parse.parse_qs(
        parsed.query
    )

    assert query["q"] == [
        "atomic layer deposition"
    ]
    assert query["limit"] == ["5"]

    assert seen["authorization"] == (
        "Bearer secret-key"
    )
    assert seen["timeout"] == 12.0

    assert len(result.candidates) == 1
    assert (
        result.candidates[0].doi
        == "10.1000/a"
    )


def test_core_search_captures_rate_limit_metadata(
    monkeypatch,
) -> None:
    def fake_urlopen(
        request,
        timeout,
    ):
        return _FakeResponse(
            {
                "limit": 5,
                "offset": 0,
                "results": [],
                "searchId": "abc",
                "totalHits": 0,
            },
            headers={
                "x-ratelimit-limit": "500",
                "x-ratelimit-remaining": "321",
                "x-ratelimit-retry-after": (
                    "2026-09-05T13:00:00+0000"
                ),
            },
        )

    monkeypatch.setattr(
        core_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )

    result = CoreClient(
        api_key="secret",
    ).search_works(
        "example"
    )

    assert result.rate_limit.limit == 500
    assert result.rate_limit.remaining == 321
    assert result.rate_limit.retry_after == (
        "2026-09-05T13:00:00+0000"
    )
    assert result.total_hits == 0
    assert result.search_id == "abc"


def test_core_empty_reference_does_not_make_request(
    monkeypatch,
) -> None:
    def fail_urlopen(
        *args,
        **kwargs,
    ):
        raise AssertionError(
            "Network request should not occur."
        )

    monkeypatch.setattr(
        core_module.urllib.request,
        "urlopen",
        fail_urlopen,
    )

    result = CoreClient(
        api_key="secret",
    ).search_works("")

    assert result.candidates == ()
    assert result.total_hits == 0


def test_core_http_error_is_explicit(
    monkeypatch,
) -> None:
    def fake_urlopen(
        request,
        timeout,
    ):
        raise urllib.error.HTTPError(
            request.full_url,
            429,
            "Too Many Requests",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(
        core_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        CoreError,
        match="HTTP 429",
    ):
        CoreClient(
            api_key="secret",
        ).search_works(
            "Example citation"
        )


def test_core_invalid_results_payload_is_explicit(
    monkeypatch,
) -> None:
    def fake_urlopen(
        request,
        timeout,
    ):
        return _FakeResponse(
            {
                "results": "not-a-list",
            }
        )

    monkeypatch.setattr(
        core_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        CoreError,
        match="results list",
    ):
        CoreClient(
            api_key="secret",
        ).search_works(
            "Example citation"
        )



def test_core_retries_transient_500_then_succeeds(
    monkeypatch,
) -> None:
    import json
    import urllib.error

    import tabulus.reference_resolution.core as core_module

    attempts = []
    sleeps = []

    class FakeResponse:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(
            self,
            exc_type,
            exc,
            tb,
        ):
            return False

        def read(self):
            return json.dumps(
                {
                    "results": [],
                    "totalHits": 0,
                    "searchId": "retry-test",
                    "limit": 5,
                    "offset": 0,
                }
            ).encode("utf-8")

    def fake_urlopen(
        request,
        timeout,
    ):
        attempts.append(
            request.full_url
        )

        if len(attempts) < 3:
            raise urllib.error.HTTPError(
                request.full_url,
                500,
                "Internal Server Error",
                hdrs={},
                fp=None,
            )

        return FakeResponse()

    monkeypatch.setattr(
        core_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr(
        core_module.time,
        "sleep",
        sleeps.append,
    )

    response = core_module.CoreClient(
        api_key="secret",
    ).search_works(
        "example reference"
    )

    assert response.total_hits == 0
    assert len(attempts) == 3
    assert sleeps == [1.0, 2.0]


def test_core_exhausts_retryable_503_budget(
    monkeypatch,
) -> None:
    import urllib.error

    import pytest

    import tabulus.reference_resolution.core as core_module

    attempts = []
    sleeps = []

    def fake_urlopen(
        request,
        timeout,
    ):
        attempts.append(
            request.full_url
        )

        raise urllib.error.HTTPError(
            request.full_url,
            503,
            "Service Unavailable",
            hdrs={},
            fp=None,
        )

    monkeypatch.setattr(
        core_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr(
        core_module.time,
        "sleep",
        sleeps.append,
    )

    with pytest.raises(
        core_module.CoreError,
        match="HTTP 503 after 4 attempts",
    ):
        core_module.CoreClient(
            api_key="secret",
        ).search_works(
            "example reference"
        )

    assert len(attempts) == 4
    assert sleeps == [1.0, 2.0, 4.0]


def test_core_retries_http_429_and_respects_retry_after(
    monkeypatch,
) -> None:
    import json
    import urllib.error

    import tabulus.reference_resolution.core as core_module

    attempts = []
    sleeps = []

    class FakeResponse:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(
            self,
            exc_type,
            exc,
            tb,
        ):
            return False

        def read(self):
            return json.dumps(
                {
                    "results": [],
                    "totalHits": 0,
                    "searchId": "rate-test",
                    "limit": 5,
                    "offset": 0,
                }
            ).encode("utf-8")

    def fake_urlopen(
        request,
        timeout,
    ):
        attempts.append(
            request.full_url
        )

        if len(attempts) == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                429,
                "Too Many Requests",
                hdrs={
                    "Retry-After": "2",
                },
                fp=None,
            )

        return FakeResponse()

    monkeypatch.setattr(
        core_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr(
        core_module.time,
        "sleep",
        sleeps.append,
    )

    core_module.CoreClient(
        api_key="secret",
    ).search_works(
        "example reference"
    )

    assert len(attempts) == 2
    assert sleeps == [2.0]


def test_core_does_not_retry_non_transient_400(
    monkeypatch,
) -> None:
    import urllib.error

    import pytest

    import tabulus.reference_resolution.core as core_module

    attempts = []
    sleeps = []

    def fake_urlopen(
        request,
        timeout,
    ):
        attempts.append(
            request.full_url
        )

        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "Bad Request",
            hdrs={},
            fp=None,
        )

    monkeypatch.setattr(
        core_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr(
        core_module.time,
        "sleep",
        sleeps.append,
    )

    with pytest.raises(
        core_module.CoreError,
        match="HTTP 400",
    ):
        core_module.CoreClient(
            api_key="secret",
        ).search_works(
            "example reference"
        )

    assert len(attempts) == 1
    assert sleeps == []



def test_sanitize_core_query_removes_query_metacharacters() -> None:
    from tabulus.reference_resolution.core import (
        sanitize_core_query,
    )

    raw = (
        "M. Akamatsu, S. Narahara, T. Kobayashi, "
        "and F. Hasegawa, Appl. Surf. Sci. "
        "82/83, 228 ͑1994͒."
    )

    cleaned = sanitize_core_query(
        raw
    )

    assert "82/83" not in cleaned
    assert "82 83" in cleaned
    assert "," not in cleaned
    assert "/" not in cleaned

    assert "Akamatsu" in cleaned
    assert "Narahara" in cleaned
    assert "Appl. Surf. Sci." in cleaned
    assert "228" in cleaned
    assert "1994" in cleaned


def test_sanitize_core_query_handles_common_query_syntax() -> None:
    from tabulus.reference_resolution.core import (
        sanitize_core_query,
    )

    cleaned = sanitize_core_query(
        'TiO2 [12/13]: "ALD-growth" + surface?'
    )

    for character in (
        "[",
        "]",
        "/",
        ":",
        '"',
        "+",
        "?",
    ):
        assert character not in cleaned

    assert "TiO2" in cleaned
    assert "12 13" in cleaned
    assert "ALD" in cleaned
    assert "growth" in cleaned
    assert "surface" in cleaned


def test_core_search_sends_sanitized_query(
    monkeypatch,
) -> None:
    import json
    import urllib.parse

    import tabulus.reference_resolution.core as core_module

    seen = {}

    class FakeResponse:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(
            self,
            exc_type,
            exc,
            tb,
        ):
            return False

        def read(self):
            return json.dumps(
                {
                    "results": [],
                    "totalHits": 0,
                    "searchId": "sanitized-test",
                    "limit": 5,
                    "offset": 0,
                }
            ).encode("utf-8")

    def fake_urlopen(
        request,
        timeout,
    ):
        seen["url"] = request.full_url
        return FakeResponse()

    monkeypatch.setattr(
        core_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )

    raw = (
        "M. Akamatsu, S. Narahara, T. Kobayashi, "
        "and F. Hasegawa, Appl. Surf. Sci. "
        "82/83, 228 ͑1994͒."
    )

    core_module.CoreClient(
        api_key="secret",
    ).search_works(
        raw
    )

    parsed = urllib.parse.urlparse(
        seen["url"]
    )

    query = urllib.parse.parse_qs(
        parsed.query
    )["q"][0]

    assert "82/83" not in query
    assert "82 83" in query
    assert "Akamatsu" in query



def test_request_json_retries_timeout(
    monkeypatch,
) -> None:
    calls = []
    sleeps = []

    class FakeResponse:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(
            self,
            exc_type,
            exc_value,
            traceback,
        ):
            return False

        def read(self):
            return b'{"results": []}'

    def fake_urlopen(
        request,
        timeout,
    ):
        calls.append(request)

        if len(calls) == 1:
            raise TimeoutError(
                "The read operation timed out"
            )

        return FakeResponse()

    monkeypatch.setattr(
        core_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )

    monkeypatch.setattr(
        core_module.time,
        "sleep",
        lambda seconds: sleeps.append(seconds),
    )

    payload, rate_limit = core_module._request_json(
        "https://api.core.ac.uk/v3/search/works/?q=test&limit=5",
        api_key="secret",
        timeout_seconds=10.0,
    )

    assert payload == {
        "results": []
    }

    assert len(calls) == 2
    assert sleeps == [1.0]
