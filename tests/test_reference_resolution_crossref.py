from __future__ import annotations

import json
import urllib.error
import urllib.parse

import pytest

import tabulus.reference_resolution.crossref as crossref_module
from tabulus.reference_resolution.crossref import (
    CrossrefClient,
    CrossrefError,
    candidate_from_crossref_item,
)


class _FakeResponse:
    def __init__(self, payload: dict):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.payload


def test_candidate_from_crossref_item_normalizes_metadata() -> None:
    candidate = candidate_from_crossref_item(
        {
            "DOI": "10.1000/EXAMPLE",
            "title": ["Example paper"],
            "author": [
                {
                    "given": "Jane",
                    "family": "Smith",
                },
                {
                    "given": "John",
                    "family": "Jones",
                },
            ],
            "issued": {
                "date-parts": [[2020, 5, 1]],
            },
            "container-title": ["Example Journal"],
            "volume": "12",
            "issue": "3",
            "page": "100-110",
            "URL": "https://doi.org/10.1000/example",
        }
    )

    assert candidate.source == "crossref"
    assert candidate.source_id == "10.1000/example"
    assert candidate.doi == "10.1000/example"
    assert candidate.title == "Example paper"
    assert candidate.authors == (
        "Jane Smith",
        "John Jones",
    )
    assert candidate.year == 2020
    assert candidate.venue == "Example Journal"
    assert candidate.volume == "12"
    assert candidate.issue == "3"
    assert candidate.pages == "100-110"


def test_candidate_from_crossref_item_handles_sparse_record() -> None:
    candidate = candidate_from_crossref_item(
        {
            "title": [],
            "author": [],
        }
    )

    assert candidate.source == "crossref"
    assert candidate.doi == ""
    assert candidate.title == ""
    assert candidate.authors == ()
    assert candidate.year is None
    assert candidate.venue == ""


def test_crossref_client_requires_mailto() -> None:
    with pytest.raises(ValueError, match="mailto"):
        CrossrefClient(mailto="")


def test_crossref_doi_lookup_builds_expected_request(
    monkeypatch,
) -> None:
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["user_agent"] = request.get_header("User-agent")
        seen["timeout"] = timeout

        return _FakeResponse(
            {
                "message": {
                    "DOI": "10.1000/example",
                    "title": ["Example paper"],
                    "issued": {
                        "date-parts": [[2020]],
                    },
                }
            }
        )

    monkeypatch.setattr(
        crossref_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )

    client = CrossrefClient(
        mailto="researcher@example.org",
        timeout_seconds=12.0,
    )

    result = client.get_by_doi(
        "https://doi.org/10.1000/EXAMPLE"
    )

    assert result is not None
    assert result.doi == "10.1000/example"
    assert seen["url"].endswith(
        "/works/10.1000%2Fexample"
    )
    assert "researcher@example.org" in seen["user_agent"]
    assert seen["timeout"] == 12.0


def test_crossref_doi_lookup_returns_none_on_404(
    monkeypatch,
) -> None:
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            404,
            "Not Found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(
        crossref_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )

    client = CrossrefClient(
        mailto="researcher@example.org",
    )

    assert client.get_by_doi("10.1000/missing") is None


def test_crossref_doi_lookup_keeps_other_http_errors_explicit(
    monkeypatch,
) -> None:
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            503,
            "Unavailable",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(
        crossref_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )

    client = CrossrefClient(
        mailto="researcher@example.org",
    )

    with pytest.raises(
        CrossrefError,
        match="HTTP 503",
    ):
        client.get_by_doi("10.1000/example")


def test_crossref_bibliographic_search_requests_multiple_candidates(
    monkeypatch,
) -> None:
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url

        return _FakeResponse(
            {
                "message": {
                    "items": [
                        {
                            "DOI": "10.1000/a",
                            "title": ["Candidate A"],
                        },
                        {
                            "DOI": "10.1000/b",
                            "title": ["Candidate B"],
                        },
                    ]
                }
            }
        )

    monkeypatch.setattr(
        crossref_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )

    client = CrossrefClient(
        mailto="researcher@example.org",
        rows=5,
    )

    candidates = client.search_bibliographic(
        "Smith J. Example paper. 2020."
    )

    assert [candidate.doi for candidate in candidates] == [
        "10.1000/a",
        "10.1000/b",
    ]

    parsed = urllib.parse.urlparse(seen["url"])
    query = urllib.parse.parse_qs(parsed.query)

    assert query["query.bibliographic"] == [
        "Smith J. Example paper. 2020."
    ]
    assert query["rows"] == ["5"]
    assert query["mailto"] == [
        "researcher@example.org"
    ]


def test_crossref_empty_reference_does_not_make_request(
    monkeypatch,
) -> None:
    def fail_urlopen(*args, **kwargs):
        raise AssertionError("Network request should not occur.")

    monkeypatch.setattr(
        crossref_module.urllib.request,
        "urlopen",
        fail_urlopen,
    )

    client = CrossrefClient(
        mailto="researcher@example.org",
    )

    assert client.search_bibliographic("") == ()


def test_crossref_invalid_search_payload_is_explicit(
    monkeypatch,
) -> None:
    def fake_urlopen(request, timeout):
        return _FakeResponse(
            {
                "message": {
                    "items": "not-a-list",
                }
            }
        )

    monkeypatch.setattr(
        crossref_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )

    client = CrossrefClient(
        mailto="researcher@example.org",
    )

    with pytest.raises(
        CrossrefError,
        match="items list",
    ):
        client.search_bibliographic(
            "Example citation"
        )
