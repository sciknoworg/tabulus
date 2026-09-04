from __future__ import annotations

from io import BytesIO
from pathlib import Path
import urllib.error

import pytest

from tabulus.bibliography import (
    BibliographyExtractionError,
    GrobidBibliographyExtractor,
    call_grobid_process_references,
)
import tabulus.bibliography.grobid_client as grobid_client


TEI_RESPONSE = b"""\
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text><back><listBibl>
    <biblStruct xml:id="b0">
      <note type="raw_reference">Smith J. Example. 2020. doi:10.1234/example</note>
    </biblStruct>
  </listBibl></back></text>
</TEI>
"""


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def _write_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.4\nfixture\n")
    return path


def test_grobid_request_preserves_raw_citations_and_skips_consolidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = _write_pdf(tmp_path)
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeResponse(TEI_RESPONSE)

    monkeypatch.setattr(grobid_client.urllib.request, "urlopen", fake_urlopen)

    xml = call_grobid_process_references(
        pdf_path,
        "http://grobid.example:8070",
        timeout_seconds=42,
    )

    request = captured["request"]
    body = request.data

    assert xml.startswith("<TEI")
    assert request.full_url == "http://grobid.example:8070/api/processReferences"
    assert captured["timeout"] == 42
    assert b'name="input"' in body
    assert b'filename="paper.pdf"' in body
    assert b'name="includeRawCitations"' in body
    assert b"\r\n1\r\n" in body
    assert b'name="consolidateCitations"' in body
    assert b"\r\n0\r\n" in body


def test_grobid_extractor_returns_normalized_bibliography(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = _write_pdf(tmp_path)
    monkeypatch.setattr(
        grobid_client.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeResponse(TEI_RESPONSE),
    )

    result = GrobidBibliographyExtractor("http://localhost:8070").extract(pdf_path)

    assert result.bibliography_count == 1
    assert result.entries[0].index == 1
    assert result.entries[0].doi == "10.1234/example"
    assert result.entries[0].source == "grobid"


def test_grobid_call_rejects_missing_pdf(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Bibliography PDF not found"):
        call_grobid_process_references(
            tmp_path / "missing.pdf",
            "http://localhost:8070",
        )


def test_grobid_http_error_is_wrapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = _write_pdf(tmp_path)

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            503,
            "Unavailable",
            hdrs=None,
            fp=BytesIO(b"service unavailable"),
        )

    monkeypatch.setattr(grobid_client.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(BibliographyExtractionError, match="HTTP 503"):
        call_grobid_process_references(
            pdf_path,
            "http://localhost:8070",
        )


def test_grobid_empty_response_is_an_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = _write_pdf(tmp_path)
    monkeypatch.setattr(
        grobid_client.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeResponse(b""),
    )

    with pytest.raises(BibliographyExtractionError, match="empty response"):
        call_grobid_process_references(
            pdf_path,
            "http://localhost:8070",
        )
