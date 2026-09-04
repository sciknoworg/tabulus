from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import urllib.error
import urllib.request
import uuid

from tabulus.bibliography.grobid import parse_grobid_tei
from tabulus.bibliography.models import Bibliography


DEFAULT_GROBID_TIMEOUT_SECONDS = 300.0


class BibliographyExtractionError(RuntimeError):
    """Raised when a bibliography extractor cannot obtain usable source data."""


def _validate_pdf_path(pdf_path: Path) -> Path:
    path = Path(pdf_path).expanduser()

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Bibliography input is not a PDF file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Bibliography PDF not found: {path}")

    return path


def _multipart_part(
    *,
    boundary: str,
    name: str,
    value: bytes,
    filename: str | None = None,
    content_type: str | None = None,
) -> bytes:
    headers = [
        f"--{boundary}",
        (
            f'Content-Disposition: form-data; name="{name}"; '
            f'filename="{filename.replace(chr(34), "_")}"'
            if filename is not None
            else f'Content-Disposition: form-data; name="{name}"'
        ),
    ]
    if content_type is not None:
        headers.append(f"Content-Type: {content_type}")
    headers.append("")

    return "\r\n".join(headers).encode("utf-8") + b"\r\n" + value + b"\r\n"


def _build_process_references_request(pdf_path: Path, grobid_url: str) -> urllib.request.Request:
    boundary = f"tabulus-{uuid.uuid4().hex}"
    body = b"".join(
        [
            _multipart_part(
                boundary=boundary,
                name="input",
                value=pdf_path.read_bytes(),
                filename=pdf_path.name,
                content_type="application/pdf",
            ),
            # Raw citations preserve the source bibliography text used later
            # for deterministic table-reference matching.
            _multipart_part(
                boundary=boundary,
                name="includeRawCitations",
                value=b"1",
            ),
            # External metadata consolidation belongs to the later DOI
            # resolution stage, not bibliography extraction.
            _multipart_part(
                boundary=boundary,
                name="consolidateCitations",
                value=b"0",
            ),
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )

    endpoint = f"{grobid_url.rstrip('/')}/api/processReferences"
    return urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Accept": "application/xml",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "tabulus/0.1",
        },
        method="POST",
    )


def call_grobid_process_references(
    pdf_path: Path,
    grobid_url: str,
    *,
    timeout_seconds: float = DEFAULT_GROBID_TIMEOUT_SECONDS,
) -> str:
    """Call GROBID ``processReferences`` and return its TEI XML response."""

    path = _validate_pdf_path(pdf_path)
    request = _build_process_references_request(path, grobid_url)

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read(500).decode("utf-8", errors="replace")
        raise BibliographyExtractionError(
            f"GROBID processReferences returned HTTP {error.code}: {detail}"
        ) from error
    except urllib.error.URLError as error:
        raise BibliographyExtractionError(
            f"Could not reach GROBID at {grobid_url}: {error.reason}"
        ) from error

    if not payload:
        raise BibliographyExtractionError(
            "GROBID processReferences returned an empty response."
        )

    return payload.decode("utf-8", errors="replace")


@dataclass(frozen=True)
class GrobidBibliographyExtractor:
    """Bibliography extractor backed by a GROBID service."""

    grobid_url: str
    timeout_seconds: float = DEFAULT_GROBID_TIMEOUT_SECONDS

    def extract(self, pdf_path: Path) -> Bibliography:
        tei_xml = call_grobid_process_references(
            pdf_path,
            self.grobid_url,
            timeout_seconds=self.timeout_seconds,
        )
        return parse_grobid_tei(tei_xml)
