from __future__ import annotations

from pathlib import Path

from tabulus.bibliography.grobid_client import (
    DEFAULT_GROBID_TIMEOUT_SECONDS,
    GrobidBibliographyExtractor,
)
from tabulus.bibliography.output import BIBLIOGRAPHY_NAME, write_bibliography_json


REFERENCES_DIR_NAME = "references"


def extract_bibliography_artifact(
    pdf_path: Path,
    artifact_root: Path,
    *,
    grobid_url: str,
    timeout_seconds: float = DEFAULT_GROBID_TIMEOUT_SECONDS,
) -> Path:
    """Extract and persist the normalized bibliography for one original PDF.

    Bibliography extraction is intentionally independent of table detection and
    reconstruction. The original PDF is sent to GROBID and the normalized
    artifact is written beneath ``references/bibliography.json``.
    """

    pdf_path = Path(pdf_path).expanduser()
    artifact_root = Path(artifact_root).expanduser()

    extractor = GrobidBibliographyExtractor(
        grobid_url=grobid_url,
        timeout_seconds=timeout_seconds,
    )
    bibliography = extractor.extract(pdf_path)

    output_path = artifact_root / REFERENCES_DIR_NAME / BIBLIOGRAPHY_NAME
    return write_bibliography_json(bibliography, output_path)
