from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from tabulus.bibliography.models import Bibliography, BibliographyEntry


GROBID_SOURCE = "grobid"
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
DOI_PATTERN = re.compile(
    r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
    re.IGNORECASE,
)


def normalize_text(value: str) -> str:
    """Collapse whitespace without otherwise rewriting bibliography text."""

    return re.sub(r"\s+", " ", str(value or "")).strip()


def extract_doi(text: str) -> str:
    """Extract and normalize the first DOI found in bibliography text."""

    if not text:
        return ""

    value = str(text).strip()
    value = re.sub(
        r"https?://(?:dx\.)?doi\.org/",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\bdoi\s*:\s*", "", value, flags=re.IGNORECASE)

    match = DOI_PATTERN.search(value)
    if match is None:
        return ""

    return match.group(1).rstrip(").,;]")


def _raw_reference_text(bibl: ET.Element) -> str:
    """Prefer GROBID's raw-reference note when available."""

    raw_note = bibl.find("tei:note[@type='raw_reference']", TEI_NS)
    if raw_note is not None:
        raw_text = normalize_text("".join(raw_note.itertext()))
        if raw_text:
            return raw_text

    return normalize_text("".join(bibl.itertext()))


def parse_grobid_tei(tei_xml: str) -> Bibliography:
    """
    Parse GROBID ``processReferences`` TEI into Tabulus bibliography entries.

    Entry indices follow TEI bibliography order and are never renumbered based
    on content. This preserves the positional mapping required for numeric
    citations such as ``[12]`` in reconstructed tables.
    """

    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError as error:
        raise ValueError("GROBID response is not valid TEI XML.") from error

    bibliography: list[BibliographyEntry] = []
    bibls = root.findall(".//tei:listBibl//tei:biblStruct", TEI_NS)

    for index, bibl in enumerate(bibls, start=1):
        raw = _raw_reference_text(bibl)
        bibliography.append(
            BibliographyEntry(
                index=index,
                raw=raw,
                doi=extract_doi(raw),
                source=GROBID_SOURCE,
            )
        )

    return Bibliography(
        source=GROBID_SOURCE,
        entries=tuple(bibliography),
    )
