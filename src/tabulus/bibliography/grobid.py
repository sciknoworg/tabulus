from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from tabulus.bibliography.models import (
    Bibliography,
    BibliographyEntry,
)


GROBID_SOURCE = "grobid"
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

DOI_PATTERN = re.compile(
    r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
    re.IGNORECASE,
)

YEAR_PATTERN = re.compile(r"\b((?:18|19|20|21)\d{2})\b")


def normalize_text(value: str) -> str:
    """Collapse whitespace without otherwise rewriting bibliography text."""

    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()


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
    value = re.sub(
        r"\bdoi\s*:\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )

    match = DOI_PATTERN.search(value)

    if match is None:
        return ""

    return match.group(1).rstrip(").,;]")


def _element_text(
    element: ET.Element | None,
) -> str:
    if element is None:
        return ""

    return normalize_text(
        "".join(element.itertext())
    )


def _raw_reference_text(
    bibl: ET.Element,
) -> str:
    """Prefer GROBID's raw-reference note when available."""

    raw_note = bibl.find(
        "tei:note[@type='raw_reference']",
        TEI_NS,
    )

    raw_text = _element_text(raw_note)

    if raw_text:
        return raw_text

    return _element_text(bibl)


def _structured_authors(
    bibl: ET.Element,
) -> tuple[str, ...]:
    authors = bibl.findall(
        "tei:analytic/tei:author",
        TEI_NS,
    )

    if not authors:
        authors = bibl.findall(
            "tei:monogr/tei:author",
            TEI_NS,
        )

    result: list[str] = []

    for author in authors:
        pers_name = author.find(
            "tei:persName",
            TEI_NS,
        )

        if pers_name is None:
            fallback = _element_text(author)

            if fallback:
                result.append(fallback)

            continue

        forenames = [
            _element_text(element)
            for element in pers_name.findall(
                "tei:forename",
                TEI_NS,
            )
        ]

        surnames = [
            _element_text(element)
            for element in pers_name.findall(
                "tei:surname",
                TEI_NS,
            )
        ]

        name = " ".join(
            part
            for part in (*forenames, *surnames)
            if part
        )

        if not name:
            name = _element_text(pers_name)

        if name:
            result.append(name)

    return tuple(result)


def _structured_title(
    bibl: ET.Element,
) -> str:
    analytic_title = _element_text(
        bibl.find(
            "tei:analytic/tei:title",
            TEI_NS,
        )
    )

    if analytic_title:
        return analytic_title

    # For a monographic work, the monograph title is the work title rather
    # than a journal/container title.
    for title in bibl.findall(
        "tei:monogr/tei:title",
        TEI_NS,
    ):
        if (
            str(title.get("level") or "").casefold()
            == "m"
        ):
            value = _element_text(title)

            if value:
                return value

    return ""


def _structured_venue(
    bibl: ET.Element,
) -> str:
    for title in bibl.findall(
        "tei:monogr/tei:title",
        TEI_NS,
    ):
        level = str(
            title.get("level") or ""
        ).casefold()

        if level in {"j", "s"}:
            value = _element_text(title)

            if value:
                return value

    return ""


def _structured_year(
    bibl: ET.Element,
) -> int | None:
    dates = bibl.findall(
        ".//tei:date",
        TEI_NS,
    )

    for date in dates:
        candidate = str(
            date.get("when") or ""
        ).strip()

        if not candidate:
            candidate = _element_text(date)

        match = YEAR_PATTERN.search(candidate)

        if match is not None:
            return int(match.group(1))

    return None


def _scope_value(
    bibl: ET.Element,
    units: set[str],
) -> str:
    for scope in bibl.findall(
        ".//tei:biblScope",
        TEI_NS,
    ):
        unit = str(
            scope.get("unit") or ""
        ).casefold()

        if unit not in units:
            continue

        value = _element_text(scope)

        if value:
            return value

        start = str(
            scope.get("from") or ""
        ).strip()
        end = str(
            scope.get("to") or ""
        ).strip()

        if start and end:
            return f"{start}-{end}"

        if start:
            return start

    return ""


def _structured_pages(
    bibl: ET.Element,
) -> str:
    for scope in bibl.findall(
        ".//tei:biblScope",
        TEI_NS,
    ):
        unit = str(
            scope.get("unit") or ""
        ).casefold()

        if unit not in {
            "page",
            "pages",
            "pp",
        }:
            continue

        start = str(
            scope.get("from") or ""
        ).strip()
        end = str(
            scope.get("to") or ""
        ).strip()

        if start and end:
            return f"{start}-{end}"

        if start:
            return start

        value = _element_text(scope)

        if value:
            return value

    return ""


def _recover_year(
    raw: str,
    structured_year: int | None,
) -> int | None:
    """Recover a missing year only from one unambiguous raw occurrence.

    Structured GROBID dates always take precedence. Raw-text recovery is
    intentionally conservative: if zero or multiple plausible year
    occurrences are present, no year is inferred.
    """

    if structured_year is not None:
        return structured_year

    years = [
        int(match.group(1))
        for match in YEAR_PATTERN.finditer(
            str(raw or "")
        )
    ]

    if len(years) == 1:
        return years[0]

    return None


def _recover_misassigned_pages(
    raw: str,
    structured_pages: str,
    *,
    structured_year: int | None,
    resolved_year: int | None,
) -> str:
    """Repair the observed GROBID year-as-page failure conservatively.

    Recovery is attempted only when:

    * GROBID supplied no structured year;
    * exactly one raw year was recovered; and
    * GROBID's page value is exactly that recovered year.

    In that situation the incorrect page value is discarded. A replacement
    locator is accepted only when exactly one simple page/page-range occurs
    immediately before the raw publication year.
    """

    if (
        structured_year is not None
        or resolved_year is None
    ):
        return structured_pages

    if (
        normalize_text(structured_pages)
        != str(resolved_year)
    ):
        return structured_pages

    year_text = re.escape(
        str(resolved_year)
    )

    pattern = re.compile(
        (
            r",\s*"
            r"([A-Za-z]?\d+"
            r"(?:\s*[-–—]\s*[A-Za-z]?\d+)?)"
            r"\s*[^0-9A-Za-z]{0,8}"
            + year_text
            + r"\b"
        )
    )

    candidates: list[str] = []

    for match in pattern.finditer(
        str(raw or "")
    ):
        value = normalize_text(
            match.group(1)
        )

        value = re.sub(
            r"\s*[-–—]\s*",
            "-",
            value,
        )

        if value not in candidates:
            candidates.append(value)

    if len(candidates) == 1:
        return candidates[0]

    # The GROBID value is known to be the year, not a trustworthy page
    # locator. Do not preserve known-bad evidence if raw recovery is
    # ambiguous or impossible.
    return ""


def parse_grobid_tei(
    tei_xml: str,
) -> Bibliography:
    """Parse GROBID ``processReferences`` TEI.

    Entry indices follow TEI bibliography order and are never renumbered based
    on content. This preserves the positional mapping required for numeric
    citations such as ``[12]`` in reconstructed tables.

    The raw citation string is preserved while structured bibliographic
    evidence is retained where GROBID provides it. A narrow fallback recovers
    an otherwise missing publication year when exactly one plausible year
    occurs in the raw citation, and repairs the observed GROBID failure in
    which that year is misassigned to the page field.
    """

    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError as error:
        raise ValueError(
            "GROBID response is not valid TEI XML."
        ) from error

    bibliography: list[
        BibliographyEntry
    ] = []

    bibls = root.findall(
        ".//tei:listBibl//tei:biblStruct",
        TEI_NS,
    )

    for index, bibl in enumerate(
        bibls,
        start=1,
    ):
        raw = _raw_reference_text(bibl)

        structured_year = _structured_year(
            bibl
        )
        year = _recover_year(
            raw,
            structured_year,
        )

        structured_pages = _structured_pages(
            bibl
        )
        pages = _recover_misassigned_pages(
            raw,
            structured_pages,
            structured_year=structured_year,
            resolved_year=year,
        )

        bibliography.append(
            BibliographyEntry(
                index=index,
                raw=raw,
                doi=extract_doi(raw),
                source=GROBID_SOURCE,
                title=_structured_title(bibl),
                authors=_structured_authors(bibl),
                year=year,
                venue=_structured_venue(bibl),
                volume=_scope_value(
                    bibl,
                    {"volume", "vol"},
                ),
                issue=_scope_value(
                    bibl,
                    {"issue", "number"},
                ),
                pages=pages,
            )
        )

    return Bibliography(
        source=GROBID_SOURCE,
        entries=tuple(bibliography),
    )
