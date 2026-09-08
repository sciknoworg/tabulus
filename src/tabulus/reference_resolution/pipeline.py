from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

LOGGER = logging.getLogger(__name__)


from tabulus.reference_resolution.models import (
    ReferenceEvidence,
    ResolutionCandidate,
)


_COMPLETE_CITATION_TAIL_PATTERN = re.compile(
    r"\b"
    r"\d+(?:/\d+)?"
    r"\s*,\s*"
    r"[A-Za-z]?\d+"
    r"\s*[^0-9A-Za-z]{0,12}"
    r"(?:18|19|20|21)\d{2}"
    r"\b"
)


def _is_bracketed_translation_pair(
    raw_reference: str,
    matches: list[re.Match[str]],
) -> bool:
    """Return whether two journal tails represent one translated publication.

    Older literature may give an original-language journal citation followed
    by a bracketed translated-journal equivalent. The translated edition can
    have a different page locator and, occasionally, an adjacent publication
    year.

    Treat the structure as one scholarly work only when:

    - exactly two complete journal-style citation tails are present;
    - the second tail is inside a bracketed secondary citation;
    - both tails have the same volume;
    - their publication years are identical or differ by one year.

    The narrow structural requirements preserve the non-atomic safeguard for
    ordinary concatenated references.
    """

    if len(matches) != 2:
        return False

    def tail_volume_year(
        match: re.Match[str],
    ) -> tuple[str, int] | None:
        text = match.group(0)

        volume_match = re.match(
            r"(?P<volume>\d+(?:/\d+)?)\s*,",
            text,
        )
        year_matches = re.findall(
            r"(?:18|19|20|21)\d{2}",
            text,
        )

        if volume_match is None or not year_matches:
            return None

        # Four-digit article/page locators can themselves fall in the
        # 1800-2199 range. The publication year is the final year-like
        # token in the complete citation tail.
        return (
            volume_match.group("volume"),
            int(year_matches[-1]),
        )

    first = tail_volume_year(matches[0])
    second = tail_volume_year(matches[1])

    if first is None or second is None:
        return False

    first_volume, first_year = first
    second_volume, second_year = second

    if first_volume != second_volume:
        return False

    if abs(first_year - second_year) > 1:
        return False

    raw = str(raw_reference or "")

    bracket_patterns = (
        re.compile(r"\[[^\]]+\]"),
        re.compile("\u0353[^\u0354]+\u0354"),
    )

    second_start = matches[1].start()
    second_end = matches[1].end()

    return any(
        bracket_match.start() <= second_start
        and second_end <= bracket_match.end()
        for pattern in bracket_patterns
        for bracket_match in pattern.finditer(raw)
    )


def is_non_atomic_reference(
    raw_reference: str,
) -> bool:
    """Return whether one bibliography entry is unsafe for one-work resolution.

    The conservative signal is the presence of at least two complete
    journal-style publication tails of the form ``volume, page (year)``.

    A narrow exception is made for an original-language journal citation
    followed by a bracketed same-year translated-journal equivalent. That
    structure represents one scholarly work in two publication forms rather
    than two independent works.

    Multiple bare years alone are deliberately not sufficient because
    proceedings references may legitimately contain both conference and
    publication years.
    """

    raw = str(raw_reference or "")

    matches = list(
        _COMPLETE_CITATION_TAIL_PATTERN.finditer(
            raw
        )
    )

    if len(matches) < 2:
        return False

    if _is_bracketed_translation_pair(
        raw,
        matches,
    ):
        return False

    return True


class CrossrefRetriever(Protocol):
    """Minimal Crossref interface required by the Stage 6 pipeline."""

    def get_by_doi(
        self,
        doi: str,
    ) -> ResolutionCandidate | None:
        ...

    def search_bibliographic(
        self,
        reference_text: str,
    ) -> tuple[ResolutionCandidate, ...]:
        ...


@dataclass(frozen=True)
class CrossrefRetrieval:
    """Crossref evidence collected for one unique bibliography entry.

    This is retrieval evidence only. It is deliberately not a final resolution
    decision: candidate validation, CORE fallback, and optional LLM
    adjudication happen in subsequent Stage 6 steps.
    """

    reference_index: int
    raw_reference: str
    existing_doi: str
    existing_doi_checked: bool
    existing_doi_candidate: ResolutionCandidate | None
    bibliographic_search_performed: bool
    bibliographic_candidates: tuple[ResolutionCandidate, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_index": self.reference_index,
            "raw_reference": self.raw_reference,
            "existing_doi": self.existing_doi,
            "existing_doi_checked": self.existing_doi_checked,
            "existing_doi_candidate": (
                self.existing_doi_candidate.to_dict()
                if self.existing_doi_candidate is not None
                else None
            ),
            "bibliographic_search_performed": (
                self.bibliographic_search_performed
            ),
            "bibliographic_candidates": [
                candidate.to_dict()
                for candidate in self.bibliographic_candidates
            ],
        }


def _load_json_object(
    path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    path = Path(path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{label} is not valid JSON: {path}"
        ) from error

    if not isinstance(value, dict):
        raise ValueError(
            f"{label} must contain a JSON object: {path}"
        )

    return value


def _load_bibliography_entries(
    bibliography_path: Path,
) -> dict[int, ReferenceEvidence]:
    payload = _load_json_object(
        bibliography_path,
        label="Bibliography artifact",
    )

    entries = payload.get("entries")

    if not isinstance(entries, list):
        raise ValueError(
            "Bibliography artifact must contain an entries list."
        )

    result: dict[int, ReferenceEvidence] = {}

    for item in entries:
        if not isinstance(item, dict):
            raise ValueError(
                "Bibliography entries must be JSON objects."
            )

        index = item.get("index")

        if (
            not isinstance(index, int)
            or index <= 0
            or index in result
        ):
            raise ValueError(
                "Bibliography entry indices must be unique "
                "positive integers."
            )

        authors_value = item.get("authors") or []

        if not isinstance(authors_value, list):
            raise ValueError(
                "Bibliography entry authors must be a list."
            )

        year_value = item.get("year")

        if (
            year_value is not None
            and not isinstance(year_value, int)
        ):
            raise ValueError(
                "Bibliography entry year must be an integer or null."
            )

        result[index] = ReferenceEvidence(
            reference_index=index,
            raw_reference=str(item.get("raw") or ""),
            doi=str(item.get("doi") or ""),
            title=str(item.get("title") or ""),
            authors=tuple(
                str(author or "").strip()
                for author in authors_value
                if str(author or "").strip()
            ),
            year=year_value,
            venue=str(item.get("venue") or ""),
            volume=str(item.get("volume") or ""),
            issue=str(item.get("issue") or ""),
            pages=str(item.get("pages") or ""),
        )

    return result


def _linked_indices_from_matches(
    reference_matches_path: Path,
) -> set[int]:
    payload = _load_json_object(
        reference_matches_path,
        label="Reference matches artifact",
    )

    matched_tables = payload.get("matched_tables")

    if not isinstance(matched_tables, list):
        raise ValueError(
            "Reference matches artifact must contain "
            "a matched_tables list."
        )

    indices: set[int] = set()

    for table in matched_tables:
        if not isinstance(table, dict):
            raise ValueError(
                "Matched table records must be JSON objects."
            )

        matches = table.get("matches")

        if not isinstance(matches, list):
            raise ValueError(
                "Matched table records must contain a matches list."
            )

        for match in matches:
            if not isinstance(match, dict):
                raise ValueError(
                    "Row-level match records must be JSON objects."
                )

            matched_indices = match.get(
                "matched_reference_indices"
            )

            if not isinstance(matched_indices, list):
                raise ValueError(
                    "Row-level match records must contain "
                    "a matched_reference_indices list."
                )

            for index in matched_indices:
                if not isinstance(index, int) or index <= 0:
                    raise ValueError(
                        "Matched reference indices must be "
                        "positive integers."
                    )

                indices.add(index)

    return indices


def discover_reference_match_artifacts(
    search_root: Path,
    paper_name: str,
) -> tuple[Path, ...]:
    """Discover one Stage 5 match artifact per adapter for one paper.

    Expected experiment layout::

        <root>/<adapter>/<run>/reconstruction/
            <paper>/<adapter>/references/reference_matches.json

    Discovery is deliberately strict. If more than one artifact is found
    for the same adapter, Tabulus refuses to choose among historical runs
    implicitly. The caller must then supply the desired artifacts
    explicitly.
    """

    root = Path(
        search_root
    ).expanduser()

    if not root.is_dir():
        raise FileNotFoundError(
            "Reference-match search root does not exist "
            f"or is not a directory: {root}"
        )

    paper = str(
        paper_name or ""
    ).strip()

    if not paper:
        raise ValueError(
            "paper_name must be a non-empty string."
        )

    by_adapter: dict[str, Path] = {}

    for path in sorted(
        root.rglob(
            "reference_matches.json"
        )
    ):
        parts = path.parts

        matched_adapter = None

        for position, part in enumerate(
            parts
        ):
            if part != "reconstruction":
                continue

            # Expected suffix:
            # reconstruction/<paper>/<adapter>/
            # references/reference_matches.json
            if position + 4 >= len(parts):
                continue

            if parts[position + 1] != paper:
                continue

            if parts[position + 3] != "references":
                continue

            if (
                parts[position + 4]
                != "reference_matches.json"
            ):
                continue

            matched_adapter = parts[
                position + 2
            ]
            break

        if matched_adapter is None:
            continue

        existing = by_adapter.get(
            matched_adapter
        )

        if (
            existing is not None
            and existing != path
        ):
            raise ValueError(
                "Multiple Stage 5 reference_matches.json "
                "artifacts were discovered for adapter "
                f"{matched_adapter!r} and paper {paper!r}: "
                f"{existing} ; {path}. "
                "Refusing to select a historical run "
                "implicitly."
            )

        by_adapter[
            matched_adapter
        ] = path

    if not by_adapter:
        raise FileNotFoundError(
            "No Stage 5 reference_matches.json artifacts "
            f"were found for paper {paper!r} under {root}."
        )

    return tuple(
        by_adapter[adapter]
        for adapter in sorted(
            by_adapter
        )
    )


def collect_resolution_targets(
    bibliography_path: Path,
    reference_matches_paths: Iterable[Path],
) -> tuple[ReferenceEvidence, ...]:
    """Return unique Stage 4 entries actually linked by Stage 5.

    Multiple Stage 5 artifacts may be supplied, for example when several table
    reconstruction adapters processed the same physical paper. Bibliography
    indices are unioned so the same scholarly reference is resolved only once.
    """

    bibliography = _load_bibliography_entries(
        Path(bibliography_path)
    )

    paths = tuple(
        Path(path).expanduser()
        for path in reference_matches_paths
    )

    if not paths:
        raise ValueError(
            "At least one reference_matches.json artifact "
            "must be supplied."
        )

    linked_indices: set[int] = set()

    for path in paths:
        linked_indices.update(
            _linked_indices_from_matches(path)
        )

    missing = sorted(
        index
        for index in linked_indices
        if index not in bibliography
    )

    if missing:
        raise ValueError(
            "Reference matches contain bibliography indices "
            "absent from bibliography.json: "
            + ", ".join(str(index) for index in missing)
        )

    return tuple(
        bibliography[index]
        for index in sorted(linked_indices)
    )


def retrieve_crossref_evidence(
    targets: Iterable[ReferenceEvidence],
    client: CrossrefRetriever,
) -> tuple[CrossrefRetrieval, ...]:
    """Retrieve Crossref evidence once for each unique Stage 6 target.

    Existing DOI values are checked first. A validated DOI avoids a
    bibliographic search. If an existing DOI cannot be found in Crossref, the
    raw citation is searched so that an incorrect or damaged DOI does not block
    recovery of the actual work.

    This function performs retrieval only. It does not automatically accept
    Crossref's first candidate and does not make final resolution decisions.
    """

    results: list[CrossrefRetrieval] = []
    seen_indices: set[int] = set()

    target_list = tuple(
        targets
    )

    for position, target in enumerate(
        target_list,
        start=1,
    ):
        LOGGER.info(
            "[Crossref] %d/%d ref=%d",
            position,
            len(target_list),
            target.reference_index,
        )
        if target.reference_index in seen_indices:
            continue

        seen_indices.add(target.reference_index)

        existing_doi = target.doi.strip()
        existing_candidate: ResolutionCandidate | None = None
        existing_checked = bool(existing_doi)

        if existing_doi:
            existing_candidate = client.get_by_doi(
                existing_doi
            )

        search_performed = False
        search_candidates: tuple[
            ResolutionCandidate, ...
        ] = ()

        if existing_candidate is None:
            search_performed = True
            search_candidates = client.search_bibliographic(
                target.raw_reference
            )

        results.append(
            CrossrefRetrieval(
                reference_index=target.reference_index,
                raw_reference=target.raw_reference,
                existing_doi=existing_doi,
                existing_doi_checked=existing_checked,
                existing_doi_candidate=existing_candidate,
                bibliographic_search_performed=search_performed,
                bibliographic_candidates=search_candidates,
            )
        )

    return tuple(results)
