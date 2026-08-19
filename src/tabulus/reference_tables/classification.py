from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence


REFERENCE_TABLE_CLASSIFICATION_NAME = "reference_table_classification.json"

TAG_PATTERN = re.compile(
    r"\b("
    r"refs?\.?|references?|"
    r"auth(?:\b|ors?\b)|"
    r"years?|"
    r"sources?|"
    r"papers?|"
    r"citations?|"
    r"literatures?|"
    r"works?|"
    r"comparative\s+works?|"
    r"research(?:es)?|"
    r"datasets?|data\s*sets?|"
    r"data\s*set(?:\s*(?:name|naem))?|"
    r"dataset(?:\s*(?:name|naem))?|"
    r"publications?|pubications?|"
    r"contributions?|"
    r"sample\s+articles?|"
    r"stud(?:y|ies)(?!\s*area)"
    r")\b",
    re.IGNORECASE,
)

CITATION_PATTERN = re.compile(
    r"("
    r"\[\s*\d{1,4}(?:\s*[-–]\s*\d{1,4})?"
    r"(?:\s*,\s*\d{1,4}(?:\s*[-–]\s*\d{1,4})?)*\s*\]"
    r"|"
    r"\(\s*\d{1,4}(?:\s*[-–]\s*\d{1,4})?"
    r"(?:\s*,\s*\d{1,4}(?:\s*[-–]\s*\d{1,4})?)*\s*\)"
    r"|"
    r"\b[A-Z][A-Za-z'`\-]+(?:\s+[A-Z][A-Za-z'`\-]+)?"
    r"\s+et\s+al\.?.*?(?:19|20)\d{2}[a-z]?\b"
    r"|"
    r"\b[A-Z][A-Za-z'`\-]+,\s*(?:19|20)\d{2}[a-z]?\b"
    r"|"
    r"\bdoi\s*:\s*10\.\S+\b"
    r"|"
    r"\b10\.\S+\b"
    r"|"
    r"\b[A-Z][A-Za-z'`\-]+\s+and\s+[A-Z][A-Za-z'`\-]+"
    r".*?(?:19|20)\d{2}[a-z]?\b"
    r")",
    re.IGNORECASE,
)

# Scientific tables commonly use a "Refs." column whose cells contain bare
# reference numbers such as "85", "88 and 89", or "83, 90, and 91".  Bare
# numbers are treated as citation evidence only inside a column whose first
# non-empty cell is itself reference-like, avoiding a broad numeric heuristic.
PLAIN_NUMERIC_REFERENCE_PATTERN = re.compile(
    r"^\s*"
    r"\d{1,4}(?:\s*[-–]\s*\d{1,4})?"
    r"(?:"
    r"(?:\s*[,;]\s*(?:and\s+)?|\s+and\s+|\s*&\s*)"
    r"\d{1,4}(?:\s*[-–]\s*\d{1,4})?"
    r")*"
    r"\s*$",
    re.IGNORECASE,
)


CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x1f\x7f-\x9f]+")

# Continuation handling is intentionally separate from the legacy reference
# classification heuristics above. These patterns only identify an explicit
# continued-table caption and its printed table identifier.
TABLE_LABEL_PATTERN = re.compile(
    r"""
    ^\s*
    (?:(?:supplementary|supplemental|appendix)\s+)?
    (?:table|tbl\.?|tab\.?)\s*
    (?P<label>
        # Labels containing digits: 1, 1A, 1-A, 2.1, S1, S-1, A.1, etc.
        (?:[A-Za-z]+\s*[-._]?\s*)?
        \d+
        (?:\s*[._-]\s*\d+)*
        (?:\s*[-._]?\s*[A-Za-z])?
        |
        # Roman numerals: I, II, IV, XII, ...
        [IVXLCDM]+
        |
        # Alphabetic appendix-style labels: A, B, C, ...
        [A-Za-z]
    )
    (?=$|[\s.,:;()\[\]\-–—])
    """,
    re.IGNORECASE | re.VERBOSE,
)

CONTINUATION_AFTER_LABEL_PATTERN = re.compile(
    r"""
    ^[\s.,:;()\[\]\-–—]*
    (?:
        continued
        | continuation
        | contd
        | cont['’]d
        | cont
    )
    (?=$|[\s.,:;()\[\]\-–—])
    """,
    re.IGNORECASE | re.VERBOSE,
)

CONTINUATION_ONLY_PATTERN = re.compile(
    r"""
    ^[\s.,:;()\[\]\-–—]*
    (?:
        continued
        | continuation
        | contd
        | cont['’]d
        | cont
    )
    [\s.,:;()\[\]\-–—]*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class _ContinuationLink:
    parent_table_id: int
    caption: str


@dataclass(frozen=True)
class ReferenceTableDecision:
    """Evidence-backed decision for one reconstructed table."""

    is_reference_table: bool
    has_tag_match: bool
    has_citation_match: bool
    matched_header_cells: tuple[str, ...]
    matched_citation_cells: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_reference_table": self.is_reference_table,
            "has_tag_match": self.has_tag_match,
            "has_citation_match": self.has_citation_match,
            "matched_header_cells": list(self.matched_header_cells),
            "matched_citation_cells": list(self.matched_citation_cells),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ReferenceTableClassification:
    """Classification plus provenance for one physical MinerU table."""

    table_id: int
    source_status: str
    source_parsed: str | None
    source_prediction: str | None
    parsed_tables: int
    decision: ReferenceTableDecision
    independent_is_reference_table: bool
    classification_source: str = "heuristic"
    continued_from_table_id: int | None = None
    continuation_caption: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "source_status": self.source_status,
            "source_parsed": self.source_parsed,
            "source_prediction": self.source_prediction,
            "parsed_tables": self.parsed_tables,
            "independent_is_reference_table": (
                self.independent_is_reference_table
            ),
            "classification_source": self.classification_source,
            "continued_from_table_id": self.continued_from_table_id,
            "continuation_caption": self.continuation_caption,
            **self.decision.to_dict(),
        }


@dataclass(frozen=True)
class ReferenceTableClassificationResult:
    """Persisted classification manifest for one reconstruction batch."""

    adapter_name: str | None
    reconstruction_dir: Path
    tables_considered: int
    reference_tables_found: int
    classifications: list[ReferenceTableClassification]
    output_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "reconstruction_dir": str(self.reconstruction_dir),
            "tables_considered": self.tables_considered,
            "reference_tables_found": self.reference_tables_found,
            "tables": [
                classification.to_dict()
                for classification in self.classifications
            ],
        }


def normalize_text(value: Any) -> str:
    """Collapse whitespace in a table cell while preserving its text."""

    return re.sub(r"\s+", " ", str(value)).strip()


def get_first_non_empty_cells_by_column(
    rows: Sequence[Sequence[Any]],
) -> list[str]:
    """Return the first non-empty cell encountered in every table column."""

    if not rows:
        return []

    max_cols = max((len(row) for row in rows), default=0)
    first_non_empty_cells: list[str] = []

    for col_idx in range(max_cols):
        first_value = ""

        for row in rows:
            cell = row[col_idx] if col_idx < len(row) else ""
            normalized = normalize_text(cell)

            if normalized:
                first_value = normalized
                break

        first_non_empty_cells.append(first_value)

    return first_non_empty_cells


def _append_unique(values: list[str], value: str, *, limit: int = 5) -> None:
    if value not in values and len(values) < limit:
        values.append(value)


def classify_reference_like_table(
    rows: Sequence[Sequence[Any]],
) -> ReferenceTableDecision:
    """
    Classify a reconstructed table using legacy-compatible reference evidence.

    The classifier keeps the legacy header/citation patterns and adds a
    conservative case for bare numeric reference lists in explicitly tagged
    reference columns. A reference-like header alone is not sufficient.
    """

    if not rows:
        return ReferenceTableDecision(
            is_reference_table=False,
            has_tag_match=False,
            has_citation_match=False,
            matched_header_cells=(),
            matched_citation_cells=(),
            reason="No rows available.",
        )

    normalized_rows = [
        [normalize_text(cell) for cell in row]
        for row in rows
    ]
    first_non_empty_cells = get_first_non_empty_cells_by_column(
        normalized_rows
    )

    tagged_columns = [
        col_idx
        for col_idx, cell in enumerate(first_non_empty_cells)
        if cell and TAG_PATTERN.search(cell)
    ]
    matched_header_cells = [
        first_non_empty_cells[col_idx]
        for col_idx in tagged_columns
    ]

    matched_citation_cells: list[str] = []

    for row in normalized_rows:
        for cell in row:
            if cell and CITATION_PATTERN.search(cell):
                _append_unique(matched_citation_cells, cell)

    for row in normalized_rows:
        for col_idx in tagged_columns:
            cell = row[col_idx] if col_idx < len(row) else ""

            if (
                re.search(r"\b(?:refs?|references?|citations?)\b", first_non_empty_cells[col_idx], re.IGNORECASE)
                and cell != first_non_empty_cells[col_idx]
                and PLAIN_NUMERIC_REFERENCE_PATTERN.fullmatch(cell)
            ):
                _append_unique(matched_citation_cells, cell)

    has_tag_match = bool(matched_header_cells)
    has_citation_match = bool(matched_citation_cells)

    if has_tag_match and has_citation_match:
        is_reference_table = True
        reason = (
            "Header-like reference tags and citation-like cell content found."
        )
    elif not has_tag_match and has_citation_match:
        is_reference_table = True
        reason = (
            "No header tag found, but citation-like cell content found."
        )
    elif has_tag_match:
        is_reference_table = False
        reason = (
            "Header-like tags found, but no citation-like content found."
        )
    else:
        is_reference_table = False
        reason = "No reference-like evidence found."

    return ReferenceTableDecision(
        is_reference_table=is_reference_table,
        has_tag_match=has_tag_match,
        has_citation_match=has_citation_match,
        matched_header_cells=tuple(matched_header_cells[:5]),
        matched_citation_cells=tuple(matched_citation_cells[:5]),
        reason=reason,
    )


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error

    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")

    return value


def _resolve_artifact_path(
    reconstruction_dir: Path,
    value: Any,
) -> Path | None:
    if not isinstance(value, str) or not value:
        return None

    path = Path(value)

    if path.is_absolute():
        return path

    return reconstruction_dir / path


def _unavailable_decision(reason: str) -> ReferenceTableDecision:
    return ReferenceTableDecision(
        is_reference_table=False,
        has_tag_match=False,
        has_citation_match=False,
        matched_header_cells=(),
        matched_citation_cells=(),
        reason=reason,
    )


def _caption_text(value: Any) -> str:
    """Normalize MinerU caption structures into one compact text string."""

    if isinstance(value, str):
        cleaned = CONTROL_CHAR_PATTERN.sub(" ", value)
        return normalize_text(cleaned)

    if isinstance(value, (list, tuple)):
        parts = [_caption_text(item) for item in value]
        return normalize_text(" ".join(part for part in parts if part))

    if isinstance(value, dict):
        for key in ("text", "content", "caption"):
            if key in value:
                text = _caption_text(value[key])

                if text:
                    return text

        parts = [_caption_text(item) for item in value.values()]
        return normalize_text(" ".join(part for part in parts if part))

    return ""


def _canonicalize_table_label(label: str) -> str:
    """
    Normalize printed table identifiers without collapsing numeric hierarchy.

    Examples:
    ``a`` -> ``A``, ``S 1`` -> ``S1``, ``A-1`` -> ``A1``,
    while ``2.1`` remains ``2.1`` rather than becoming ``21``.
    """

    value = re.sub(r"\s+", "", label).upper()
    value = re.sub(r"(?<=[A-Z])[._-](?=\d)", "", value)
    value = re.sub(r"(?<=\d)[._-](?=[A-Z])", "", value)
    return value


def _table_label(caption: str) -> str | None:
    match = TABLE_LABEL_PATTERN.match(caption)

    if match is None:
        return None

    return _canonicalize_table_label(match.group("label"))


def _is_explicit_continuation_caption(caption: str) -> bool:
    """
    Return whether a caption explicitly marks this physical table as continued.

    For labeled captions, the continuation token must occur immediately after
    the printed table identifier (apart from punctuation/whitespace). This
    avoids treating captions that merely mention another table or use the word
    "continued" later in descriptive prose as continuation evidence.
    """

    if not caption:
        return False

    label_match = TABLE_LABEL_PATTERN.match(caption)

    if label_match is not None:
        remainder = caption[label_match.end():]
        return bool(CONTINUATION_AFTER_LABEL_PATTERN.match(remainder))

    return bool(CONTINUATION_ONLY_PATTERN.fullmatch(caption))


def _resolve_crop_index_path(
    reconstruction_dir: Path,
    summary: dict[str, Any],
) -> Path | None:
    candidates: list[Path] = []
    crop_root_value = summary.get("crop_root")

    if isinstance(crop_root_value, str) and crop_root_value:
        candidates.append(Path(crop_root_value))

    conventional_root = reconstruction_dir.parent.parent

    if conventional_root not in candidates:
        candidates.append(conventional_root)

    for crop_root in candidates:
        index_path = crop_root / "tables_index.json"

        if index_path.is_file():
            return index_path

    return None


def _load_continuation_links(
    reconstruction_dir: Path,
    summary: dict[str, Any],
) -> dict[int, _ContinuationLink]:
    """
    Resolve explicit continued-table relationships from tables_index.json.

    A continuation is inferred only from an explicit continuation marker in the
    current caption. Printed identifiers are normalized across common forms
    such as Table 1, Table A, Table I, Table S1, and hierarchical identifiers.
    When a label is present, the nearest earlier physical table with the same
    normalized label is used as its parent. An unlabeled "Continued" caption
    inherits only from the immediately preceding physical table.
    """

    index_path = _resolve_crop_index_path(reconstruction_dir, summary)

    if index_path is None:
        return {}

    index = _load_json_object(
        index_path,
        label="Canonical table-crop index",
    )
    records = index.get("tables")

    if not isinstance(records, list):
        raise ValueError(
            "Canonical table-crop index has no valid 'tables' list: "
            f"{index_path}"
        )

    previous_records: list[tuple[int, str]] = []
    links: dict[int, _ContinuationLink] = {}
    seen_table_ids: set[int] = set()

    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(
                "Canonical table-crop index contains a non-object table at "
                f"position {position}."
            )

        table_id = record.get("table_id")

        if not isinstance(table_id, int):
            raise ValueError(
                "Canonical table-crop index contains an invalid table_id at "
                f"position {position}."
            )

        if table_id in seen_table_ids:
            raise ValueError(
                f"Duplicate table_id in canonical table-crop index: {table_id}"
            )

        seen_table_ids.add(table_id)
        caption = _caption_text(record.get("table_caption"))

        if _is_explicit_continuation_caption(caption) and previous_records:
            label = _table_label(caption)
            parent_table_id: int | None = None

            if label is not None:
                for earlier_table_id, earlier_caption in reversed(
                    previous_records
                ):
                    if _table_label(earlier_caption) == label:
                        parent_table_id = earlier_table_id
                        break
            else:
                # An explicitly unlabeled "Continued" caption can only inherit
                # from the immediately preceding physical table.
                parent_table_id = previous_records[-1][0]

            if parent_table_id is not None:
                links[table_id] = _ContinuationLink(
                    parent_table_id=parent_table_id,
                    caption=caption,
                )

        previous_records.append((table_id, caption))

    return links


def _apply_continuation_inheritance(
    classifications: list[ReferenceTableClassification],
    links: dict[int, _ContinuationLink],
) -> list[ReferenceTableClassification]:
    """
    Propagate a positive reference decision through explicit continuations.

    Independent heuristic evidence is retained separately so evaluation can
    distinguish directly detected positives from inherited continuation
    positives. Continuation chains are resolved in physical table order.
    """

    resolved: list[ReferenceTableClassification] = []
    by_table_id: dict[int, ReferenceTableClassification] = {}

    for classification in classifications:
        link = links.get(classification.table_id)
        updated = classification

        if link is not None:
            updated = replace(
                updated,
                continued_from_table_id=link.parent_table_id,
                continuation_caption=link.caption,
            )

            parent = by_table_id.get(link.parent_table_id)

            if (
                parent is not None
                and parent.decision.is_reference_table
                and not updated.decision.is_reference_table
            ):
                updated = replace(
                    updated,
                    decision=replace(
                        updated.decision,
                        is_reference_table=True,
                        reason=(
                            "Explicit continuation of reference table "
                            f"{link.parent_table_id}; reference classification "
                            "inherited."
                        ),
                    ),
                    classification_source="continued_table",
                )

        resolved.append(updated)
        by_table_id[updated.table_id] = updated

    return resolved


def classify_reconstruction_tables(
    reconstruction_dir: Path,
    *,
    output_path: Path | None = None,
) -> ReferenceTableClassificationResult:
    """
    Classify every physical table recorded by a reconstruction batch.

    The stage reads the common ``parsed/`` representation and never modifies
    ``predictions/*.csv``. After independent heuristic classification, explicit
    continued-table relationships from ``tables_index.json`` propagate positive
    reference decisions forward. The output is written only after the complete
    input manifest has been processed successfully.
    """

    reconstruction_dir = Path(reconstruction_dir)
    summary_path = reconstruction_dir / "batch_summary.json"
    summary = _load_json_object(
        summary_path,
        label="Table reconstruction batch summary",
    )

    items = summary.get("items")

    if not isinstance(items, list):
        raise ValueError(
            "Table reconstruction batch summary has no valid 'items' list: "
            f"{summary_path}"
        )

    classifications: list[ReferenceTableClassification] = []
    seen_table_ids: set[int] = set()

    for position, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                "Table reconstruction batch summary contains a non-object "
                f"item at position {position}."
            )

        table_id = item.get("table_id")

        if not isinstance(table_id, int):
            raise ValueError(
                "Table reconstruction batch summary contains an invalid "
                f"table_id at position {position}."
            )

        if table_id in seen_table_ids:
            raise ValueError(
                "Duplicate table_id in reconstruction batch summary: "
                f"{table_id}"
            )

        seen_table_ids.add(table_id)

        source_status = str(item.get("status", "unknown"))
        source_parsed_value = item.get("parsed_result")
        source_prediction_value = item.get("prediction_csv")
        parsed_path = _resolve_artifact_path(
            reconstruction_dir,
            source_parsed_value,
        )

        parsed_count = 0

        if parsed_path is None or not parsed_path.is_file():
            decision = _unavailable_decision(
                "Parsed reconstruction is unavailable."
            )
        else:
            parsed_payload = _load_json_object(
                parsed_path,
                label=f"Parsed reconstruction for table {table_id}",
            )

            parsed_table_id = parsed_payload.get("table_id")

            if parsed_table_id != table_id:
                raise ValueError(
                    "Parsed reconstruction table identity does not match "
                    f"batch summary: expected {table_id}, "
                    f"found {parsed_table_id!r}."
                )

            parsed_tables = parsed_payload.get("tables")

            if not isinstance(parsed_tables, list):
                raise ValueError(
                    "Parsed reconstruction has no valid 'tables' list: "
                    f"{parsed_path}"
                )

            parsed_count = len(parsed_tables)

            if parsed_count == 0:
                decision = _unavailable_decision(
                    "No structured table was parsed for classification."
                )
            elif parsed_count > 1:
                decision = _unavailable_decision(
                    "Multiple structured tables were parsed from one crop; "
                    "reference-table classification is ambiguous."
                )
            elif not isinstance(parsed_tables[0], dict):
                raise ValueError(
                    "Parsed reconstruction table entry must be an object: "
                    f"{parsed_path}"
                )
            else:
                rows = parsed_tables[0].get("rows")

                if not isinstance(rows, list):
                    raise ValueError(
                        "Parsed reconstruction table has no valid 'rows' list: "
                        f"{parsed_path}"
                    )

                decision = classify_reference_like_table(rows)

        classifications.append(
            ReferenceTableClassification(
                table_id=table_id,
                source_status=source_status,
                source_parsed=(
                    str(source_parsed_value)
                    if isinstance(source_parsed_value, str)
                    else None
                ),
                source_prediction=(
                    str(source_prediction_value)
                    if isinstance(source_prediction_value, str)
                    else None
                ),
                parsed_tables=parsed_count,
                decision=decision,
                independent_is_reference_table=(
                    decision.is_reference_table
                ),
            )
        )

    continuation_links = _load_continuation_links(
        reconstruction_dir,
        summary,
    )
    classifications = _apply_continuation_inheritance(
        classifications,
        continuation_links,
    )

    final_output_path = (
        Path(output_path)
        if output_path is not None
        else reconstruction_dir / REFERENCE_TABLE_CLASSIFICATION_NAME
    )

    result = ReferenceTableClassificationResult(
        adapter_name=(
            str(summary["adapter_name"])
            if summary.get("adapter_name") is not None
            else None
        ),
        reconstruction_dir=reconstruction_dir,
        tables_considered=len(classifications),
        reference_tables_found=sum(
            item.decision.is_reference_table
            for item in classifications
        ),
        classifications=classifications,
        output_path=final_output_path,
    )

    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    final_output_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return result
