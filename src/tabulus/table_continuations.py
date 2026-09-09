from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence


CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x1f\x7f-\x9f]+")

TABLE_LABEL_PATTERN = re.compile(
    r"""
    ^\s*
    (?:(?:supplementary|supplemental|appendix)\s+)?
    (?:table|tbl\.?|tab\.?)\s*
    (?P<label>
        # Labels containing digits:
        # 1, 1A, 1-A, 2.1, S1, S-1, A.1, ...
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
class ContinuationLink:
    """Resolved symbolic relationship between two physical table crops."""

    parent_table_id: int
    root_table_id: int
    caption: str
    printed_table_label: str | None


def normalize_text(value: Any) -> str:
    """Collapse whitespace while preserving textual content."""

    return re.sub(r"\s+", " ", str(value)).strip()


def caption_text(value: Any) -> str:
    """Normalize MinerU caption structures into one compact string."""

    if isinstance(value, str):
        cleaned = CONTROL_CHAR_PATTERN.sub(" ", value)
        return normalize_text(cleaned)

    if isinstance(value, (list, tuple)):
        parts = [caption_text(item) for item in value]
        return normalize_text(
            " ".join(part for part in parts if part)
        )

    if isinstance(value, dict):
        for key in ("text", "content", "caption"):
            if key in value:
                text = caption_text(value[key])

                if text:
                    return text

        parts = [caption_text(item) for item in value.values()]
        return normalize_text(
            " ".join(part for part in parts if part)
        )

    return ""


def _table_caption_evidence(value: Any) -> str:
    """
    Return the caption fragment that carries the physical table label.

    MinerU can attach multiple caption fragments to one detected table. Keep
    label matching anchored at the beginning of an individual fragment so a
    preceding figure caption does not hide the table label and descriptive
    mentions of other tables are not mistaken for the table's own label.
    """

    if isinstance(value, (list, tuple)):
        for item in value:
            candidate = caption_text(item)

            if (
                TABLE_LABEL_PATTERN.match(candidate)
                or CONTINUATION_ONLY_PATTERN.fullmatch(candidate)
            ):
                return candidate

    return caption_text(value)


def canonicalize_table_label(label: str) -> str:
    """
    Normalize printed identifiers without collapsing numeric hierarchy.

    Examples:
    ``a`` -> ``A``
    ``S 1`` -> ``S1``
    ``A-1`` -> ``A1``
    ``2.1`` -> ``2.1``
    """

    value = re.sub(r"\s+", "", label).upper()
    value = re.sub(r"(?<=[A-Z])[._-](?=\d)", "", value)
    value = re.sub(r"(?<=\d)[._-](?=[A-Z])", "", value)
    return value


def table_label(caption: str) -> str | None:
    """Return the normalized printed table identifier, when present."""

    match = TABLE_LABEL_PATTERN.match(caption)

    if match is None:
        return None

    return canonicalize_table_label(match.group("label"))


def is_explicit_continuation_caption(caption: str) -> bool:
    """
    Return whether a caption explicitly marks a continued table.

    For a labeled caption, the continuation marker must occur
    immediately after the printed table identifier, apart from
    punctuation and whitespace. Descriptive uses of "continued"
    elsewhere in a caption are not treated as continuation evidence.
    """

    if not caption:
        return False

    label_match = TABLE_LABEL_PATTERN.match(caption)

    if label_match is not None:
        remainder = caption[label_match.end():]
        return bool(
            CONTINUATION_AFTER_LABEL_PATTERN.match(remainder)
        )

    return bool(
        CONTINUATION_ONLY_PATTERN.fullmatch(caption)
    )


def continuation_metadata_for_records(
    records: Sequence[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """
    Build deterministic continuation metadata in physical table order.

    A labeled continuation links to the nearest earlier physical table
    with the same normalized printed identifier. An explicitly
    unlabeled continuation links only to the immediately preceding
    record.

    Every physical table receives a continuation_root_table_id. For a
    standalone/root table this is its own table_id. Continuation chains
    retain both the immediate parent and the root table identifier.
    """

    previous: list[tuple[int, str, str | None]] = []
    roots: dict[int, int] = {}
    metadata: dict[int, dict[str, Any]] = {}
    seen_table_ids: set[int] = set()

    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(
                "Table index contains a non-object table at "
                f"position {position}."
            )

        table_id = record.get("table_id")

        if not isinstance(table_id, int):
            raise ValueError(
                "Table index contains an invalid table_id at "
                f"position {position}."
            )

        if table_id in seen_table_ids:
            raise ValueError(
                f"Duplicate table_id in table index: {table_id}"
            )

        seen_table_ids.add(table_id)

        caption = _table_caption_evidence(
            record.get("table_caption")
        )
        label = table_label(caption)
        explicit = is_explicit_continuation_caption(caption)

        parent_table_id: int | None = None

        if explicit and previous:
            if label is not None:
                for (
                    earlier_table_id,
                    _earlier_caption,
                    earlier_label,
                ) in reversed(previous):
                    if earlier_label == label:
                        parent_table_id = earlier_table_id
                        break
            else:
                parent_table_id = previous[-1][0]

        if parent_table_id is not None:
            root_table_id = roots.get(
                parent_table_id,
                parent_table_id,
            )
            link_status = "linked"
        elif explicit:
            root_table_id = table_id
            link_status = "unresolved"
        else:
            root_table_id = table_id
            link_status = "not_continuation"

        roots[table_id] = root_table_id

        metadata[table_id] = {
            "is_continuation": explicit,
            "continued_from_table_id": parent_table_id,
            "continuation_root_table_id": root_table_id,
            "printed_table_label": label,
            "evidence": (
                "explicit_caption"
                if explicit
                else None
            ),
            "link_status": link_status,
        }

        previous.append((table_id, caption, label))

    return metadata


def annotate_continuations(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Return table records enriched with normalized continuation metadata.

    Input records are not modified.
    """

    metadata = continuation_metadata_for_records(records)
    annotated: list[dict[str, Any]] = []

    for record in records:
        table_id = record["table_id"]
        updated = dict(record)
        updated["continuation"] = metadata[table_id]
        annotated.append(updated)

    return annotated


def _validate_structured_continuation(
    table_id: int,
    value: Any,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            "Structured continuation metadata must be an object "
            f"for table {table_id}."
        )

    is_continuation = value.get("is_continuation")
    parent = value.get("continued_from_table_id")
    root = value.get("continuation_root_table_id")
    label = value.get("printed_table_label")
    evidence = value.get("evidence")
    link_status = value.get("link_status")

    if not isinstance(is_continuation, bool):
        raise ValueError(
            "Invalid continuation is_continuation value for "
            f"table {table_id}."
        )

    if parent is not None and not isinstance(parent, int):
        raise ValueError(
            "Invalid continued_from_table_id for "
            f"table {table_id}."
        )

    if not isinstance(root, int):
        raise ValueError(
            "Invalid continuation_root_table_id for "
            f"table {table_id}."
        )

    if label is not None and not isinstance(label, str):
        raise ValueError(
            "Invalid printed_table_label for "
            f"table {table_id}."
        )

    if evidence is not None and evidence != "explicit_caption":
        raise ValueError(
            f"Invalid continuation evidence for table {table_id}."
        )

    if link_status not in {
        "linked",
        "unresolved",
        "not_continuation",
    }:
        raise ValueError(
            f"Invalid continuation link_status for table {table_id}."
        )

    if link_status == "linked":
        if not is_continuation or parent is None:
            raise ValueError(
                "Linked continuation metadata is inconsistent for "
                f"table {table_id}."
            )
    elif link_status == "unresolved":
        if not is_continuation or parent is not None:
            raise ValueError(
                "Unresolved continuation metadata is inconsistent "
                f"for table {table_id}."
            )
    else:
        if is_continuation or parent is not None:
            raise ValueError(
                "Non-continuation metadata is inconsistent for "
                f"table {table_id}."
            )

    return value


def continuation_links_from_records(
    records: Sequence[dict[str, Any]],
) -> dict[int, ContinuationLink]:
    """
    Return linked continuation relationships from a table index.

    New Step 1 indexes use the structured ``continuation`` field.
    Older indexes remain supported by deterministically reconstructing
    the same relationships from the preserved MinerU captions.
    """

    # Also validates table IDs/order and provides legacy fallback.
    fallback = continuation_metadata_for_records(records)

    links: dict[int, ContinuationLink] = {}

    for record in records:
        table_id = record["table_id"]

        if "continuation" in record:
            metadata = _validate_structured_continuation(
                table_id,
                record["continuation"],
            )
        else:
            metadata = fallback[table_id]

        parent_table_id = metadata[
            "continued_from_table_id"
        ]

        if parent_table_id is None:
            continue

        links[table_id] = ContinuationLink(
            parent_table_id=parent_table_id,
            root_table_id=metadata[
                "continuation_root_table_id"
            ],
            caption=caption_text(
                record.get("table_caption")
            ),
            printed_table_label=metadata[
                "printed_table_label"
            ],
        )

    return links
