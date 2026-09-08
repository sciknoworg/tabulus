from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import html
import json
from pathlib import Path
import re
from typing import Any


REFERENCE_CONTEXT_SCHEMA_VERSION = 1
DEFAULT_MAX_CONTEXTS_PER_REFERENCE = 3
MAX_EXPANDED_CITATION_RANGE = 200

_SUPERSCRIPT_RE = re.compile(
    r"<sup>(.*?)</sup>",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class ReferenceContext:
    """One body-text context in which a bibliography reference occurs."""

    page_index: int
    block_index: int
    block_type: str
    bbox: tuple[int | float, ...] | None
    citation_marker: str
    citation_count: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_index": self.page_index,
            "block_index": self.block_index,
            "block_type": self.block_type,
            "bbox": (
                list(self.bbox)
                if self.bbox is not None
                else None
            ),
            "citation_marker": self.citation_marker,
            "citation_count": self.citation_count,
            "text": self.text,
        }


@dataclass(frozen=True)
class ReferenceContextEntry:
    """All retained document contexts for one bibliography index."""

    reference_index: int
    contexts: tuple[ReferenceContext, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_index": self.reference_index,
            "context_count": len(self.contexts),
            "contexts": [
                context.to_dict()
                for context in self.contexts
            ],
        }


def _normalize_bbox(
    value: Any,
) -> tuple[int | float, ...] | None:
    if not isinstance(value, list):
        return None

    if len(value) != 4:
        return None

    if not all(
        isinstance(item, (int, float))
        and not isinstance(item, bool)
        for item in value
    ):
        return None

    return tuple(value)


def _paragraph_text(
    block: dict[str, Any],
) -> str:
    """Extract only actual textual content from a MinerU paragraph block."""

    content = block.get("content")

    if not isinstance(content, dict):
        return ""

    items = content.get(
        "paragraph_content"
    )

    if not isinstance(items, list):
        return ""

    parts: list[str] = []

    for item in items:
        if isinstance(item, str):
            value = item

        elif isinstance(item, dict):
            value = item.get(
                "content"
            )

        else:
            continue

        if isinstance(value, str):
            value = value.strip()

            if value:
                parts.append(value)

    return re.sub(
        r"\s+",
        " ",
        " ".join(parts),
    ).strip()


def _expand_numeric_marker(
    marker: str,
) -> set[int]:
    """Expand a numeric citation marker such as ``1,3-5``."""

    marker = html.unescape(
        marker
    )

    marker = re.sub(
        r"<[^>]+>",
        "",
        marker,
    )

    marker = (
        marker
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
    )

    numbers: set[int] = set()

    for part in re.split(
        r"[,;]\s*",
        marker,
    ):
        part = part.strip()

        if not part:
            continue

        range_match = re.fullmatch(
            r"(\d+)\s*-\s*(\d+)",
            part,
        )

        if range_match:
            start = int(
                range_match.group(1)
            )
            end = int(
                range_match.group(2)
            )

            if (
                start <= end
                and (
                    end - start
                    <= MAX_EXPANDED_CITATION_RANGE
                )
            ):
                numbers.update(
                    range(
                        start,
                        end + 1,
                    )
                )

            continue

        if re.fullmatch(
            r"\d+",
            part,
        ):
            numbers.add(
                int(part)
            )

    return numbers


def _clean_marker(
    marker: str,
) -> str:
    marker = html.unescape(
        marker
    )

    marker = re.sub(
        r"<[^>]+>",
        "",
        marker,
    )

    return re.sub(
        r"\s+",
        " ",
        marker,
    ).strip()


def extract_reference_contexts(
    mineru_content: Any,
    *,
    max_contexts_per_reference: int = (
        DEFAULT_MAX_CONTEXTS_PER_REFERENCE
    ),
) -> tuple[ReferenceContextEntry, ...]:
    """Extract ranked body-text citation contexts from MinerU content.

    Only paragraph blocks are considered. Reference-list ``list`` blocks,
    figures, tables, captions, and other non-body structures are therefore
    excluded by construction.

    Contexts are ranked conservatively by citation specificity: a context
    containing a single cited bibliography index outranks one containing a
    large citation group.
    """

    if max_contexts_per_reference <= 0:
        raise ValueError(
            "max_contexts_per_reference must be greater than zero."
        )

    if not isinstance(
        mineru_content,
        list,
    ):
        raise ValueError(
            "MinerU content_list_v2 must be a top-level list."
        )

    contexts_by_reference: dict[
        int,
        list[ReferenceContext],
    ] = defaultdict(list)

    for page_index, page in enumerate(
        mineru_content
    ):
        if not isinstance(page, list):
            continue

        for block_index, block in enumerate(
            page
        ):
            if not isinstance(block, dict):
                continue

            # Deliberately use body paragraphs only.
            if block.get("type") != "paragraph":
                continue

            text = _paragraph_text(
                block
            )

            if not text:
                continue

            # A target reference can occur more than once inside one
            # paragraph. Keep only the most specific marker for that
            # reference in that block.
            best_marker_by_reference: dict[
                int,
                tuple[int, str],
            ] = {}

            for raw_marker in _SUPERSCRIPT_RE.findall(
                text
            ):
                references = _expand_numeric_marker(
                    raw_marker
                )

                if not references:
                    continue

                citation_count = len(
                    references
                )

                cleaned_marker = _clean_marker(
                    raw_marker
                )

                for reference_index in references:
                    previous = (
                        best_marker_by_reference.get(
                            reference_index
                        )
                    )

                    candidate = (
                        citation_count,
                        cleaned_marker,
                    )

                    if (
                        previous is None
                        or candidate[0] < previous[0]
                    ):
                        best_marker_by_reference[
                            reference_index
                        ] = candidate

            bbox = _normalize_bbox(
                block.get("bbox")
            )

            for (
                reference_index,
                (
                    citation_count,
                    citation_marker,
                ),
            ) in best_marker_by_reference.items():
                contexts_by_reference[
                    reference_index
                ].append(
                    ReferenceContext(
                        page_index=page_index,
                        block_index=block_index,
                        block_type="paragraph",
                        bbox=bbox,
                        citation_marker=citation_marker,
                        citation_count=citation_count,
                        text=text,
                    )
                )

    entries: list[
        ReferenceContextEntry
    ] = []

    for reference_index in sorted(
        contexts_by_reference
    ):
        ranked = sorted(
            contexts_by_reference[
                reference_index
            ],
            key=lambda context: (
                context.citation_count,
                context.page_index,
                context.block_index,
            ),
        )

        entries.append(
            ReferenceContextEntry(
                reference_index=reference_index,
                contexts=tuple(
                    ranked[
                        :max_contexts_per_reference
                    ]
                ),
            )
        )

    return tuple(entries)


def load_mineru_reference_contexts(
    mineru_content_path: Path,
    *,
    max_contexts_per_reference: int = (
        DEFAULT_MAX_CONTEXTS_PER_REFERENCE
    ),
) -> tuple[ReferenceContextEntry, ...]:
    path = Path(
        mineru_content_path
    )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    return extract_reference_contexts(
        payload,
        max_contexts_per_reference=(
            max_contexts_per_reference
        ),
    )


def write_reference_context_artifact(
    entries: tuple[
        ReferenceContextEntry,
        ...
    ],
    artifact_root: Path,
    *,
    source_path: Path,
    max_contexts_per_reference: int = (
        DEFAULT_MAX_CONTEXTS_PER_REFERENCE
    ),
) -> Path:
    """Persist an auditable optional document-context artifact."""

    seen: set[int] = set()

    for entry in entries:
        if entry.reference_index in seen:
            raise ValueError(
                "Duplicate reference-context bibliography index "
                f"{entry.reference_index}."
            )

        seen.add(
            entry.reference_index
        )

    output_path = (
        Path(artifact_root)
        / "references"
        / "reference_context.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": (
            REFERENCE_CONTEXT_SCHEMA_VERSION
        ),
        "source_kind": (
            "mineru_content_list_v2"
        ),
        "source_path": str(
            Path(source_path)
        ),
        "max_contexts_per_reference": (
            max_contexts_per_reference
        ),
        "reference_count": len(
            entries
        ),
        "context_count": sum(
            len(entry.contexts)
            for entry in entries
        ),
        "entries": [
            entry.to_dict()
            for entry in entries
        ],
    }

    temporary_path = (
        output_path.with_suffix(
            output_path.suffix + ".tmp"
        )
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(
        output_path
    )

    return output_path


def extract_reference_context_artifact(
    mineru_content_path: Path,
    artifact_root: Path,
    *,
    max_contexts_per_reference: int = (
        DEFAULT_MAX_CONTEXTS_PER_REFERENCE
    ),
) -> Path:
    """Extract and write optional citation-context evidence."""

    entries = load_mineru_reference_contexts(
        mineru_content_path,
        max_contexts_per_reference=(
            max_contexts_per_reference
        ),
    )

    return write_reference_context_artifact(
        entries,
        artifact_root,
        source_path=mineru_content_path,
        max_contexts_per_reference=(
            max_contexts_per_reference
        ),
    )


def load_reference_context_artifact(
    path: Path,
) -> dict[
    int,
    tuple[ReferenceContext, ...],
]:
    """Load a persisted context artifact for optional Stage 6 use."""

    payload = json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )

    if payload.get(
        "schema_version"
    ) != REFERENCE_CONTEXT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported reference-context schema version."
        )

    raw_entries = payload.get(
        "entries"
    )

    if not isinstance(
        raw_entries,
        list,
    ):
        raise ValueError(
            "Reference-context artifact entries must be a list."
        )

    result: dict[
        int,
        tuple[ReferenceContext, ...],
    ] = {}

    for raw_entry in raw_entries:
        if not isinstance(
            raw_entry,
            dict,
        ):
            raise ValueError(
                "Reference-context entry must be an object."
            )

        reference_index = raw_entry.get(
            "reference_index"
        )

        if (
            not isinstance(
                reference_index,
                int,
            )
            or isinstance(
                reference_index,
                bool,
            )
            or reference_index <= 0
        ):
            raise ValueError(
                "Reference-context bibliography index "
                "must be a positive integer."
            )

        if reference_index in result:
            raise ValueError(
                "Duplicate reference-context bibliography index "
                f"{reference_index}."
            )

        raw_contexts = raw_entry.get(
            "contexts"
        )

        if not isinstance(
            raw_contexts,
            list,
        ):
            raise ValueError(
                "Reference-context contexts must be a list."
            )

        contexts: list[
            ReferenceContext
        ] = []

        for raw_context in raw_contexts:
            if not isinstance(
                raw_context,
                dict,
            ):
                raise ValueError(
                    "Reference context must be an object."
                )

            page_index = raw_context.get(
                "page_index"
            )
            block_index = raw_context.get(
                "block_index"
            )
            citation_count = raw_context.get(
                "citation_count"
            )

            if not all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                for value in (
                    page_index,
                    block_index,
                )
            ):
                raise ValueError(
                    "Reference context page/block indices "
                    "must be non-negative integers."
                )

            if (
                not isinstance(
                    citation_count,
                    int,
                )
                or isinstance(
                    citation_count,
                    bool,
                )
                or citation_count <= 0
            ):
                raise ValueError(
                    "Reference context citation_count "
                    "must be a positive integer."
                )

            bbox = _normalize_bbox(
                raw_context.get(
                    "bbox"
                )
            )

            contexts.append(
                ReferenceContext(
                    page_index=page_index,
                    block_index=block_index,
                    block_type=str(
                        raw_context.get(
                            "block_type"
                        )
                        or ""
                    ),
                    bbox=bbox,
                    citation_marker=str(
                        raw_context.get(
                            "citation_marker"
                        )
                        or ""
                    ),
                    citation_count=citation_count,
                    text=str(
                        raw_context.get(
                            "text"
                        )
                        or ""
                    ),
                )
            )

        result[
            reference_index
        ] = tuple(
            contexts
        )

    return result
