from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from tabulus.models import TableRegion


_REFERENCE_HEADINGS = (
    re.compile(r"^\s*references\s*$", re.IGNORECASE),
    re.compile(r"^\s*bibliography\s*$", re.IGNORECASE),
    re.compile(r"^\s*literaturverzeichnis\s*$", re.IGNORECASE),
    re.compile(r"^\s*quellen\s*$", re.IGNORECASE),
    re.compile(r"^\s*referenzen\s*$", re.IGNORECASE),
)


def find_content_list(mineru_output_dir: Path) -> Path:
    """Find the MinerU *_content_list.json file for one document."""

    mineru_output_dir = Path(mineru_output_dir)

    matches = sorted(
        mineru_output_dir.rglob("*_content_list.json")
    )

    if not matches:
        raise FileNotFoundError(
            f"No *_content_list.json found under {mineru_output_dir}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            "Expected one MinerU document output, but found multiple "
            f"content lists: {matches}"
        )

    return matches[0]


def load_content_items(content_list_path: Path) -> list[dict[str, Any]]:
    """Load MinerU's flat content-list representation."""

    data = json.loads(
        Path(content_list_path).read_text(encoding="utf-8")
    )

    if isinstance(data, list):
        return [
            item
            for item in data
            if isinstance(item, dict)
        ]

    if isinstance(data, dict):
        for key in ("content_list", "items", "data"):
            items = data.get(key)

            if isinstance(items, list):
                return [
                    item
                    for item in items
                    if isinstance(item, dict)
                ]

    raise ValueError(
        f"Unsupported MinerU content-list structure: {content_list_path}"
    )


def resolve_image_path(
    content_list_path: Path,
    img_path: str,
) -> Path:
    """Resolve an image path emitted by MinerU."""

    content_list_path = Path(content_list_path)
    candidate = Path(img_path)

    if candidate.is_absolute() and candidate.is_file():
        return candidate

    relative = content_list_path.parent / candidate

    if relative.is_file():
        return relative

    matches = sorted(
        content_list_path.parent.rglob(candidate.name)
    )

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not resolve MinerU image path: {img_path}"
    )


def _item_text(item: dict[str, Any]) -> str:
    for key in ("text", "content", "raw_text", "title"):
        value = item.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def find_references_start_page(
    items: list[dict[str, Any]],
) -> int | None:
    """
    Return the one-based page where the references section starts.

    The final matching heading is used to avoid earlier textual mentions
    of "References" or "Bibliography".
    """

    last_match: int | None = None

    for item in items:
        page_idx = item.get("page_idx")

        if not isinstance(page_idx, int):
            continue

        text = _item_text(item)

        if not text:
            continue

        heading_like = (
            item.get("type") in {"title", "heading"}
            or (len(text) <= 40 and "\n" not in text)
        )

        if not heading_like:
            continue

        if any(pattern.match(text) for pattern in _REFERENCE_HEADINGS):
            last_match = page_idx + 1

    return last_match


def discover_tables(
    mineru_output_dir: Path,
    *,
    skip_missing_img_path: bool = False,
) -> tuple[list[TableRegion], int | None]:
    """
    Discover table regions from an existing MinerU result.

    This function does not run MinerU and does not modify any files.
    """

    content_list_path = find_content_list(mineru_output_dir)
    items = load_content_items(content_list_path)

    refs_start_page = find_references_start_page(items)

    table_items = [
        item
        for item in items
        if item.get("type") == "table"
    ]

    tables: list[TableRegion] = []

    for table_id, item in enumerate(table_items, start=1):
        page_idx = item.get("page_idx")
        page_nr = (
            page_idx + 1
            if isinstance(page_idx, int)
            else None
        )

        img_path = item.get("img_path")

        if not isinstance(img_path, str) or not img_path.strip():
            if skip_missing_img_path:
                continue
            raise ValueError(
                f"Table {table_id} has no valid img_path"
            )

        source_image = resolve_image_path(
            content_list_path,
            img_path,
        )

        tables.append(
            TableRegion(
                table_id=table_id,
                page_nr=page_nr,
                image_path=source_image,
                source_image_path=source_image,
                mineru_img_path=img_path,
                bbox=item.get("bbox"),
                caption=item.get("table_caption") or [],
                footnote=item.get("table_footnote") or [],
                mineru_table_body=item.get("table_body"),
                in_references=bool(
                    refs_start_page
                    and page_nr
                    and page_nr >= refs_start_page
                ),
            )
        )

    return tables, refs_start_page
