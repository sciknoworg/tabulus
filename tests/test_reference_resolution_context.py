from __future__ import annotations

import json

import pytest

from tabulus.reference_resolution.reference_context import (
    extract_reference_context_artifact,
    extract_reference_contexts,
    load_reference_context_artifact,
)


def _paragraph(
    text: str,
    *,
    bbox=None,
):
    return {
        "type": "paragraph",
        "content": {
            "paragraph_content": [
                {
                    "type": "text",
                    "content": text,
                }
            ]
        },
        "bbox": (
            bbox
            if bbox is not None
            else [10, 20, 30, 40]
        ),
    }


def test_extracts_numeric_body_citation_contexts_and_excludes_lists() -> None:
    mineru = [
        [
            _paragraph(
                "Single citation.<sup>5</sup>"
            ),
            _paragraph(
                "Range citation.<sup>7–9</sup>"
            ),
            {
                "type": "list",
                "content": {
                    "list_type": "reference_list",
                    "list_items": [
                        "<sup>5</sup> Bibliography entry"
                    ],
                },
                "bbox": [1, 2, 3, 4],
            },
        ]
    ]

    entries = extract_reference_contexts(
        mineru
    )

    by_index = {
        entry.reference_index: entry
        for entry in entries
    }

    assert set(by_index) == {
        5,
        7,
        8,
        9,
    }

    assert len(
        by_index[5].contexts
    ) == 1

    context = by_index[
        5
    ].contexts[0]

    assert context.page_index == 0
    assert context.block_index == 0
    assert context.citation_marker == "5"
    assert context.citation_count == 1
    assert context.block_type == "paragraph"
    assert "Single citation" in context.text

    assert (
        by_index[7]
        .contexts[0]
        .citation_count
        == 3
    )


def test_ranks_contexts_by_citation_specificity_and_caps_output() -> None:
    mineru = [
        [
            _paragraph(
                "Large group.<sup>1–10</sup>"
            ),
            _paragraph(
                "Three references.<sup>1,2,3</sup>"
            ),
            _paragraph(
                "Single reference.<sup>1</sup>"
            ),
            _paragraph(
                "Two references.<sup>1,4</sup>"
            ),
        ]
    ]

    entries = extract_reference_contexts(
        mineru,
        max_contexts_per_reference=3,
    )

    entry = next(
        item
        for item in entries
        if item.reference_index == 1
    )

    assert [
        context.citation_count
        for context in entry.contexts
    ] == [
        1,
        2,
        3,
    ]

    assert [
        context.block_index
        for context in entry.contexts
    ] == [
        2,
        3,
        1,
    ]


def test_keeps_only_best_marker_for_same_reference_in_one_block() -> None:
    mineru = [
        [
            _paragraph(
                "General evidence.<sup>1–5</sup> "
                "Specific evidence.<sup>1</sup>"
            )
        ]
    ]

    entries = extract_reference_contexts(
        mineru
    )

    entry = next(
        item
        for item in entries
        if item.reference_index == 1
    )

    assert len(
        entry.contexts
    ) == 1

    assert (
        entry.contexts[0].citation_count
        == 1
    )

    assert (
        entry.contexts[0].citation_marker
        == "1"
    )


def test_artifact_round_trip(tmp_path) -> None:
    mineru_path = (
        tmp_path
        / "paper_content_list_v2.json"
    )

    mineru_path.write_text(
        json.dumps(
            [
                [
                    _paragraph(
                        "First.<sup>12</sup>"
                    ),
                    _paragraph(
                        "Second.<sup>12,13</sup>"
                    ),
                ]
            ]
        ),
        encoding="utf-8",
    )

    output = extract_reference_context_artifact(
        mineru_path,
        tmp_path,
    )

    assert output == (
        tmp_path
        / "references"
        / "reference_context.json"
    )

    payload = json.loads(
        output.read_text(
            encoding="utf-8"
        )
    )

    assert payload[
        "schema_version"
    ] == 1

    assert payload[
        "source_kind"
    ] == "mineru_content_list_v2"

    assert payload[
        "reference_count"
    ] == 2

    assert payload[
        "context_count"
    ] == 3

    loaded = load_reference_context_artifact(
        output
    )

    assert set(loaded) == {
        12,
        13,
    }

    assert [
        context.citation_count
        for context in loaded[12]
    ] == [
        1,
        2,
    ]


def test_rejects_non_list_mineru_document() -> None:
    with pytest.raises(
        ValueError,
        match="top-level list",
    ):
        extract_reference_contexts(
            {
                "pages": []
            }
        )
