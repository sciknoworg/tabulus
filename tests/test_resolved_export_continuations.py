from __future__ import annotations

import csv
import json
from pathlib import Path

from tabulus.resolved_export import (
    export_resolved_csvs,
)


def _write_csv(
    path: Path,
    rows: list[list[str]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        csv.writer(handle).writerows(rows)


def _fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    reconstruction = (
        tmp_path
        / "reconstructions"
        / "adapter"
    )

    prediction = (
        reconstruction
        / "predictions"
        / "table_001.csv"
    )

    _write_csv(
        prediction,
        [
            ["Material", "Refs."],
            ["Al2O3", "1"],
            ["HfO2", "2"],
        ],
    )

    references = (
        reconstruction / "references"
    )

    references.mkdir(
        parents=True,
        exist_ok=True,
    )

    matches_path = (
        references
        / "reference_matches.json"
    )

    matches_path.write_text(
        json.dumps(
            {
                "matched_tables": [
                    {
                        "table_id": 1,
                        "source_parsed": str(
                            reconstruction
                            / "parsed"
                            / "table_001.json"
                        ),
                        "source_prediction": (
                            "predictions/"
                            "table_001.csv"
                        ),
                        "reference_column_index": 1,
                        "matches": [
                            {
                                "row_index": 0,
                                "matched_reference_indices": [],
                                "unmatched_tokens": [],
                                "is_header": True,
                            },
                            {
                                "row_index": 1,
                                "matched_reference_indices": [1],
                                "unmatched_tokens": [],
                                "is_header": False,
                            },
                            {
                                "row_index": 2,
                                "matched_reference_indices": [2],
                                "unmatched_tokens": [],
                                "is_header": False,
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    resolution_path = (
        references
        / "reference_resolution.json"
    )

    resolution_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "resolution": {
                            "reference_index": 1,
                            "raw_reference": "Ref 1",
                            "status": "validated_with_doi",
                            "canonical_doi": "10.1/a",
                            "canonical_title": "A",
                            "canonical_authors": ["A"],
                            "canonical_year": 2020,
                            "canonical_venue": "J",
                            "source": "crossref",
                            "confidence": 1.0,
                            "reason": "ok",
                        }
                    },
                    {
                        "resolution": {
                            "reference_index": 2,
                            "raw_reference": "Ref 2",
                            "status": "validated_without_doi",
                            "canonical_doi": "",
                            "canonical_title": "B",
                            "canonical_authors": ["B"],
                            "canonical_year": 2021,
                            "canonical_venue": "J",
                            "source": "core",
                            "confidence": 0.9,
                            "reason": "ok",
                        }
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    return (
        matches_path,
        resolution_path,
        prediction,
    )


def _add_child(
    matches_path: Path,
    rows: list[list[str]],
    *,
    reference_column: int,
    header_row: int | None,
) -> Path:
    reconstruction = (
        matches_path.parent.parent
    )

    prediction = (
        reconstruction
        / "predictions"
        / "table_002.csv"
    )

    _write_csv(
        prediction,
        rows,
    )

    payload = json.loads(
        matches_path.read_text(
            encoding="utf-8"
        )
    )

    matches = []

    for row_index, row in enumerate(rows):
        value = (
            row[reference_column]
            if reference_column < len(row)
            else ""
        )

        is_header = (
            row_index == header_row
        )

        indices = []

        if not is_header:
            if value.strip() == "1":
                indices = [1]
            elif value.strip() == "2":
                indices = [2]

        matches.append(
            {
                "row_index": row_index,
                "matched_reference_indices": indices,
                "unmatched_tokens": [],
                "is_header": is_header,
            }
        )

    payload["matched_tables"].append(
        {
            "table_id": 2,
            "source_parsed": str(
                reconstruction
                / "parsed"
                / "table_002.json"
            ),
            "source_prediction": (
                "predictions/table_002.csv"
            ),
            "reference_column_index": (
                reference_column
            ),
            "matches": matches,
        }
    )

    matches_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    return prediction


def _write_index(
    tmp_path: Path,
    *,
    third_fragment: bool = False,
) -> Path:
    records = [
        {
            "table_id": 1,
            "table_caption": [
                "Table 4. Comparison"
            ],
            "continuation": {
                "is_continuation": False,
                "continued_from_table_id": None,
                "continuation_root_table_id": 1,
                "printed_table_label": "4",
                "evidence": None,
                "link_status": "not_continuation",
            },
        },
        {
            "table_id": 2,
            "table_caption": [
                "Table 4 (continued)"
            ],
            "continuation": {
                "is_continuation": True,
                "continued_from_table_id": 1,
                "continuation_root_table_id": 1,
                "printed_table_label": "4",
                "evidence": "explicit_caption",
                "link_status": "linked",
            },
        },
    ]

    if third_fragment:
        records.append(
            {
                "table_id": 3,
                "table_caption": [
                    "Table 4 continued"
                ],
                "continuation": {
                    "is_continuation": True,
                    "continued_from_table_id": 2,
                    "continuation_root_table_id": 1,
                    "printed_table_label": "4",
                    "evidence": "explicit_caption",
                    "link_status": "linked",
                },
            }
        )

    path = tmp_path / "tables_index.json"

    path.write_text(
        json.dumps(
            {"tables": records}
        ),
        encoding="utf-8",
    )

    return path


def test_merge_repeated_header(
    tmp_path: Path,
) -> None:
    (
        matches,
        resolution,
        root_prediction,
    ) = _fixture(tmp_path)

    child_prediction = _add_child(
        matches,
        [
            ["Material", "Refs."],
            ["ZnO", "1"],
        ],
        reference_column=1,
        header_row=0,
    )

    _write_index(tmp_path)

    root_before = root_prediction.read_bytes()
    child_before = child_prediction.read_bytes()

    result = export_resolved_csvs(
        matches,
        resolution,
        merge_continuations=True,
    )

    assert root_prediction.read_bytes() == root_before
    assert child_prediction.read_bytes() == child_before

    group = result.continuation_groups[0]

    assert group["merge_status"] == "merged"
    assert group[
        "physical_table_ids"
    ] == [1, 2]
    assert group[
        "dropped_repeated_header_table_ids"
    ] == [2]
    assert group["alignment"]["2"] == (
        "exact_repeated_header"
    )

    with Path(
        group["merged_csv"]
    ).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.reader(handle))

    assert len(rows) == 4
    assert rows[0][:2] == [
        "Material",
        "Refs.",
    ]
    assert rows[0][-1] == (
        "tabulus_physical_table_id"
    )
    assert rows[-1][:2] == [
        "ZnO",
        "1",
    ]
    assert rows[-1][-1] == "2"

    assert all(
        Path(item["resolved_csv"]).is_file()
        for item in result.tables
    )


def test_merge_without_repeated_header(
    tmp_path: Path,
) -> None:
    matches, resolution, _ = _fixture(
        tmp_path
    )

    _add_child(
        matches,
        [
            ["ZnO", "1"],
            ["SiO2", "2"],
        ],
        reference_column=1,
        header_row=None,
    )

    _write_index(tmp_path)

    result = export_resolved_csvs(
        matches,
        resolution,
        merge_continuations=True,
    )

    group = result.continuation_groups[0]

    assert group["merge_status"] == "merged"
    assert group["alignment"]["2"] == (
        "positional_same_width"
    )

    with Path(
        group["merged_csv"]
    ).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.reader(handle))

    assert len(rows) == 5
    assert rows[-2][:2] == [
        "ZnO",
        "1",
    ]
    assert rows[-2][-1] == "2"


def test_different_column_count_is_not_merged(
    tmp_path: Path,
) -> None:
    matches, resolution, _ = _fixture(
        tmp_path
    )

    _add_child(
        matches,
        [
            [
                "Material",
                "Condition",
                "Refs.",
            ],
            [
                "ZnO",
                "annealed",
                "1",
            ],
        ],
        reference_column=2,
        header_row=0,
    )

    _write_index(tmp_path)

    result = export_resolved_csvs(
        matches,
        resolution,
        merge_continuations=True,
    )

    group = result.continuation_groups[0]

    assert (
        group["merge_status"]
        == "incompatible"
    )
    assert group["merged_csv"] is None

    assert all(
        Path(item["resolved_csv"]).is_file()
        for item in result.tables
    )


def test_conflicting_repeated_header_is_not_merged(
    tmp_path: Path,
) -> None:
    matches, resolution, _ = _fixture(
        tmp_path
    )

    _add_child(
        matches,
        [
            [
                "Different material",
                "Refs.",
            ],
            ["ZnO", "1"],
        ],
        reference_column=1,
        header_row=0,
    )

    _write_index(tmp_path)

    result = export_resolved_csvs(
        matches,
        resolution,
        merge_continuations=True,
    )

    group = result.continuation_groups[0]

    assert (
        group["merge_status"]
        == "incompatible"
    )
    assert "conflicts" in group["reason"]
    assert group["merged_csv"] is None


def test_incomplete_group_is_not_partially_merged(
    tmp_path: Path,
) -> None:
    matches, resolution, _ = _fixture(
        tmp_path
    )

    _add_child(
        matches,
        [
            ["Material", "Refs."],
            ["ZnO", "1"],
        ],
        reference_column=1,
        header_row=0,
    )

    _write_index(
        tmp_path,
        third_fragment=True,
    )

    result = export_resolved_csvs(
        matches,
        resolution,
        merge_continuations=True,
    )

    group = result.continuation_groups[0]

    assert group[
        "physical_table_ids"
    ] == [1, 2, 3]

    assert group[
        "missing_table_ids"
    ] == [3]

    assert (
        group["merge_status"]
        == "incomplete"
    )

    assert group["merged_csv"] is None
