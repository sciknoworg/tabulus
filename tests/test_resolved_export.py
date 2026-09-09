from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tabulus.resolved_export import (
    ENRICHMENT_COLUMNS,
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


def _write_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    reconstruction = (
        tmp_path
        / "reconstructions"
        / "paddleocr-vl"
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
            ["HfO2", "2, 3"],
            ["TiO2", "4"],
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
                "schema_version": 1,
                "matched_tables": [
                    {
                        "table_id": 1,
                        "source_file": "table_001.csv",
                        "source_parsed": str(
                            reconstruction
                            / "parsed"
                            / "table_001.json"
                        ),
                        "source_prediction": (
                            "predictions/table_001.csv"
                        ),
                        "reference_column_index": 1,
                        "matches": [
                            {
                                "row_index": 0,
                                "value": "Refs.",
                                "found": False,
                                "matched_reference_indices": [],
                                "unmatched_tokens": [],
                                "is_header": True,
                            },
                            {
                                "row_index": 1,
                                "value": "1",
                                "found": True,
                                "matched_reference_indices": [1],
                                "unmatched_tokens": [],
                                "is_header": False,
                            },
                            {
                                "row_index": 2,
                                "value": "2, 3",
                                "found": True,
                                "matched_reference_indices": [2, 3],
                                "unmatched_tokens": [],
                                "is_header": False,
                            },
                            {
                                "row_index": 3,
                                "value": "4",
                                "found": False,
                                "matched_reference_indices": [],
                                "unmatched_tokens": ["4"],
                                "is_header": False,
                            },
                        ],
                    }
                ],
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
                "schema_version": 1,
                "entries": [
                    {
                        "resolution": {
                            "reference_index": 1,
                            "raw_reference": "Smith 2020",
                            "status": "validated_with_doi",
                            "canonical_doi": "10.1000/a",
                            "canonical_title": "Paper A",
                            "canonical_authors": ["A. Smith"],
                            "canonical_year": 2020,
                            "canonical_venue": "Journal A",
                            "source": "crossref",
                            "confidence": 0.98,
                            "reason": "validated",
                        }
                    },
                    {
                        "resolution": {
                            "reference_index": 2,
                            "raw_reference": "Jones 2021",
                            "status": "validated_without_doi",
                            "canonical_doi": "",
                            "canonical_title": "Paper B",
                            "canonical_authors": ["B. Jones"],
                            "canonical_year": 2021,
                            "canonical_venue": "Journal B",
                            "source": "core",
                            "confidence": 0.91,
                            "reason": "validated",
                        }
                    },
                    {
                        "resolution": {
                            "reference_index": 3,
                            "raw_reference": "Unknown 2022",
                            "status": "rejected",
                            "canonical_doi": "",
                            "canonical_title": "",
                            "canonical_authors": [],
                            "canonical_year": None,
                            "canonical_venue": "",
                            "source": "",
                            "confidence": None,
                            "reason": "insufficient evidence",
                        }
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    return (
        matches_path,
        resolution_path,
        prediction,
    )


def test_export_preserves_prediction_and_appends_resolution(
    tmp_path: Path,
) -> None:
    (
        matches_path,
        resolution_path,
        prediction,
    ) = _write_fixture(tmp_path)

    original = prediction.read_bytes()

    result = export_resolved_csvs(
        matches_path,
        resolution_path,
    )

    assert result.tables_exported == 1
    assert prediction.read_bytes() == original

    resolved = Path(
        result.tables[0]["resolved_csv"]
    )

    with resolved.open(
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
    assert rows[0][2:] == list(
        ENRICHMENT_COLUMNS
    )

    # Single resolved bibliography index.
    assert json.loads(rows[1][2]) == [1]
    assert json.loads(rows[1][3]) == [
        "validated_with_doi"
    ]
    assert json.loads(rows[1][4]) == [
        "10.1000/a"
    ]

    # Multiple references retain one-to-one positional order,
    # including a rejected resolution.
    assert json.loads(rows[2][2]) == [2, 3]
    assert json.loads(rows[2][3]) == [
        "validated_without_doi",
        "rejected",
    ]
    assert json.loads(rows[2][5]) == [
        "Paper B",
        "",
    ]
    assert json.loads(rows[2][6]) == [
        ["B. Jones"],
        [],
    ]

    # An unmatched reference token remains traceable.
    assert json.loads(rows[3][2]) == []
    assert json.loads(rows[3][-1]) == ["4"]

    manifest = json.loads(
        result.manifest_path.read_text(
            encoding="utf-8"
        )
    )

    assert manifest["schema_version"] == 1
    assert manifest["tables_exported"] == 1
    assert manifest["enrichment_columns"] == list(
        ENRICHMENT_COLUMNS
    )
    assert manifest["tables"][0][
        "header_row_index"
    ] == 0
    assert manifest["tables"][0][
        "resolution_occurrences"
    ] == 3
    assert manifest["tables"][0][
        "merge_status"
    ] == "physical_only"


def test_no_header_does_not_insert_or_shift_rows(
    tmp_path: Path,
) -> None:
    (
        matches_path,
        resolution_path,
        prediction,
    ) = _write_fixture(tmp_path)

    payload = json.loads(
        matches_path.read_text(
            encoding="utf-8"
        )
    )

    payload["matched_tables"][0]["matches"][0][
        "is_header"
    ] = False

    matches_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    original_rows = list(
        csv.reader(
            prediction.open(
                "r",
                newline="",
                encoding="utf-8",
            )
        )
    )

    result = export_resolved_csvs(
        matches_path,
        resolution_path,
    )

    resolved = Path(
        result.tables[0]["resolved_csv"]
    )

    with resolved.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.reader(handle))

    assert len(rows) == len(original_rows)
    assert [
        row[:2]
        for row in rows
    ] == original_rows
    assert result.tables[0][
        "header_row_index"
    ] is None


def test_missing_step6_resolution_aborts_before_output(
    tmp_path: Path,
) -> None:
    (
        matches_path,
        resolution_path,
        _prediction,
    ) = _write_fixture(tmp_path)

    payload = json.loads(
        resolution_path.read_text(
            encoding="utf-8"
        )
    )
    payload["entries"] = payload["entries"][:1]

    resolution_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    reconstruction = (
        matches_path.parent.parent
    )
    output_dir = (
        reconstruction
        / "resolved_reference_tables"
    )

    with pytest.raises(
        ValueError,
        match="Step 6 has no final resolution",
    ):
        export_resolved_csvs(
            matches_path,
            resolution_path,
        )

    assert not output_dir.exists()


def test_out_of_range_step5_row_is_rejected(
    tmp_path: Path,
) -> None:
    (
        matches_path,
        resolution_path,
        _prediction,
    ) = _write_fixture(tmp_path)

    payload = json.loads(
        matches_path.read_text(
            encoding="utf-8"
        )
    )
    payload["matched_tables"][0]["matches"][1][
        "row_index"
    ] = 99

    matches_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="outside the prediction CSV",
    ):
        export_resolved_csvs(
            matches_path,
            resolution_path,
        )


def test_non_final_step6_status_is_rejected(
    tmp_path: Path,
) -> None:
    (
        matches_path,
        resolution_path,
        _prediction,
    ) = _write_fixture(tmp_path)

    payload = json.loads(
        resolution_path.read_text(
            encoding="utf-8"
        )
    )
    payload["entries"][0]["resolution"][
        "status"
    ] = "unresolved"

    resolution_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="non-final status",
    ):
        export_resolved_csvs(
            matches_path,
            resolution_path,
        )
