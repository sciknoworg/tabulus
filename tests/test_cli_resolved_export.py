from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from tabulus.cli import build_parser, main


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
) -> tuple[Path, Path]:
    reconstruction = (
        tmp_path
        / "table-crops"
        / "paper"
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
        ],
    )

    references = reconstruction / "references"
    references.mkdir(
        parents=True,
        exist_ok=True,
    )

    matches = references / "reference_matches.json"

    matches.write_text(
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
                            "predictions/table_001.csv"
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
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    resolution = (
        tmp_path
        / "paper-artifacts"
        / "references"
        / "reference_resolution.json"
    )
    resolution.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolution.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "resolution": {
                            "reference_index": 1,
                            "raw_reference": "Smith 2020",
                            "status": "validated_with_doi",
                            "canonical_doi": "10.1000/example",
                            "canonical_title": "Example paper",
                            "canonical_authors": ["A. Smith"],
                            "canonical_year": 2020,
                            "canonical_venue": "Example Journal",
                            "source": "crossref",
                            "confidence": 0.99,
                            "reason": "validated",
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    return matches, resolution


def test_parser_accepts_step7_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "export-resolved-csv",
            "--reference-matches",
            "matches.json",
            "--reference-resolution",
            "resolution.json",
            "--merge-continuations",
            "--tables-index",
            "tables_index.json",
        ]
    )

    assert args.command == "export-resolved-csv"
    assert args.merge_continuations is True
    assert args.tables_index == Path("tables_index.json")


def test_step7_cli_exports_physical_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matches, resolution = _fixture(tmp_path)
    output_dir = tmp_path / "resolved"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tabulus",
            "export-resolved-csv",
            "--reference-matches",
            str(matches),
            "--reference-resolution",
            str(resolution),
            "--out",
            str(output_dir),
        ],
    )

    main()

    manifest = output_dir / "resolved_tables.json"
    assert manifest.is_file()

    payload = json.loads(
        manifest.read_text(encoding="utf-8")
    )

    assert payload["tables_exported"] == 1
    assert payload["merge_continuations"] is False

    resolved_csv = Path(
        payload["tables"][0]["resolved_csv"]
    )
    assert resolved_csv.is_file()

    with resolved_csv.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.reader(handle))

    assert rows[0][:2] == ["Material", "Refs."]
    assert json.loads(rows[1][2]) == [1]
    assert json.loads(rows[1][4]) == [
        "10.1000/example"
    ]


def test_tables_index_requires_merge_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matches, resolution = _fixture(tmp_path)

    tables_index = tmp_path / "tables_index.json"
    tables_index.write_text(
        json.dumps({"tables": []}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tabulus",
            "export-resolved-csv",
            "--reference-matches",
            str(matches),
            "--reference-resolution",
            str(resolution),
            "--tables-index",
            str(tables_index),
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "--tables-index requires "
            "--merge-continuations"
        ),
    ):
        main()
