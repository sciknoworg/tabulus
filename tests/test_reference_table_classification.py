from __future__ import annotations

import json
from pathlib import Path

from tabulus.reference_tables import (
    REFERENCE_TABLE_CLASSIFICATION_NAME,
    SELECTED_REFERENCE_TABLES_NAME,
    classify_reconstruction_tables,
    classify_reference_like_table,
)
from tabulus.reference_tables.classification import (
    _caption_text,
    _is_explicit_continuation_caption,
    _table_label,
)


def test_reference_header_and_plain_numeric_cells_are_classified() -> None:
    decision = classify_reference_like_table(
        [
            ["Material", "Reactant A", "Refs."],
            ["B2O3", "BBr3", "85"],
            ["B3P3O2", "B(OMe)3", "88 and 89"],
            ["Al2O3", "AlCl3", "83, 90, and 91"],
        ]
    )

    assert decision.is_reference_table is True
    assert decision.has_tag_match is True
    assert decision.has_citation_match is True
    assert decision.matched_header_cells == ("Refs.",)
    assert decision.matched_citation_cells == (
        "85",
        "88 and 89",
        "83, 90, and 91",
    )


def test_legacy_author_year_citation_without_tag_is_classified() -> None:
    decision = classify_reference_like_table(
        [
            ["Material", "Notes"],
            ["Al2O3", "Smith et al. 2020"],
        ]
    )

    assert decision.is_reference_table is True
    assert decision.has_tag_match is False
    assert decision.has_citation_match is True


def test_reference_header_without_citation_is_not_classified() -> None:
    decision = classify_reference_like_table(
        [
            ["Material", "References"],
            ["Al2O3", "see discussion"],
        ]
    )

    assert decision.is_reference_table is False
    assert decision.has_tag_match is True
    assert decision.has_citation_match is False


def _write_reconstruction_fixture(root: Path) -> Path:
    parsed_dir = root / "parsed"
    predictions_dir = root / "predictions"
    parsed_dir.mkdir(parents=True)
    predictions_dir.mkdir(parents=True)

    prediction_1 = predictions_dir / "page_006_table_001.csv"
    prediction_1.write_text(
        "Material,Refs.\nB2O3,85\n",
        encoding="utf-8",
    )

    prediction_2 = predictions_dir / "page_007_table_002.csv"
    prediction_2.write_text(
        "Material,Temperature\nAl2O3,300\n",
        encoding="utf-8",
    )

    (parsed_dir / "page_006_table_001.json").write_text(
        json.dumps(
            {
                "table_id": 1,
                "status": "ok",
                "tables": [
                    {
                        "rows": [
                            ["Material", "Refs."],
                            ["B2O3", "85"],
                        ],
                        "n_rows": 2,
                        "n_cols": 2,
                        "source": "html",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    (parsed_dir / "page_007_table_002.json").write_text(
        json.dumps(
            {
                "table_id": 2,
                "status": "ok",
                "tables": [
                    {
                        "rows": [
                            ["Material", "Temperature"],
                            ["Al2O3", "300"],
                        ],
                        "n_rows": 2,
                        "n_cols": 2,
                        "source": "html",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    (root / "batch_summary.json").write_text(
        json.dumps(
            {
                "adapter_name": "paddleocr-vl",
                "items": [
                    {
                        "table_id": 1,
                        "status": "ok",
                        "parsed_tables": 1,
                        "parsed_result": "parsed/page_006_table_001.json",
                        "prediction_csv": (
                            "predictions/page_006_table_001.csv"
                        ),
                    },
                    {
                        "table_id": 2,
                        "status": "ok",
                        "parsed_tables": 1,
                        "parsed_result": "parsed/page_007_table_002.json",
                        "prediction_csv": (
                            "predictions/page_007_table_002.csv"
                        ),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    return prediction_1


def test_batch_classification_writes_manifest_without_changing_predictions(
    tmp_path: Path,
) -> None:
    reconstruction_dir = tmp_path / "reconstructions/paddleocr-vl"
    prediction_1 = _write_reconstruction_fixture(reconstruction_dir)
    original_prediction = prediction_1.read_text(encoding="utf-8")

    result = classify_reconstruction_tables(reconstruction_dir)

    assert result.tables_considered == 2
    assert result.reference_tables_found == 1
    assert result.output_path == (
        reconstruction_dir / REFERENCE_TABLE_CLASSIFICATION_NAME
    )
    assert prediction_1.read_text(encoding="utf-8") == original_prediction

    payload = json.loads(
        result.output_path.read_text(encoding="utf-8")
    )

    assert payload["adapter_name"] == "paddleocr-vl"
    assert payload["tables_considered"] == 2
    assert payload["reference_tables_found"] == 1
    assert [table["table_id"] for table in payload["tables"]] == [1, 2]
    assert payload["tables"][0]["is_reference_table"] is True
    assert payload["tables"][0]["source_prediction"] == (
        "predictions/page_006_table_001.csv"
    )
    assert payload["tables"][1]["is_reference_table"] is False


    selected_path = reconstruction_dir / SELECTED_REFERENCE_TABLES_NAME
    selected = json.loads(selected_path.read_text(encoding="utf-8"))

    assert selected["schema_version"] == 1
    assert selected["adapter_name"] == "paddleocr-vl"
    assert selected["tables_considered"] == 2
    assert selected["reference_tables_selected"] == 1
    assert [table["table_id"] for table in selected["tables"]] == [1]
    assert selected["tables"][0]["source_prediction"] == (
        "predictions/page_006_table_001.csv"
    )

    # Selection is a logical view only: non-reference reconstructions remain.
    assert (
        reconstruction_dir
        / "predictions/page_007_table_002.csv"
    ).is_file()


def test_missing_structured_table_gets_explicit_negative_decision(
    tmp_path: Path,
) -> None:
    reconstruction_dir = tmp_path / "reconstructions/paddleocr-vl"
    parsed_dir = reconstruction_dir / "parsed"
    parsed_dir.mkdir(parents=True)

    parsed_path = parsed_dir / "page_001_table_001.json"
    parsed_path.write_text(
        json.dumps(
            {
                "table_id": 1,
                "status": "empty",
                "tables": [],
            }
        ),
        encoding="utf-8",
    )

    (reconstruction_dir / "batch_summary.json").write_text(
        json.dumps(
            {
                "adapter_name": "paddleocr-vl",
                "items": [
                    {
                        "table_id": 1,
                        "status": "empty",
                        "parsed_tables": 0,
                        "parsed_result": (
                            "parsed/page_001_table_001.json"
                        ),
                        "prediction_csv": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = classify_reconstruction_tables(reconstruction_dir)
    classification = result.classifications[0]

    assert classification.decision.is_reference_table is False
    assert classification.parsed_tables == 0
    assert classification.decision.reason == (
        "No structured table was parsed for classification."
    )


def test_identity_mismatch_does_not_replace_previous_manifest(
    tmp_path: Path,
) -> None:
    reconstruction_dir = tmp_path / "reconstructions/paddleocr-vl"
    parsed_dir = reconstruction_dir / "parsed"
    parsed_dir.mkdir(parents=True)

    (parsed_dir / "page_001_table_001.json").write_text(
        json.dumps(
            {
                "table_id": 999,
                "tables": [{"rows": [["Refs."], ["85"]]}],
            }
        ),
        encoding="utf-8",
    )

    (reconstruction_dir / "batch_summary.json").write_text(
        json.dumps(
            {
                "adapter_name": "paddleocr-vl",
                "items": [
                    {
                        "table_id": 1,
                        "status": "ok",
                        "parsed_result": (
                            "parsed/page_001_table_001.json"
                        ),
                        "prediction_csv": (
                            "predictions/page_001_table_001.csv"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output_path = (
        reconstruction_dir / REFERENCE_TABLE_CLASSIFICATION_NAME
    )
    output_path.write_text("previous", encoding="utf-8")

    try:
        classify_reconstruction_tables(reconstruction_dir)
    except ValueError as error:
        assert "table identity does not match" in str(error)
    else:
        raise AssertionError("Expected classification identity failure.")

    assert output_path.read_text(encoding="utf-8") == "previous"


def test_continuation_caption_supports_common_table_identifier_forms() -> None:
    cases = (
        ("Table 1 (continued)", "1"),
        ("Table 1. Continued", "1"),
        ("TABLE A. CONTINUED", "A"),
        ("Table a — continued", "A"),
        ("Table I. Continued", "I"),
        ("Table IV (cont.)", "IV"),
        ("Table S1, contd.", "S1"),
        ("Supplementary Table S 1 — cont'd", "S1"),
        ("Table A-1 continued", "A1"),
        ("Table 1-A continued", "1A"),
        ("Table 2.1 (continued)", "2.1"),
        ("TABLE III. \x01Continued.-", "III"),
    )

    for raw_caption, expected_label in cases:
        caption = _caption_text(raw_caption)

        assert _table_label(caption) == expected_label
        assert _is_explicit_continuation_caption(caption) is True


def test_continuation_caption_does_not_use_descriptive_table_mentions() -> None:
    captions = (
        "TABLE V. Typical processes (references in Table III).",
        "Table 5. Results discussed and continued in the text.",
        "Table A. Comparison with Table I.",
    )

    for raw_caption in captions:
        caption = _caption_text(raw_caption)
        assert _is_explicit_continuation_caption(caption) is False


def _write_parsed_table(
    reconstruction_dir: Path,
    table_id: int,
    rows: list[list[str]],
) -> str:
    parsed_dir = reconstruction_dir / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    name = f"page_{table_id:03d}_table_{table_id:03d}.json"
    path = parsed_dir / name
    path.write_text(
        json.dumps(
            {
                "table_id": table_id,
                "status": "ok",
                "tables": [
                    {
                        "rows": rows,
                        "n_rows": len(rows),
                        "n_cols": max(
                            (len(row) for row in rows),
                            default=0,
                        ),
                        "source": "html",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return f"parsed/{name}"


def test_explicit_continuation_inherits_reference_classification(
    tmp_path: Path,
) -> None:
    crop_root = tmp_path / "crops"
    reconstruction_dir = (
        crop_root / "reconstructions/paddleocr-vl"
    )

    parsed_paths = {
        1: _write_parsed_table(
            reconstruction_dir,
            1,
            [
                ["Material", "Refs."],
                ["B2O3", "85"],
            ],
        ),
        2: _write_parsed_table(
            reconstruction_dir,
            2,
            [
                ["Material", "Temperature"],
                ["Al2O3", "300"],
            ],
        ),
        3: _write_parsed_table(
            reconstruction_dir,
            3,
            [
                ["Material", "Temperature"],
                ["HfO2", "325"],
            ],
        ),
        4: _write_parsed_table(
            reconstruction_dir,
            4,
            [
                ["Material", "Temperature"],
                ["TiO2", "350"],
            ],
        ),
    }

    crop_root.mkdir(parents=True, exist_ok=True)
    (crop_root / "tables_index.json").write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "table_id": 1,
                        "page_nr": 1,
                        "table_caption": ["Table 4. Precursors"],
                    },
                    {
                        "table_id": 2,
                        "page_nr": 2,
                        "table_caption": ["TABLE 4. \x01Continued.-"],
                    },
                    {
                        "table_id": 3,
                        "page_nr": 3,
                        "table_caption": ["Table 4—Continued"],
                    },
                    {
                        "table_id": 4,
                        "page_nr": 4,
                        "table_caption": ["Table 5. Temperatures"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    (reconstruction_dir / "batch_summary.json").write_text(
        json.dumps(
            {
                "adapter_name": "paddleocr-vl",
                "crop_root": str(crop_root),
                "items": [
                    {
                        "table_id": table_id,
                        "status": "ok",
                        "parsed_result": parsed_paths[table_id],
                        "prediction_csv": None,
                    }
                    for table_id in (1, 2, 3, 4)
                ],
            }
        ),
        encoding="utf-8",
    )

    result = classify_reconstruction_tables(reconstruction_dir)

    table_1, table_2, table_3, table_4 = result.classifications

    assert table_1.decision.is_reference_table is True
    assert table_1.independent_is_reference_table is True
    assert table_1.classification_source == "heuristic"
    assert table_1.continued_from_table_id is None

    assert table_2.independent_is_reference_table is False
    assert table_2.decision.is_reference_table is True
    assert table_2.classification_source == "continued_table"
    assert table_2.continued_from_table_id == 1
    assert table_2.continuation_caption == "TABLE 4. Continued.-"

    assert table_3.independent_is_reference_table is False
    assert table_3.decision.is_reference_table is True
    assert table_3.classification_source == "continued_table"
    assert table_3.continued_from_table_id == 2

    assert table_4.independent_is_reference_table is False
    assert table_4.decision.is_reference_table is False
    assert table_4.classification_source == "heuristic"
    assert table_4.continued_from_table_id is None

    assert result.reference_tables_found == 3

    payload = json.loads(
        result.output_path.read_text(encoding="utf-8")
    )
    assert payload["tables"][1]["classification_source"] == (
        "continued_table"
    )
    assert payload["tables"][1]["continued_from_table_id"] == 1
    assert payload["tables"][1]["independent_is_reference_table"] is False


def test_continuation_does_not_inherit_from_non_reference_parent(
    tmp_path: Path,
) -> None:
    crop_root = tmp_path / "crops"
    reconstruction_dir = (
        crop_root / "reconstructions/paddleocr-vl"
    )

    parsed_1 = _write_parsed_table(
        reconstruction_dir,
        1,
        [
            ["Material", "Temperature"],
            ["Al2O3", "300"],
        ],
    )
    parsed_2 = _write_parsed_table(
        reconstruction_dir,
        2,
        [
            ["Material", "Temperature"],
            ["HfO2", "325"],
        ],
    )

    crop_root.mkdir(parents=True, exist_ok=True)
    (crop_root / "tables_index.json").write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "table_id": 1,
                        "table_caption": ["Table 7. Temperatures"],
                    },
                    {
                        "table_id": 2,
                        "table_caption": ["Table 7 (continued)"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    (reconstruction_dir / "batch_summary.json").write_text(
        json.dumps(
            {
                "adapter_name": "paddleocr-vl",
                "crop_root": str(crop_root),
                "items": [
                    {
                        "table_id": 1,
                        "status": "ok",
                        "parsed_result": parsed_1,
                        "prediction_csv": None,
                    },
                    {
                        "table_id": 2,
                        "status": "ok",
                        "parsed_result": parsed_2,
                        "prediction_csv": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = classify_reconstruction_tables(reconstruction_dir)
    continuation = result.classifications[1]

    assert continuation.continued_from_table_id == 1
    assert continuation.independent_is_reference_table is False
    assert continuation.decision.is_reference_table is False
    assert continuation.classification_source == "heuristic"
