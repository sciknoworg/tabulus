from __future__ import annotations

import json
from pathlib import Path

import pytest

from tabulus.bibliography.models import BibliographyEntry
from tabulus.reference_matching import (
    AUTHOR_ONLY_METHOD,
    AUTHOR_YEAR_METHOD,
    DOI_EXACT_METHOD,
    NUMERIC_POSITION_METHOD,
    detect_reference_column,
    extract_numeric_reference_tokens,
    match_reference_value,
    match_selected_reference_tables,
)


def _bibliography() -> tuple[BibliographyEntry, ...]:
    return (
        BibliographyEntry(1, "Smith J. Example paper. 2020.", "10.1000/example", "grobid"),
        BibliographyEntry(2, "Jones A. Another paper. 2021.", "", "grobid"),
        BibliographyEntry(3, "Müller B. Third paper. 2022.", "", "grobid"),
        BibliographyEntry(4, "Smith R. Later work. 2024.", "", "grobid"),
    )


def test_numeric_reference_tokens_support_lists_ranges_and_conjunctions() -> None:
    assert extract_numeric_reference_tokens("[1, 3-4]") == ["1", "3", "4"]
    assert extract_numeric_reference_tokens("2 and 4") == ["2", "4"]
    assert extract_numeric_reference_tokens("1, 3, and 4") == ["1", "3", "4"]
    assert extract_numeric_reference_tokens("(2 and 4)") == ["2", "4"]
    assert extract_numeric_reference_tokens("Smith 2020") == []
    assert extract_numeric_reference_tokens("Smith (2020)") == []
    assert extract_numeric_reference_tokens("Smith et al. (2020)") == []


def test_numeric_matching_records_positional_provenance() -> None:
    result = match_reference_value("[1,3]", _bibliography())
    assert result.matched_reference_indices == (1, 3)
    assert [candidate.method for candidate in result.candidates] == [
        NUMERIC_POSITION_METHOD,
        NUMERIC_POSITION_METHOD,
    ]
    assert result.unmatched_tokens == ()

    parenthesized = match_reference_value("(2)", _bibliography())
    assert parenthesized.matched_reference_indices == (2,)
    assert parenthesized.candidates[0].method == NUMERIC_POSITION_METHOD


def test_numeric_matching_preserves_unmatched_positions() -> None:
    result = match_reference_value("1, 99", _bibliography())
    assert result.matched_reference_indices == (1,)
    assert result.unmatched_tokens == ("99",)


def test_doi_and_author_year_matching_are_deterministic() -> None:
    doi_result = match_reference_value("doi:10.1000/example", _bibliography())
    assert doi_result.matched_reference_indices == (1,)
    assert doi_result.candidates[0].method == DOI_EXACT_METHOD

    ay_result = match_reference_value("Jones et al. 2021", _bibliography())
    assert ay_result.matched_reference_indices == (2,)
    assert ay_result.candidates[0].method == AUTHOR_YEAR_METHOD

    parenthesized_year = match_reference_value(
        "Smith et al. (2020)", _bibliography()
    )
    assert parenthesized_year.matched_reference_indices == (1,)
    assert parenthesized_year.candidates[0].method == AUTHOR_YEAR_METHOD


def test_reference_column_detection_prefers_reference_header_and_values() -> None:
    rows = [
        ["Material", "Temperature", "Refs."],
        ["Al2O3", "300", "1"],
        ["HfO2", "350", "2 and 3"],
    ]
    assert detect_reference_column(rows) == 2


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    reconstruction_dir = root / "reconstructions" / "adapter-x"
    parsed_dir = reconstruction_dir / "parsed"
    predictions_dir = reconstruction_dir / "predictions"
    parsed_dir.mkdir(parents=True)
    predictions_dir.mkdir(parents=True)

    parsed_path = parsed_dir / "page_001_table_001.json"
    parsed_path.write_text(
        json.dumps(
            {
                "table_id": 1,
                "status": "ok",
                "tables": [
                    {
                        "rows": [
                            ["Material", "Refs."],
                            ["Al2O3", "1"],
                            ["HfO2", "2 and 3"],
                        ],
                        "n_rows": 3,
                        "n_cols": 2,
                        "source": "html",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    prediction_path = predictions_dir / "page_001_table_001.csv"
    prediction_path.write_text(
        "Material,Refs.\nAl2O3,1\nHfO2,2 and 3\n",
        encoding="utf-8",
    )

    selected_path = reconstruction_dir / "selected_reference_tables.json"
    selected_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "adapter_name": "adapter-x",
                "reconstruction_dir": str(reconstruction_dir),
                "reference_tables_selected": 1,
                "tables": [
                    {
                        "table_id": 1,
                        "source_parsed": "parsed/page_001_table_001.json",
                        "source_prediction": "predictions/page_001_table_001.csv",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    bibliography_path = root / "references" / "bibliography.json"
    bibliography_path.parent.mkdir(parents=True)
    bibliography_path.write_text(
        json.dumps(
            {
                "bibliography_count": 3,
                "bibliography_source": "grobid",
                "entries": [entry.to_dict() for entry in _bibliography()[:3]],
            }
        ),
        encoding="utf-8",
    )
    return selected_path, bibliography_path, prediction_path


def test_pipeline_writes_reference_matches_without_mutating_prediction_csv(
    tmp_path: Path,
) -> None:
    selected_path, bibliography_path, prediction_path = _write_fixture(tmp_path)
    prediction_before = prediction_path.read_text(encoding="utf-8")

    result = match_selected_reference_tables(selected_path, bibliography_path)

    assert result.reference_tables_checked == 1
    assert result.output_path == (
        selected_path.parent / "references" / "reference_matches.json"
    )
    assert prediction_path.read_text(encoding="utf-8") == prediction_before

    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    table = payload["matched_tables"][0]
    assert table["reference_column_index"] == 1
    assert table["matches_found"] == 2
    assert table["matches_total"] == 2
    assert table["matches"][0]["is_header"] is True
    assert table["matches"][1]["matched_reference_indices"] == [1]
    assert table["matches"][2]["matched_reference_indices"] == [2, 3]
    assert table["matches"][2]["match_provenance"] == [
        {"reference_index": 2, "method": "numeric_position", "token": "2"},
        {"reference_index": 3, "method": "numeric_position", "token": "3"},
    ]
    assert payload["numeric_reference_semantics"].startswith("1-based bibliography position")


def test_pipeline_rejects_parsed_table_identity_mismatch_without_writing_output(
    tmp_path: Path,
) -> None:
    selected_path, bibliography_path, _ = _write_fixture(tmp_path)
    parsed_path = selected_path.parent / "parsed" / "page_001_table_001.json"
    payload = json.loads(parsed_path.read_text(encoding="utf-8"))
    payload["table_id"] = 999
    parsed_path.write_text(json.dumps(payload), encoding="utf-8")

    output_path = selected_path.parent / "references" / "reference_matches.json"
    with pytest.raises(ValueError, match="identity does not match"):
        match_selected_reference_tables(selected_path, bibliography_path)
    assert not output_path.exists()


def test_textual_matching_robustness_cases() -> None:
    bibliography = (
        BibliographyEntry(
            1,
            "Müller B. Atomic layer deposition study. 2022.",
            "",
            "grobid",
        ),
        BibliographyEntry(
            2,
            "O'Connor J. Surface chemistry in thin films. 2020.",
            "",
            "grobid",
        ),
        BibliographyEntry(
            3,
            "Smith-Jones A. Conformal coatings. 2021.",
            "",
            "grobid",
        ),
        BibliographyEntry(
            4,
            "Smith J. Example paper. 2020.",
            "",
            "grobid",
        ),
        BibliographyEntry(
            5,
            "Jones A. Another example paper. 2021.",
            "",
            "grobid",
        ),
        BibliographyEntry(
            6,
            "Brown C. IR microscopy characterization of ALD films. 2019.",
            "",
            "grobid",
        ),
    )

    for value, expected_index in (
        ("Müller 2022", 1),
        ("O'Connor 2020", 2),
        ("Smith-Jones 2021", 3),
    ):
        result = match_reference_value(value, bibliography)
        assert result.matched_reference_indices == (expected_index,)
        assert result.candidates[0].method == AUTHOR_YEAR_METHOD

    multiline = match_reference_value(
        "Smith 2020\nJones 2021",
        bibliography,
    )
    assert multiline.tokens == ("Smith 2020", "Jones 2021")
    assert multiline.matched_reference_indices == (4, 5)
    assert multiline.unmatched_tokens == ()

    assert not match_reference_value(
        "IR microscopy",
        bibliography,
    ).found

    assert not match_reference_value(
        "Atomic layer deposition",
        bibliography,
    ).found


def test_reference_header_vocabulary_matches_stage3_style_terms() -> None:
    from tabulus.reference_matching.matching import (
        looks_like_reference_header,
    )

    for value in (
        "Refs.",
        "References",
        "Citation",
        "Sources",
        "Literature",
        "Publications",
        "Studies",
        "Works",
    ):
        assert looks_like_reference_header(value)


def test_pipeline_records_unavailable_and_ambiguous_parsed_tables(
    tmp_path: Path,
) -> None:
    selected_path, bibliography_path, _ = _write_fixture(tmp_path)
    reconstruction_dir = selected_path.parent
    parsed_dir = reconstruction_dir / "parsed"

    empty_path = parsed_dir / "page_002_table_002.json"
    empty_path.write_text(
        json.dumps(
            {
                "table_id": 2,
                "status": "empty",
                "tables": [],
            }
        ),
        encoding="utf-8",
    )

    multiple_path = parsed_dir / "page_003_table_003.json"
    multiple_path.write_text(
        json.dumps(
            {
                "table_id": 3,
                "status": "ok",
                "tables": [
                    {
                        "rows": [["Refs."], ["1"]],
                        "n_rows": 2,
                        "n_cols": 1,
                        "source": "html",
                    },
                    {
                        "rows": [["Refs."], ["2"]],
                        "n_rows": 2,
                        "n_cols": 1,
                        "source": "html",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    selected["reference_tables_selected"] = 3
    selected["tables"].extend(
        [
            {
                "table_id": 2,
                "source_status": "empty",
                "source_parsed": "parsed/page_002_table_002.json",
                "source_prediction": "predictions/page_002_table_002.csv",
            },
            {
                "table_id": 3,
                "source_status": "ok",
                "source_parsed": "parsed/page_003_table_003.json",
                "source_prediction": "predictions/page_003_table_003.csv",
            },
        ]
    )
    selected_path.write_text(json.dumps(selected), encoding="utf-8")

    result = match_selected_reference_tables(
        selected_path,
        bibliography_path,
    )

    assert result.reference_tables_selected == 3
    assert result.reference_tables_checked == 1
    assert result.reference_tables_skipped == 2

    assert [item["reason"] for item in result.skipped_tables] == [
        "no_parsed_table",
        "multiple_parsed_tables",
    ]
    assert [item["parsed_table_count"] for item in result.skipped_tables] == [
        0,
        2,
    ]

    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert payload["reference_tables_selected"] == 3
    assert payload["reference_tables_checked"] == 1
    assert payload["reference_tables_skipped"] == 2
    assert len(payload["matched_tables"]) == 1
    assert len(payload["skipped_tables"]) == 2


def test_numeric_ocr_spacing_recovery_and_author_only_position() -> None:
    assert extract_numeric_reference_tokens(
        "1989 and1990"
    ) == ["1989", "1990"]

    assert extract_numeric_reference_tokens(
        "and 1974"
    ) == ["1974"]

    assert extract_numeric_reference_tokens(
        "1975 and 1976 1954, 1977, and 1978 1979"
    ) == [
        "1975",
        "1976",
        "1954",
        "1977",
        "1978",
        "1979",
    ]

    assert extract_numeric_reference_tokens(
        "Smith 2020"
    ) == []

    bibliography = (
        BibliographyEntry(
            1,
            "Smith J. Example paper. 2020.",
            "",
            "grobid",
        ),
        BibliographyEntry(
            2,
            "Smith-Jones A. Another paper. 2021.",
            "",
            "grobid",
        ),
        BibliographyEntry(
            3,
            (
                "Brown C. Conformal substrate study using "
                "AAO, As, Al, HO, and inherent methods. 2019."
            ),
            "",
            "grobid",
        ),
        BibliographyEntry(
            4,
            "J. Taylor. Another example paper. 2018.",
            "",
            "grobid",
        ),
    )

    smith = match_reference_value("Smith", bibliography)
    assert smith.matched_reference_indices == (1,)
    assert smith.candidates[0].method == AUTHOR_ONLY_METHOD

    # Very short author-like tokens are deliberately not resolved by the
    # conservative author-only fallback.
    assert not match_reference_value("He", bibliography).found
    assert not match_reference_value("HO", bibliography).found

    taylor = match_reference_value("Taylor", bibliography)
    assert taylor.matched_reference_indices == (4,)
    assert taylor.candidates[0].method == AUTHOR_ONLY_METHOD

    for value in (
        "Conformal",
        "Substrate",
        "AAO",
        "As",
        "Al",
        "HO",
        "Inherent",
    ):
        assert not match_reference_value(
            value,
            bibliography,
        ).found
