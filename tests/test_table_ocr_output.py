from __future__ import annotations

import csv
import json
from pathlib import Path

from tabulus.table_ocr.base import TableOCRResult
from tabulus.table_ocr.output import (
    parse_result_tables,
    write_table_ocr_artifacts,
)
from tabulus.table_ocr.parsing import ParsedTable


def make_result(
    source_image: Path,
    *,
    status: str = "ok",
    native_markdown: list[object] | None = None,
) -> TableOCRResult:
    return TableOCRResult(
        table_id=1,
        adapter_name="paddleocr-vl",
        device="gpu:0",
        source_image=source_image,
        status=status,  # type: ignore[arg-type]
        provenance={"page_nr": 6},
        adapter_version="3.7.0",
        model_version="PaddleOCR-VL v1.6",
        result_count=1,
        native_json=[{"res": {"input_path": str(source_image)}}],
        native_markdown=native_markdown or [],
    )


def test_write_artifacts_keeps_native_parsed_and_prediction_layers(
    tmp_path: Path,
) -> None:
    source_image = tmp_path / "page_006_table_001.jpg"
    result = make_result(
        source_image,
        native_markdown=[
            {
                "markdown_texts": """
                <table>
                  <tr><th>Material</th><th>Refs.</th></tr>
                  <tr><td>Al2O3</td><td>83, 90, and 91</td></tr>
                </table>
                """
            }
        ],
    )

    paths = write_table_ocr_artifacts(result, tmp_path / "paddleocr-vl")

    assert paths.native_result == (
        tmp_path / "paddleocr-vl/native/page_006_table_001.json"
    )
    assert paths.parsed_result == (
        tmp_path / "paddleocr-vl/parsed/page_006_table_001.json"
    )
    assert paths.prediction_csv == (
        tmp_path / "paddleocr-vl/predictions/page_006_table_001.csv"
    )

    native = json.loads(paths.native_result.read_text(encoding="utf-8"))
    assert native["table_id"] == 1
    assert native["adapter_name"] == "paddleocr-vl"
    assert native["native_json"] == [
        {"res": {"input_path": str(source_image)}}
    ]

    parsed = json.loads(paths.parsed_result.read_text(encoding="utf-8"))
    assert parsed["tables_found"] == 1
    assert parsed["tables"][0]["source"] == "html"
    assert parsed["tables"][0]["rows"] == [
        ["Material", "Refs."],
        ["Al2O3", "83, 90, and 91"],
    ]
    assert parsed["prediction_csv"] == (
        "predictions/page_006_table_001.csv"
    )
    assert parsed["warnings"] == []

    assert paths.prediction_csv is not None
    with paths.prediction_csv.open(newline="", encoding="utf-8") as handle:
        assert list(csv.reader(handle)) == [
            ["Material", "Refs."],
            ["Al2O3", "83, 90, and 91"],
        ]


def test_parse_result_tables_uses_markdown_view_not_native_json(
    tmp_path: Path,
) -> None:
    result = make_result(
        tmp_path / "page_006_table_001.jpg",
        native_markdown=[
            {
                "markdown_texts": """
                | Material | Refs. |
                | --- | --- |
                | Al2O3 | 90 |
                """
            }
        ],
    )
    result.native_json = [
        {"some_other_serialization": "<table><tr><td>ignored</td></tr></table>"}
    ]

    tables = parse_result_tables(result)

    assert len(tables) == 1
    assert tables[0].source == "markdown"
    assert tables[0].rows == [
        ["Material", "Refs."],
        ["Al2O3", "90"],
    ]


def test_writer_accepts_explicit_common_parsed_table(
    tmp_path: Path,
) -> None:
    result = make_result(
        tmp_path / "page_010_table_003.jpg",
        native_markdown=[],
    )

    paths = write_table_ocr_artifacts(
        result,
        tmp_path / "future-adapter",
        parsed_tables=[
            ParsedTable(
                rows=[["Header", "Value"], ["A", "B"]],
                source="html",
            )
        ],
    )

    assert paths.prediction_csv is not None
    with paths.prediction_csv.open(newline="", encoding="utf-8") as handle:
        assert list(csv.reader(handle)) == [
            ["Header", "Value"],
            ["A", "B"],
        ]


def test_multiple_parsed_tables_are_preserved_without_arbitrary_csv_choice(
    tmp_path: Path,
) -> None:
    result = make_result(
        tmp_path / "page_006_table_001.jpg",
        native_markdown=[
            {
                "markdown_texts": """
                <table><tr><td>A</td></tr></table>
                <table><tr><td>B</td></tr></table>
                """
            }
        ],
    )

    paths = write_table_ocr_artifacts(result, tmp_path / "paddleocr-vl")

    parsed = json.loads(paths.parsed_result.read_text(encoding="utf-8"))
    assert parsed["tables_found"] == 2
    assert parsed["tables"][0]["rows"] == [["A"]]
    assert parsed["tables"][1]["rows"] == [["B"]]
    assert parsed["prediction_csv"] is None
    assert "ambiguous" in parsed["warnings"][0]
    assert paths.prediction_csv is None
    assert not (
        tmp_path / "paddleocr-vl/predictions/page_006_table_001.csv"
    ).exists()


def test_non_ok_result_is_persisted_without_prediction_csv(
    tmp_path: Path,
) -> None:
    result = make_result(
        tmp_path / "page_006_table_001.jpg",
        status="error",
        native_markdown=[],
    )
    result.error = "inference failed"

    paths = write_table_ocr_artifacts(result, tmp_path / "paddleocr-vl")

    native = json.loads(paths.native_result.read_text(encoding="utf-8"))
    parsed = json.loads(paths.parsed_result.read_text(encoding="utf-8"))

    assert native["error"] == "inference failed"
    assert parsed["status"] == "error"
    assert parsed["tables_found"] == 0
    assert parsed["prediction_csv"] is None
    assert paths.prediction_csv is None
