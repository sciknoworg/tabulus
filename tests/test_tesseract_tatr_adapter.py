from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tabulus.table_ocr.base import TableOCRInput
from tabulus.table_ocr.output import parse_result_tables
from tabulus.table_ocr.registry import list_table_ocr_adapters
from tabulus.table_ocr.tesseract_tatr import (
    TesseractTATRAdapter,
    _cells_to_html,
    _parse_tesseract_tsv,
)


class FakeRuntimeLoader:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.runtime = SimpleNamespace(tesseract_version="tesseract 5.5.3")

    def __call__(self, device: str) -> Any:
        self.calls.append(device)
        return self.runtime


class FakeInferenceRunner:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {
            "html": "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>",
            "tokens": [{"text": "A", "bbox": [0, 0, 1, 1]}],
            "objects": [],
            "structure": {"rows": [[0, 0, 2, 1]], "columns": [[0, 0, 1, 1], [1, 0, 2, 1]]},
            "cells": [
                {"row_nums": [0], "column_nums": [0], "cell_text": "A"},
                {"row_nums": [0], "column_nums": [1], "cell_text": "B"},
                {"row_nums": [1], "column_nums": [0], "cell_text": "1"},
                {"row_nums": [1], "column_nums": [1], "cell_text": "2"},
            ],
            "token_slot_confidence": 0.8,
            "tesseract_command": "tesseract crop stdout -l eng --psm 6 tsv",
        }
        self.calls: list[tuple[Any, Any]] = []

    def __call__(self, image_path, runtime: Any) -> dict[str, Any]:
        self.calls.append((image_path, runtime))
        return self.payload


def make_input(tmp_path, table_id: int = 7) -> TableOCRInput:
    image = tmp_path / "table.jpg"
    image.write_bytes(b"fake image")
    return TableOCRInput(
        table_id=table_id,
        image_path=image,
        provenance={"table_id": table_id, "source": "mineru"},
    )


def make_adapter(*, device: str = "gpu:0", payload: dict[str, Any] | None = None):
    loader = FakeRuntimeLoader()
    runner = FakeInferenceRunner(payload)
    adapter = TesseractTATRAdapter(
        device=device,
        runtime_loader=loader,
        inference_runner=runner,
    )
    return adapter, loader, runner


def test_registry_reports_tesseract_tatr() -> None:
    specs = {spec.name: spec for spec in list_table_ocr_adapters()}
    spec = specs["tesseract-tatr"]
    assert spec.display_name == "Tesseract + Table Transformer"
    assert spec.cpu_supported is True
    assert spec.gpu_supported is True


def test_gpu_device_translation_and_runtime_reuse(tmp_path) -> None:
    adapter, loader, runner = make_adapter()
    first = make_input(tmp_path, 1)
    second_path = tmp_path / "second.jpg"
    second_path.write_bytes(b"fake image")
    second = TableOCRInput(table_id=2, image_path=second_path)

    assert adapter.extract(first).status == "ok"
    assert adapter.extract(second).status == "ok"
    assert loader.calls == ["cuda:0"]
    assert len(runner.calls) == 2


def test_cpu_device_is_supported() -> None:
    adapter, loader, _ = make_adapter(device="cpu")
    assert adapter.capabilities.supports_device("cpu") is True
    assert loader.calls == []


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="gpu:<index>"):
        TesseractTATRAdapter(device="cuda:0")


def test_result_preserves_model_and_native_evidence(tmp_path) -> None:
    adapter, *_ = make_adapter()
    result = adapter.extract(make_input(tmp_path, 12))

    assert result.status == "ok"
    assert result.table_id == 12
    assert result.adapter_name == "tesseract-tatr"
    assert result.model_version == "microsoft/table-transformer-structure-recognition-v1.1-all"
    assert result.result_count == 1
    assert result.native_json[0]["tesseract"]["version"] == "tesseract 5.5.3"
    assert result.native_json[0]["tesseract"]["psm"] == 6
    assert result.native_json[0]["tatr"]["threshold"] == 0.5
    assert result.native_markdown[0].startswith("<table>")


def test_shared_parser_receives_one_rectangular_table(tmp_path) -> None:
    adapter, *_ = make_adapter()
    result = adapter.extract(make_input(tmp_path))
    parsed = parse_result_tables(result)
    assert len(parsed) == 1
    assert parsed[0].rows == [["A", "B"], ["1", "2"]]


def test_empty_tesseract_tokens_are_explicit(tmp_path) -> None:
    payload = {
        "html": "",
        "tokens": [],
        "objects": [],
        "structure": {},
        "cells": [],
        "token_slot_confidence": 0.0,
        "tesseract_command": "tesseract ...",
    }
    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))
    assert result.status == "empty"
    assert "no OCR word tokens" in result.error


def test_missing_image_does_not_load_runtime(tmp_path) -> None:
    adapter, loader, _ = make_adapter()
    result = adapter.extract(TableOCRInput(table_id=1, image_path=tmp_path / "missing.jpg"))
    assert result.status == "error"
    assert loader.calls == []


def test_parse_tesseract_tsv_filters_to_word_records() -> None:
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "4\t1\t1\t1\t1\t0\t0\t0\t10\t10\t-1\t\n"
        "5\t1\t1\t1\t1\t1\t10\t20\t30\t40\t96.5\tMaterial\n"
    )
    assert _parse_tesseract_tsv(tsv) == [
        {
            "text": "Material",
            "bbox": [10.0, 20.0, 40.0, 60.0],
            "block_num": 1,
            "line_num": 1,
            "span_num": 1,
            "confidence": 96.5,
        }
    ]


def test_cells_to_html_preserves_spans() -> None:
    cells = [
        {
            "row_nums": [0],
            "column_nums": [0, 1],
            "header": True,
            "cell_text": "Header",
        },
        {
            "row_nums": [1],
            "column_nums": [0],
            "header": False,
            "cell_text": "A&B",
        },
        {
            "row_nums": [1],
            "column_nums": [1],
            "header": False,
            "cell_text": "<x>",
        },
    ]
    html_table = _cells_to_html(cells)
    assert '<th colspan="2">Header</th>' in html_table
    assert "A&amp;B" in html_table
    assert "&lt;x&gt;" in html_table
