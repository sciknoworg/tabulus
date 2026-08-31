from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tabulus.table_ocr.base import TableOCRInput
from tabulus.table_ocr.output import parse_result_tables
from tabulus.table_ocr.rapidocr_tableformer import (
    MODEL_VERSION,
    RapidOCRTableFormerAdapter,
    _cells_to_html,
)
from tabulus.table_ocr.registry import list_table_ocr_adapters


def _cell(
    row: int,
    col: int,
    text: str,
    *,
    column_header: bool = False,
) -> dict[str, Any]:
    return {
        "bbox": None,
        "row_span": 1,
        "col_span": 1,
        "start_row_offset_idx": row,
        "end_row_offset_idx": row + 1,
        "start_col_offset_idx": col,
        "end_col_offset_idx": col + 1,
        "text": text,
        "column_header": column_header,
        "row_header": False,
        "row_section": False,
        "fillable": False,
    }


class FakeRuntimeLoader:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.runtime = SimpleNamespace(
            table_device=None,
            docling_version="2.123.1",
            docling_core_version="2.92.0",
            docling_ibm_models_version="4.0.0",
            rapidocr_version="3.9.2",
            onnxruntime_version="1.29.0",
        )

    def __call__(self, device: str) -> Any:
        self.calls.append(device)
        self.runtime.table_device = device
        return self.runtime


class FakeInferenceRunner:
    def __init__(
        self,
        payload: dict[str, Any] | None = None,
    ) -> None:
        cells = [
            _cell(0, 0, "A", column_header=True),
            _cell(0, 1, "B", column_header=True),
            _cell(1, 0, "1"),
            _cell(1, 1, "2"),
        ]
        self.payload = payload or {
            "html": (
                "<table><tr><th>A</th><th>B</th></tr>"
                "<tr><td>1</td><td>2</td></tr></table>"
            ),
            "image_size": [200, 100],
            "table_box": [0.0, 0.0, 200.0, 100.0],
            "ocr_tokens": [
                {
                    "id": 0,
                    "text": "A",
                    "confidence": 0.99,
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                }
            ],
            "otsl_seq": [
                "ched",
                "ched",
                "nl",
                "fcel",
                "fcel",
                "nl",
            ],
            "num_rows": 2,
            "num_cols": 2,
            "cells": cells,
        }
        self.calls: list[tuple[Any, Any]] = []

    def __call__(
        self,
        image_path,
        runtime: Any,
    ) -> dict[str, Any]:
        self.calls.append((image_path, runtime))
        return self.payload


def make_input(
    tmp_path,
    table_id: int = 7,
) -> TableOCRInput:
    image = tmp_path / "table.jpg"
    image.write_bytes(b"fake image")
    return TableOCRInput(
        table_id=table_id,
        image_path=image,
        provenance={
            "table_id": table_id,
            "source": "mineru",
        },
    )


def make_adapter(
    *,
    device: str = "gpu:0",
    payload: dict[str, Any] | None = None,
):
    loader = FakeRuntimeLoader()
    runner = FakeInferenceRunner(payload)
    adapter = RapidOCRTableFormerAdapter(
        device=device,
        runtime_loader=loader,
        inference_runner=runner,
    )
    return adapter, loader, runner


def test_registry_reports_rapidocr_tableformer() -> None:
    specs = {
        spec.name: spec
        for spec in list_table_ocr_adapters()
    }
    spec = specs["rapidocr-tableformer"]
    assert (
        spec.display_name
        == "RapidOCR + Docling TableFormer"
    )
    assert spec.cpu_supported is True
    assert spec.gpu_supported is True


def test_gpu_device_translation_and_runtime_reuse(
    tmp_path,
) -> None:
    adapter, loader, runner = make_adapter()
    first = make_input(tmp_path, 1)
    second_path = tmp_path / "second.jpg"
    second_path.write_bytes(b"fake image")
    second = TableOCRInput(
        table_id=2,
        image_path=second_path,
    )

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
        RapidOCRTableFormerAdapter(device="cuda:0")


def test_result_preserves_versions_and_native_evidence(
    tmp_path,
) -> None:
    adapter, *_ = make_adapter()
    result = adapter.extract(make_input(tmp_path, 12))

    assert result.status == "ok"
    assert result.table_id == 12
    assert result.adapter_name == "rapidocr-tableformer"
    assert result.adapter_version == "2.123.1"
    assert result.model_version == MODEL_VERSION
    assert result.result_count == 1

    native = result.native_json[0]
    assert native["rapidocr"]["package_version"] == "3.9.2"
    assert native["rapidocr"]["backend"] == "onnxruntime"
    assert native["rapidocr"]["execution_device"] == "cpu"
    assert native["tableformer"]["model_revision"] == "v2.3.0"
    assert native["tableformer"]["mode"] == "accurate"
    assert native["tableformer"]["do_cell_matching"] is True
    assert native["tableformer"]["execution_device"] == "cuda:0"
    assert native["tableformer"]["num_rows"] == 2
    assert native["tableformer"]["num_cols"] == 2
    assert native["tableformer"]["otsl_seq"][0] == "ched"


def test_shared_parser_receives_one_rectangular_table(
    tmp_path,
) -> None:
    adapter, *_ = make_adapter()
    result = adapter.extract(make_input(tmp_path))
    parsed = parse_result_tables(result)

    assert len(parsed) == 1
    assert parsed[0].rows == [
        ["A", "B"],
        ["1", "2"],
    ]


def test_empty_rapidocr_tokens_are_explicit(tmp_path) -> None:
    payload = {
        "html": "",
        "image_size": [100, 50],
        "table_box": [0.0, 0.0, 100.0, 50.0],
        "ocr_tokens": [],
        "otsl_seq": [],
        "num_rows": 0,
        "num_cols": 0,
        "cells": [],
    }
    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))

    assert result.status == "empty"
    assert "no OCR text tokens" in result.error


def test_missing_image_does_not_load_runtime(tmp_path) -> None:
    adapter, loader, _ = make_adapter()
    result = adapter.extract(
        TableOCRInput(
            table_id=1,
            image_path=tmp_path / "missing.jpg",
        )
    )

    assert result.status == "error"
    assert loader.calls == []


def test_cells_to_html_preserves_spans() -> None:
    cells = [
        {
            **_cell(
                0,
                0,
                "Header",
                column_header=True,
            ),
            "col_span": 2,
            "end_col_offset_idx": 2,
        },
        _cell(1, 0, "A&B"),
        _cell(1, 1, "<x>"),
    ]

    html_table = _cells_to_html(cells, 2, 2)

    assert '<th colspan="2">Header</th>' in html_table
    assert "A&amp;B" in html_table
    assert "&lt;x&gt;" in html_table
