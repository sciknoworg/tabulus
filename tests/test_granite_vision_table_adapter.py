from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tabulus.table_ocr.base import TableOCRInput
from tabulus.table_ocr.granite_vision_table import (
    MODEL_VERSION,
    GraniteVisionTableAdapter,
    _cells_to_html,
)
from tabulus.table_ocr.output import parse_result_tables
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
            model_device=None,
            docling_version="2.123.1",
            transformers_version="4.57.3",
            torch_version="2.13.0+cu130",
        )

    def __call__(self, device: str) -> Any:
        self.calls.append(device)
        self.runtime.model_device = device
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
            "raw_output": (
                "[<otsl><ched>A<ched>B<nl>"
                "<fcel>1<fcel>2<nl></otsl>]"
            ),
            "html": (
                "<table><tr><th>A</th><th>B</th></tr>"
                "<tr><td>1</td><td>2</td></tr></table>"
            ),
            "image_size": [200, 100],
            "generated_tokens": 12,
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
    adapter = GraniteVisionTableAdapter(
        device=device,
        runtime_loader=loader,
        inference_runner=runner,
    )
    return adapter, loader, runner


def test_registry_reports_granite_vision_table() -> None:
    specs = {
        spec.name: spec
        for spec in list_table_ocr_adapters()
    }
    spec = specs["granite-vision-table"]
    assert spec.display_name == "Granite Vision 4.1 4B"
    assert spec.cpu_supported is False
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


def test_cpu_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="gpu:<index>"):
        GraniteVisionTableAdapter(device="cpu")


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="gpu:<index>"):
        GraniteVisionTableAdapter(device="cuda:0")


def test_result_preserves_versions_and_native_evidence(
    tmp_path,
) -> None:
    adapter, *_ = make_adapter()
    result = adapter.extract(make_input(tmp_path, 12))

    assert result.status == "ok"
    assert result.table_id == 12
    assert result.adapter_name == "granite-vision-table"
    assert result.adapter_version == "2.123.1"
    assert result.model_version == MODEL_VERSION
    assert result.result_count == 1

    native = result.native_json[0]["granite_vision"]
    assert native["model_repo"] == "ibm-granite/granite-vision-4.1-4b"
    assert native["model_revision"] == (
        "dd48e97503de471803850df70843cf9eb5da8712"
    )
    assert native["prompt"] == "<tables_otsl>"
    assert native["dtype"] == "bfloat16"
    assert native["attention_implementation"] == "sdpa"
    assert native["execution_device"] == "cuda:0"
    assert native["transformers_version"] == "4.57.3"
    assert native["num_rows"] == 2
    assert native["num_cols"] == 2
    assert native["otsl_seq"][0] == "ched"
    assert native["generated_tokens"] == 12


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


def test_empty_generated_output_is_explicit(tmp_path) -> None:
    payload = {
        "raw_output": "",
        "html": "",
        "image_size": [100, 50],
        "generated_tokens": 0,
        "otsl_seq": [],
        "num_rows": 0,
        "num_cols": 0,
        "cells": [],
    }
    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))

    assert result.status == "empty"
    assert "no generated table output" in result.error


def test_unusable_otsl_is_explicit(tmp_path) -> None:
    payload = {
        "raw_output": "not an OTSL table",
        "html": "",
        "image_size": [100, 50],
        "generated_tokens": 5,
        "otsl_seq": [],
        "num_rows": 0,
        "num_cols": 0,
        "cells": [],
    }
    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))

    assert result.status == "empty"
    assert "usable OTSL table" in result.error


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
