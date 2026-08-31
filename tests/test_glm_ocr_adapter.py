from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tabulus.table_ocr.base import TableOCRInput
from tabulus.table_ocr.glm_ocr import (
    GLMOCRAdapter,
    MODEL_VERSION,
)
from tabulus.table_ocr.output import parse_result_tables
from tabulus.table_ocr.registry import list_table_ocr_adapters


SIMPLE_HTML = (
    '<table border="1">'
    "<tr><td>A</td><td>B</td></tr>"
    "<tr><td>1</td><td>2</td></tr>"
    "</table>"
)


class FakeRuntimeLoader:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.runtime = SimpleNamespace(
            model_device=None,
            model_dtype="torch.bfloat16",
            transformers_version="5.16.1",
            torch_version="2.13.0",
            accelerate_version="1.14.0",
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
        self.payload = payload or {
            "raw_output": SIMPLE_HTML + "<|user|>",
            "clean_output": SIMPLE_HTML,
            "image_size": [200, 100],
            "prompt_tokens": 25,
            "generated_tokens": 30,
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

    adapter = GLMOCRAdapter(
        device=device,
        runtime_loader=loader,
        inference_runner=runner,
    )

    return adapter, loader, runner


def test_registry_reports_glm_ocr() -> None:
    specs = {
        spec.name: spec
        for spec in list_table_ocr_adapters()
    }

    spec = specs["glm-ocr"]

    assert spec.display_name == "GLM-OCR"
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
        GLMOCRAdapter(device="cpu")


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="gpu:<index>"):
        GLMOCRAdapter(device="cuda:0")


def test_result_preserves_versions_and_native_evidence(
    tmp_path,
) -> None:
    adapter, *_ = make_adapter()

    result = adapter.extract(
        make_input(tmp_path, 12)
    )

    assert result.status == "ok"
    assert result.table_id == 12
    assert result.adapter_name == "glm-ocr"
    assert result.adapter_version == "5.16.1"
    assert result.model_version == MODEL_VERSION
    assert result.result_count == 1

    native = result.native_json[0]["glm_ocr"]

    assert native["model_repo"] == "zai-org/GLM-OCR"
    assert native["model_revision"] == (
        "ca5d8b3e287e52589e37c28385d9655ee4372f9d"
    )
    assert native["prompt"] == "Table Recognition:"
    assert native["model_load_dtype"] == "auto"
    assert native["resolved_model_dtype"] == "torch.bfloat16"
    assert native["max_new_tokens"] == 8192
    assert native["execution_device"] == "cuda:0"
    assert native["transformers_version"] == "5.16.1"
    assert native["torch_version"] == "2.13.0"
    assert native["accelerate_version"] == "1.14.0"
    assert native["image_size"] == [200, 100]
    assert native["prompt_tokens"] == 25
    assert native["generated_tokens"] == 30
    assert native["raw_output"].endswith("<|user|>")
    assert native["clean_output"] == SIMPLE_HTML
    assert native["native_format"] == "html"
    assert native["special_tokens_removed_for_parsing"] is True
    assert native["normalization"] == "none"
    assert native["html_tables_detected"] == 1
    assert native["input_policy"] == "canonical_mineru_crop"
    assert native["layout_redetection"] is False
    assert native["recropping"] is False


def test_shared_parser_receives_native_html(
    tmp_path,
) -> None:
    adapter, *_ = make_adapter()

    result = adapter.extract(
        make_input(tmp_path)
    )
    parsed = parse_result_tables(result)

    assert len(parsed) == 1
    assert parsed[0].source == "html"
    assert parsed[0].rows == [
        ["A", "B"],
        ["1", "2"],
    ]


def test_rowspan_inconsistency_is_preserved_not_repaired(
    tmp_path,
) -> None:
    html = (
        "<table>"
        "<tr>"
        "<td>A</td><td>B</td><td>C</td><td>D</td><td>E</td>"
        "</tr>"
        "<tr>"
        '<td rowspan="2">X</td>'
        "<td></td><td>1</td><td>2</td><td>3</td>"
        "</tr>"
        "<tr>"
        "<td></td><td></td><td>4</td><td>5</td><td>6</td>"
        "</tr>"
        "</table>"
    )

    payload = {
        "raw_output": html + "<|user|>",
        "clean_output": html,
        "image_size": [200, 100],
        "prompt_tokens": 25,
        "generated_tokens": 50,
    }

    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))
    parsed = parse_result_tables(result)

    assert len(parsed) == 1
    assert parsed[0].n_cols == 6
    assert parsed[0].rows == [
        ["A", "B", "C", "D", "E", ""],
        ["X", "", "1", "2", "3", ""],
        ["", "", "", "4", "5", "6"],
    ]


def test_empty_generated_output_is_explicit(
    tmp_path,
) -> None:
    payload = {
        "raw_output": "<|user|>",
        "clean_output": "",
        "image_size": [100, 50],
        "prompt_tokens": 20,
        "generated_tokens": 1,
    }

    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))

    assert result.status == "empty"
    assert "no generated table output" in result.error


def test_non_html_output_is_explicit(
    tmp_path,
) -> None:
    payload = {
        "raw_output": "not a table<|user|>",
        "clean_output": "not a table",
        "image_size": [100, 50],
        "prompt_tokens": 20,
        "generated_tokens": 4,
    }

    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))

    assert result.status == "empty"
    assert "usable HTML table" in result.error


def test_multiple_html_tables_are_not_collapsed(
    tmp_path,
) -> None:
    html = (
        "<table><tr><td>A</td></tr></table>"
        "<table><tr><td>B</td></tr></table>"
    )

    payload = {
        "raw_output": html + "<|user|>",
        "clean_output": html,
        "image_size": [100, 50],
        "prompt_tokens": 20,
        "generated_tokens": 20,
    }

    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))
    parsed = parse_result_tables(result)

    assert result.status == "ok"
    assert result.result_count == 2
    assert len(parsed) == 2
    assert parsed[0].rows == [["A"]]
    assert parsed[1].rows == [["B"]]


def test_missing_image_does_not_load_runtime(
    tmp_path,
) -> None:
    adapter, loader, _ = make_adapter()

    result = adapter.extract(
        TableOCRInput(
            table_id=1,
            image_path=tmp_path / "missing.jpg",
        )
    )

    assert result.status == "error"
    assert loader.calls == []
