from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tabulus.table_ocr.base import TableOCRInput
from tabulus.table_ocr.output import parse_result_tables
from tabulus.table_ocr.registry import list_table_ocr_adapters
from tabulus.table_ocr.trivia import (
    MODEL_VERSION,
    TRiviaAdapter,
)


class FakeRuntimeLoader:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.runtime = SimpleNamespace(
            model_device=None,
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
            "raw_output": (
                "<fcel>A<fcel>B<nl>"
                "<fcel>1<fcel>2<nl>"
            ),
            "image_size": [200, 100],
            "prompt_tokens": 25,
            "generated_tokens": 12,
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

    adapter = TRiviaAdapter(
        device=device,
        runtime_loader=loader,
        inference_runner=runner,
    )

    return adapter, loader, runner


def test_registry_reports_trivia() -> None:
    specs = {
        spec.name: spec
        for spec in list_table_ocr_adapters()
    }

    spec = specs["trivia"]

    assert spec.display_name == "TRivia-3B"
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
        TRiviaAdapter(device="cpu")


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="gpu:<index>"):
        TRiviaAdapter(device="cuda:0")


def test_result_preserves_versions_and_native_evidence(
    tmp_path,
) -> None:
    adapter, *_ = make_adapter()

    result = adapter.extract(
        make_input(tmp_path, 12)
    )

    assert result.status == "ok"
    assert result.table_id == 12
    assert result.adapter_name == "trivia"
    assert result.adapter_version == "5.16.1"
    assert result.model_version == MODEL_VERSION
    assert result.result_count == 1

    native = result.native_json[0]["trivia"]

    assert native["model_repo"] == "opendatalab/TRivia-3B"
    assert native["model_revision"] == (
        "fcf890f3869afaa9fc768a14e72ab1ff46bfc813"
    )
    assert native["dtype"] == "bfloat16"
    assert native["max_new_tokens"] == 8192
    assert native["do_sample"] is False
    assert native["repetition_penalty"] == 1.05
    assert native["execution_device"] == "cuda:0"
    assert native["transformers_version"] == "5.16.1"
    assert native["torch_version"] == "2.13.0"
    assert native["accelerate_version"] == "1.14.0"
    assert native["image_size"] == [200, 100]
    assert native["prompt_tokens"] == 25
    assert native["generated_tokens"] == 12
    assert native["raw_otsl"].startswith("<fcel>A")
    assert native["normalization"].endswith(
        "otsl_table_to_html"
    )


def test_shared_parser_receives_one_rectangular_table(
    tmp_path,
) -> None:
    adapter, *_ = make_adapter()

    result = adapter.extract(
        make_input(tmp_path)
    )
    parsed = parse_result_tables(result)

    assert len(parsed) == 1
    assert parsed[0].rows == [
        ["A", "B"],
        ["1", "2"],
    ]


def test_ragged_otsl_is_preserved_then_rectangularized(
    tmp_path,
) -> None:
    payload = {
        "raw_output": (
            "<fcel>A<fcel>B<fcel>C<nl>"
            "<fcel>1<fcel>2<fcel>3<fcel>4<nl>"
        ),
        "image_size": [200, 100],
        "prompt_tokens": 25,
        "generated_tokens": 20,
    }

    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))
    parsed = parse_result_tables(result)

    assert parsed[0].rows == [
        ["A", "B", "C", ""],
        ["1", "2", "3", "4"],
    ]


def test_empty_generated_output_is_explicit(
    tmp_path,
) -> None:
    payload = {
        "raw_output": "",
        "image_size": [100, 50],
        "prompt_tokens": 20,
        "generated_tokens": 0,
    }

    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))

    assert result.status == "empty"
    assert "no generated OTSL output" in result.error


def test_unusable_otsl_is_explicit(
    tmp_path,
) -> None:
    payload = {
        "raw_output": "not an OTSL table",
        "image_size": [100, 50],
        "prompt_tokens": 20,
        "generated_tokens": 5,
    }

    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))

    assert result.status == "empty"
    assert "usable OTSL table" in result.error


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
