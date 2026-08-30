from __future__ import annotations

from typing import Any

import pytest

from tabulus.table_ocr.base import TableOCRDependencyError, TableOCRInput
from tabulus.table_ocr.nuextract3 import NuExtract3Adapter
from tabulus.table_ocr.registry import list_table_ocr_adapters
import tabulus.table_ocr.nuextract3 as nuextract_module


class FakeRuntimeLoader:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.runtime = object()

    def __call__(self, device: str) -> object:
        self.calls.append(device)
        return self.runtime


class FakeInferenceRunner:
    def __init__(
        self,
        output: str = "",
        *,
        error: Exception | None = None,
    ) -> None:
        self.output = output
        self.error = error
        self.calls: list[tuple[Any, Any]] = []

    def __call__(self, image_path, runtime: Any) -> str:
        self.calls.append((image_path, runtime))

        if self.error is not None:
            raise self.error

        return self.output


def make_input(tmp_path, table_id: int = 7) -> TableOCRInput:
    image = tmp_path / "table.jpg"
    image.write_bytes(b"fake image")

    return TableOCRInput(
        table_id=table_id,
        image_path=image,
        provenance={
            "table_id": table_id,
            "page_nr": 7,
            "source": "mineru",
        },
    )


def make_adapter(
    *,
    device: str = "gpu:0",
    output: str = "<table><tr><td>A</td></tr></table>",
    inference_error: Exception | None = None,
):
    loader = FakeRuntimeLoader()
    runner = FakeInferenceRunner(output, error=inference_error)
    adapter = NuExtract3Adapter(
        device=device,
        runtime_loader=loader,
        inference_runner=runner,
    )
    return adapter, loader, runner


def test_registry_reports_gpu_only_nuextract3() -> None:
    specs = {spec.name: spec for spec in list_table_ocr_adapters()}

    assert "nuextract3" in specs
    assert specs["nuextract3"].display_name == "NuExtract3"
    assert specs["nuextract3"].cpu_supported is False
    assert specs["nuextract3"].gpu_supported is True


def test_capabilities_report_gpu_only_support() -> None:
    adapter, *_ = make_adapter()

    assert adapter.capabilities.cpu_supported is False
    assert adapter.capabilities.gpu_supported is True
    assert adapter.capabilities.supports_device("cpu") is False
    assert adapter.capabilities.supports_device("gpu:0") is True


def test_gpu_device_is_translated_and_inference_receives_crop(tmp_path) -> None:
    adapter, loader, runner = make_adapter()
    table = make_input(tmp_path)

    result = adapter.extract(table)

    assert loader.calls == ["cuda:0"]
    assert runner.calls == [(table.image_path, loader.runtime)]
    assert result.status == "ok"


def test_result_preserves_identity_provenance_and_native_output(tmp_path) -> None:
    raw = '<table><tr><td rowspan="2">A</td></tr></table>'
    adapter, *_ = make_adapter(output=raw)
    table = make_input(tmp_path, table_id=12)

    result = adapter.extract(table)

    assert result.table_id == 12
    assert result.adapter_name == "nuextract3"
    assert result.model_version == "numind/NuExtract3"
    assert result.device == "gpu:0"
    assert result.source_image == table.image_path
    assert result.provenance["page_nr"] == 7
    assert result.result_count == 1
    assert result.native_markdown == [raw]
    assert result.native_json == [
        {
            "raw": raw,
            "mode": "markdown",
            "enable_thinking": False,
            "max_new_tokens": 8192,
        }
    ]


def test_empty_generated_text_is_explicit(tmp_path) -> None:
    adapter, *_ = make_adapter(output="   ")

    result = adapter.extract(make_input(tmp_path))

    assert result.status == "empty"
    assert result.result_count == 1
    assert result.error is not None


def test_inference_failure_is_explicit(tmp_path) -> None:
    adapter, *_ = make_adapter(
        inference_error=RuntimeError("inference exploded")
    )

    result = adapter.extract(make_input(tmp_path))

    assert result.status == "error"
    assert "inference exploded" in result.error


def test_runtime_instance_is_reused(tmp_path) -> None:
    adapter, loader, runner = make_adapter()

    first = make_input(tmp_path, table_id=1)
    second_image = tmp_path / "table2.jpg"
    second_image.write_bytes(b"fake image")
    second = TableOCRInput(
        table_id=2,
        image_path=second_image,
    )

    adapter.extract(first)
    adapter.extract(second)

    assert loader.calls == ["cuda:0"]
    assert len(runner.calls) == 2


def test_missing_image_does_not_initialize_runtime(tmp_path) -> None:
    adapter, loader, _ = make_adapter()
    table = TableOCRInput(
        table_id=1,
        image_path=tmp_path / "missing.jpg",
    )

    result = adapter.extract(table)

    assert result.status == "error"
    assert "not found" in result.error
    assert loader.calls == []


def test_missing_transformers_has_actionable_error(tmp_path, monkeypatch) -> None:
    real_import_module = nuextract_module.importlib.import_module

    def fake_import_module(name: str):
        if name == "transformers":
            raise ImportError("not installed")
        return real_import_module(name)

    monkeypatch.setattr(
        nuextract_module.importlib,
        "import_module",
        fake_import_module,
    )

    adapter = NuExtract3Adapter(device="gpu:0")

    with pytest.raises(
        TableOCRDependencyError,
        match="requires PyTorch, Transformers, Accelerate, and Pillow",
    ):
        adapter.extract(make_input(tmp_path))


def test_cpu_and_invalid_devices_are_rejected() -> None:
    with pytest.raises(ValueError, match="GPU-only"):
        NuExtract3Adapter(device="cpu")

    with pytest.raises(ValueError, match="GPU-only"):
        NuExtract3Adapter(device="cuda:0")
