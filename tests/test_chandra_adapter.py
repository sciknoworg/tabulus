from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tabulus.table_ocr import TableOCRDependencyError, TableOCRInput
from tabulus.table_ocr.chandra import ChandraAdapter
import tabulus.table_ocr.chandra as chandra_module


class FakeModelLoader:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.model = object()

    def __call__(self, device: str) -> object:
        self.calls.append(device)
        return self.model


class FakeGenerator:
    def __init__(
        self,
        results: list[Any] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.results = results if results is not None else []
        self.error = error
        self.calls: list[tuple[list[Any], Any]] = []

    def __call__(self, batch: list[Any], model: Any) -> list[Any]:
        self.calls.append((batch, model))

        if self.error is not None:
            raise self.error

        return self.results


class FakeInputFactory:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(**kwargs)


def make_input(tmp_path, table_id: int = 7) -> TableOCRInput:
    image = tmp_path / "table.png"
    image.write_bytes(b"fake image")

    return TableOCRInput(
        table_id=table_id,
        image_path=image,
        provenance={
            "table_id": table_id,
            "page_nr": 3,
            "source": "mineru",
        },
    )


def make_adapter(
    *,
    device: str = "gpu:0",
    results: list[Any] | None = None,
    generator_error: Exception | None = None,
):
    loader = FakeModelLoader()
    generator = FakeGenerator(results, error=generator_error)
    input_factory = FakeInputFactory()
    fake_image = object()

    adapter = ChandraAdapter(
        device=device,
        model_loader=loader,
        generator=generator,
        input_factory=input_factory,
        image_loader=lambda _: fake_image,
    )

    return adapter, loader, generator, input_factory, fake_image


def test_capabilities_report_cpu_and_gpu_support() -> None:
    adapter, *_ = make_adapter()

    assert adapter.capabilities.cpu_supported is True
    assert adapter.capabilities.gpu_supported is True
    assert adapter.capabilities.supports_device("cpu") is True
    assert adapter.capabilities.supports_device("gpu:0") is True


def test_gpu_device_is_translated_and_ocr_prompt_is_used(tmp_path) -> None:
    generated = SimpleNamespace(
        raw="<table><tr><td>A</td></tr></table>",
        token_count=12,
        error=False,
    )
    adapter, loader, generator, input_factory, fake_image = make_adapter(
        results=[generated]
    )
    table = make_input(tmp_path)

    result = adapter.extract(table)

    assert loader.calls == ["cuda:0"]
    assert input_factory.calls == [
        {
            "image": fake_image,
            "prompt_type": "ocr",
        }
    ]
    assert len(generator.calls) == 1
    assert generator.calls[0][1] is loader.model
    assert result.status == "ok"


def test_cpu_device_is_translated_to_cpu() -> None:
    adapter, loader, *_ = make_adapter(device="cpu")

    adapter._get_model()

    assert loader.calls == ["cpu"]


def test_result_preserves_identity_provenance_and_native_output(tmp_path) -> None:
    generated = SimpleNamespace(
        raw=(
            '<table><tr><td rowspan="2">A</td><td>B</td></tr>'
            "<tr><td>C</td></tr></table>"
        ),
        token_count=42,
        error=False,
    )
    adapter, *_ = make_adapter(results=[generated])
    table = make_input(tmp_path, table_id=12)

    result = adapter.extract(table)

    assert result.table_id == 12
    assert result.adapter_name == "chandra"
    assert result.model_version == "datalab-to/chandra-ocr-2"
    assert result.device == "gpu:0"
    assert result.source_image == table.image_path
    assert result.provenance["page_nr"] == 3
    assert result.result_count == 1
    assert result.native_json == [
        {
            "raw": generated.raw,
            "token_count": 42,
            "error": False,
        }
    ]
    assert result.native_markdown == [generated.raw]


def test_empty_generation_list_is_explicit(tmp_path) -> None:
    adapter, *_ = make_adapter(results=[])

    result = adapter.extract(make_input(tmp_path))

    assert result.status == "empty"
    assert result.result_count == 0
    assert result.error is not None


def test_empty_generated_text_is_explicit(tmp_path) -> None:
    generated = SimpleNamespace(raw="   ", token_count=0, error=False)
    adapter, *_ = make_adapter(results=[generated])

    result = adapter.extract(make_input(tmp_path))

    assert result.status == "empty"
    assert result.result_count == 1
    assert result.error is not None


def test_reported_generation_error_is_explicit(tmp_path) -> None:
    generated = SimpleNamespace(raw="", token_count=0, error=True)
    adapter, *_ = make_adapter(results=[generated])

    result = adapter.extract(make_input(tmp_path))

    assert result.status == "error"
    assert result.result_count == 1
    assert result.error is not None


def test_inference_failure_is_explicit(tmp_path) -> None:
    adapter, *_ = make_adapter(
        generator_error=RuntimeError("inference exploded")
    )

    result = adapter.extract(make_input(tmp_path))

    assert result.status == "error"
    assert "inference exploded" in result.error


def test_model_instance_is_reused(tmp_path) -> None:
    generated = SimpleNamespace(
        raw="<table><tr><td>A</td></tr></table>",
        token_count=5,
        error=False,
    )
    adapter, loader, generator, *_ = make_adapter(results=[generated])

    first = make_input(tmp_path, table_id=1)
    second_image = tmp_path / "table2.png"
    second_image.write_bytes(b"fake image")
    second = TableOCRInput(
        table_id=2,
        image_path=second_image,
    )

    adapter.extract(first)
    adapter.extract(second)

    assert loader.calls == ["cuda:0"]
    assert len(generator.calls) == 2


def test_missing_image_does_not_initialize_model(tmp_path) -> None:
    adapter, loader, *_ = make_adapter()
    table = TableOCRInput(
        table_id=1,
        image_path=tmp_path / "missing.png",
    )

    result = adapter.extract(table)

    assert result.status == "error"
    assert "not found" in result.error
    assert loader.calls == []


def test_multiple_generation_results_are_rejected(tmp_path) -> None:
    generated = SimpleNamespace(
        raw="<table><tr><td>A</td></tr></table>",
        token_count=5,
        error=False,
    )
    adapter, *_ = make_adapter(results=[generated, generated])

    result = adapter.extract(make_input(tmp_path))

    assert result.status == "error"
    assert result.result_count == 2
    assert "multiple" in result.error.lower()


def test_missing_chandra_has_actionable_error(tmp_path, monkeypatch) -> None:
    real_import_module = chandra_module.importlib.import_module

    def fake_import_module(name: str):
        if name == "chandra.model.hf":
            raise ImportError("not installed")
        return real_import_module(name)

    monkeypatch.setattr(
        chandra_module.importlib,
        "import_module",
        fake_import_module,
    )

    adapter = ChandraAdapter(
        image_loader=lambda _: object(),
        input_factory=lambda **kwargs: SimpleNamespace(**kwargs),
        generator=lambda batch, model: [],
    )

    with pytest.raises(
        TableOCRDependencyError,
        match="chandra-ocr",
    ):
        adapter.extract(make_input(tmp_path))


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported Chandra device"):
        ChandraAdapter(device="cuda:0")
