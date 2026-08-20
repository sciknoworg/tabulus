from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tabulus.table_ocr import (
    TableOCRDependencyError,
    TableOCRInput,
    list_table_ocr_adapters,
)
from tabulus.table_ocr.xberg_tatr import XbergTATRAdapter
import tabulus.table_ocr.xberg_tatr as xberg_module


class FakeConfigFactory:
    def __init__(self) -> None:
        self.calls = 0
        self.config = object()

    def __call__(self) -> object:
        self.calls += 1
        return self.config


class FakeInputFactory:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(**kwargs)


class FakeExtractor:
    def __init__(
        self,
        output: Any,
        *,
        error: Exception | None = None,
    ) -> None:
        self.output = output
        self.error = error
        self.calls: list[tuple[Any, Any]] = []

    def __call__(self, extract_input: Any, config: Any) -> Any:
        self.calls.append((extract_input, config))

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
            "page_nr": 6,
            "source": "mineru",
        },
    )


def make_output(
    *,
    tables: list[Any] | None = None,
    errors: list[Any] | None = None,
    content: str = "document text",
) -> Any:
    document = SimpleNamespace(
        content=content,
        tables=tables if tables is not None else [],
    )
    return SimpleNamespace(
        results=[document],
        errors=errors if errors is not None else [],
    )


def make_adapter(output: Any):
    config_factory = FakeConfigFactory()
    input_factory = FakeInputFactory()
    extractor = FakeExtractor(output)
    adapter = XbergTATRAdapter(
        config_factory=config_factory,
        input_factory=input_factory,
        extractor=extractor,
    )
    return adapter, config_factory, input_factory, extractor


def test_registry_exposes_xberg_tatr_as_cpu_only() -> None:
    specs = {
        spec.name: spec
        for spec in list_table_ocr_adapters()
    }

    spec = specs["xberg-tatr"]
    assert spec.cpu_supported is True
    assert spec.gpu_supported is False


def test_capabilities_report_cpu_only() -> None:
    adapter, *_ = make_adapter(make_output())

    assert adapter.capabilities.cpu_supported is True
    assert adapter.capabilities.gpu_supported is False
    assert adapter.capabilities.supports_device("cpu") is True
    assert adapter.capabilities.supports_device("gpu:0") is False


def test_single_table_preserves_identity_and_native_output(tmp_path) -> None:
    markdown = "| A | B |\n| --- | --- |\n| 1 | 2 |"
    extracted_table = SimpleNamespace(
        page_number=1,
        markdown=markdown,
        cells=[["A", "B"], ["1", "2"]],
    )
    adapter, config_factory, input_factory, extractor = make_adapter(
        make_output(tables=[extracted_table])
    )
    table = make_input(tmp_path, table_id=12)

    result = adapter.extract(table)

    assert result.status == "ok"
    assert result.table_id == 12
    assert result.adapter_name == "xberg-tatr"
    assert result.model_version == "RT-DETR v2 + TATR"
    assert result.device == "cpu"
    assert result.source_image == table.image_path
    assert result.provenance["page_nr"] == 6
    assert result.result_count == 1
    assert result.native_markdown == [markdown]
    assert result.native_json[0]["content"] == "document text"
    assert result.native_json[0]["tables"][0]["cells"] == [
        ["A", "B"],
        ["1", "2"],
    ]
    assert config_factory.calls == 1
    assert input_factory.calls == [
        {
            "kind": "uri",
            "uri": str(table.image_path),
        }
    ]
    assert extractor.calls[0][1] is config_factory.config


def test_cells_are_used_when_xberg_markdown_is_empty(tmp_path) -> None:
    extracted_table = SimpleNamespace(
        page_number=1,
        markdown="",
        cells=[["A", "B"], ["1", "2"]],
    )
    adapter, *_ = make_adapter(
        make_output(tables=[extracted_table])
    )

    result = adapter.extract(make_input(tmp_path))

    assert result.status == "ok"
    assert result.native_markdown == [
        "| A | B |\n| --- | --- |\n| 1 | 2 |"
    ]


def test_zero_tables_is_explicit_empty_result(tmp_path) -> None:
    adapter, *_ = make_adapter(make_output(tables=[]))

    result = adapter.extract(make_input(tmp_path))

    assert result.status == "empty"
    assert result.result_count == 0
    assert result.native_json[0]["content"] == "document text"
    assert result.error is not None


def test_multiple_tables_are_rejected(tmp_path) -> None:
    first = SimpleNamespace(
        page_number=1,
        markdown="| A |\n| --- |\n| 1 |",
        cells=[["A"], ["1"]],
    )
    second = SimpleNamespace(
        page_number=1,
        markdown="| B |\n| --- |\n| 2 |",
        cells=[["B"], ["2"]],
    )
    adapter, *_ = make_adapter(
        make_output(tables=[first, second])
    )

    result = adapter.extract(make_input(tmp_path))

    assert result.status == "error"
    assert result.result_count == 2
    assert "multiple" in result.error.lower()


def test_xberg_errors_are_preserved(tmp_path) -> None:
    adapter, *_ = make_adapter(
        make_output(errors=[{"message": "OCR failed"}])
    )

    result = adapter.extract(make_input(tmp_path))

    assert result.status == "error"
    assert result.native_json == [
        {"errors": [{"message": "OCR failed"}]}
    ]


def test_extraction_failure_is_explicit(tmp_path) -> None:
    output = make_output()
    config_factory = FakeConfigFactory()
    input_factory = FakeInputFactory()
    extractor = FakeExtractor(
        output,
        error=RuntimeError("extraction exploded"),
    )
    adapter = XbergTATRAdapter(
        config_factory=config_factory,
        input_factory=input_factory,
        extractor=extractor,
    )

    result = adapter.extract(make_input(tmp_path))

    assert result.status == "error"
    assert "extraction exploded" in result.error


def test_config_instance_is_reused(tmp_path) -> None:
    extracted_table = SimpleNamespace(
        page_number=1,
        markdown="| A |\n| --- |\n| 1 |",
        cells=[["A"], ["1"]],
    )
    adapter, config_factory, _, extractor = make_adapter(
        make_output(tables=[extracted_table])
    )

    first = make_input(tmp_path, table_id=1)
    second_image = tmp_path / "table2.jpg"
    second_image.write_bytes(b"fake image")
    second = TableOCRInput(
        table_id=2,
        image_path=second_image,
    )

    adapter.extract(first)
    adapter.extract(second)

    assert config_factory.calls == 1
    assert len(extractor.calls) == 2


def test_missing_image_does_not_initialize_xberg(tmp_path) -> None:
    adapter, config_factory, _, extractor = make_adapter(make_output())
    table = TableOCRInput(
        table_id=1,
        image_path=tmp_path / "missing.jpg",
    )

    result = adapter.extract(table)

    assert result.status == "error"
    assert "not found" in result.error
    assert config_factory.calls == 0
    assert extractor.calls == []


def test_missing_xberg_has_actionable_error(tmp_path, monkeypatch) -> None:
    real_import_module = xberg_module.importlib.import_module

    def fake_import_module(name: str):
        if name == "xberg":
            raise ImportError("not installed")
        return real_import_module(name)

    monkeypatch.setattr(
        xberg_module.importlib,
        "import_module",
        fake_import_module,
    )

    adapter = XbergTATRAdapter()

    with pytest.raises(
        TableOCRDependencyError,
        match="xberg==1.0.14",
    ):
        adapter.extract(make_input(tmp_path))


def test_gpu_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="CPU-only"):
        XbergTATRAdapter(device="gpu:0")
