from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tabulus.table_ocr import TableOCRDependencyError, TableOCRInput
from tabulus.table_ocr.paddleocr_vl import PaddleOCRVLAdapter
import tabulus.table_ocr.paddleocr_vl as paddle_module


class FakeArray:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def tolist(self) -> list[int]:
        return self.values


class FakePipeline:
    def __init__(
        self,
        results: list[Any] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.results = results if results is not None else []
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def predict(self, image: str, **kwargs: Any) -> list[Any]:
        self.calls.append((image, kwargs))

        if self.error is not None:
            raise self.error

        return self.results


class FakeFactory:
    def __init__(self, pipeline: FakePipeline) -> None:
        self.pipeline = pipeline
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> FakePipeline:
        self.calls.append(kwargs)
        return self.pipeline


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


def test_capabilities_report_cpu_and_gpu_support() -> None:
    adapter = PaddleOCRVLAdapter(pipeline_factory=lambda **_: object())

    assert adapter.capabilities.cpu_supported is True
    assert adapter.capabilities.gpu_supported is True
    assert adapter.capabilities.supports_device("cpu") is True
    assert adapter.capabilities.supports_device("gpu:0") is True


def test_cpu_pipeline_and_cropped_table_prediction_arguments(tmp_path) -> None:
    result_object = SimpleNamespace(
        json={"parsing_res_list": []},
        markdown={"text": "| A | B |"},
    )
    pipeline = FakePipeline([result_object])
    factory = FakeFactory(pipeline)
    adapter = PaddleOCRVLAdapter(
        device="cpu",
        engine="paddle",
        pipeline_factory=factory,
    )
    table = make_input(tmp_path)

    result = adapter.extract(table)

    assert factory.calls == [
        {
            "pipeline_version": "v1.6",
            "device": "cpu",
            "engine": "paddle",
            "use_layout_detection": False,
        }
    ]
    assert pipeline.calls == [
        (
            str(table.image_path),
            {
                "use_layout_detection": False,
                "prompt_label": "table",
            },
        )
    ]
    assert result.status == "ok"


def test_result_preserves_identity_provenance_and_native_outputs(tmp_path) -> None:
    result_object = SimpleNamespace(
        json={
            "parsing_res_list": [
                {
                    "block_label": "table",
                    "block_bbox": FakeArray([1, 2, 3, 4]),
                }
            ]
        },
        markdown={"text": "| A | B |\n|---|---|\n| 1 | 2 |"},
    )
    pipeline = FakePipeline([result_object])
    adapter = PaddleOCRVLAdapter(
        pipeline_factory=FakeFactory(pipeline),
    )
    table = make_input(tmp_path, table_id=12)

    result = adapter.extract(table)

    assert result.table_id == 12
    assert result.adapter_name == "paddleocr-vl"
    assert result.model_version == "PaddleOCR-VL v1.6"
    assert result.device == "cpu"
    assert result.source_image == table.image_path
    assert result.provenance["page_nr"] == 3
    assert result.result_count == 1
    assert result.native_json[0]["parsing_res_list"][0]["block_bbox"] == [
        1,
        2,
        3,
        4,
    ]
    assert "A" in result.native_markdown[0]["text"]


def test_empty_output_is_explicit(tmp_path) -> None:
    pipeline = FakePipeline([])
    adapter = PaddleOCRVLAdapter(
        pipeline_factory=FakeFactory(pipeline),
    )

    result = adapter.extract(make_input(tmp_path))

    assert result.status == "empty"
    assert result.result_count == 0
    assert result.native_json == []
    assert result.native_markdown == []
    assert result.error is not None


def test_inference_failure_is_explicit(tmp_path) -> None:
    pipeline = FakePipeline(error=RuntimeError("inference exploded"))
    adapter = PaddleOCRVLAdapter(
        pipeline_factory=FakeFactory(pipeline),
    )

    result = adapter.extract(make_input(tmp_path))

    assert result.status == "error"
    assert "inference exploded" in result.error


def test_pipeline_instance_is_reused(tmp_path) -> None:
    result_object = SimpleNamespace(
        json={"result": "ok"},
        markdown={"text": "ok"},
    )
    pipeline = FakePipeline([result_object])
    factory = FakeFactory(pipeline)
    adapter = PaddleOCRVLAdapter(pipeline_factory=factory)

    first = make_input(tmp_path, table_id=1)
    second_image = tmp_path / "table2.png"
    second_image.write_bytes(b"fake image")
    second = TableOCRInput(
        table_id=2,
        image_path=second_image,
    )

    adapter.extract(first)
    adapter.extract(second)

    assert len(factory.calls) == 1
    assert len(pipeline.calls) == 2


def test_missing_image_does_not_initialize_pipeline(tmp_path) -> None:
    pipeline = FakePipeline()
    factory = FakeFactory(pipeline)
    adapter = PaddleOCRVLAdapter(pipeline_factory=factory)
    table = TableOCRInput(
        table_id=1,
        image_path=tmp_path / "missing.png",
    )

    result = adapter.extract(table)

    assert result.status == "error"
    assert "not found" in result.error
    assert factory.calls == []


def test_missing_paddleocr_has_actionable_error(tmp_path, monkeypatch) -> None:
    real_import_module = paddle_module.importlib.import_module

    def fake_import_module(name: str):
        if name == "paddleocr":
            raise ImportError("not installed")
        return real_import_module(name)

    monkeypatch.setattr(
        paddle_module.importlib,
        "import_module",
        fake_import_module,
    )

    adapter = PaddleOCRVLAdapter()

    with pytest.raises(
        TableOCRDependencyError,
        match="requires PaddleOCR and PaddlePaddle",
    ):
        adapter.extract(make_input(tmp_path))
