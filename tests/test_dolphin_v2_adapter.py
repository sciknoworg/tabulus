from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tabulus.table_ocr.base import TableOCRInput
from tabulus.table_ocr.dolphin_v2 import (
    DolphinV2Adapter,
    MODEL_VERSION,
    _dolphin_resize,
)
from tabulus.table_ocr.output import parse_result_tables
from tabulus.table_ocr.registry import list_table_ocr_adapters


SIMPLE_HTML = (
    "<table>"
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
            model_class="Qwen2_5_VLForConditionalGeneration",
            transformers_version="4.51.0",
            torch_version="2.6.0",
            qwen_vl_utils_version="0.0.14",
            pillow_version="11.1.0",
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
            "raw_output": SIMPLE_HTML + "<|im_end|>",
            "clean_output": SIMPLE_HTML,
            "source_image_size": [1012, 1903],
            "model_input_image_size": [850, 1600],
            "prompt_tokens": 1738,
            "generated_tokens": 2317,
        }
        self.calls: list[tuple[Any, Any]] = []

    def __call__(
        self,
        image_path,
        runtime: Any,
    ) -> dict[str, Any]:
        self.calls.append((image_path, runtime))
        return self.payload


class FakeImage:
    def __init__(
        self,
        size: tuple[int, int],
    ) -> None:
        self.size = size
        self.resize_calls: list[tuple[int, int]] = []

    def resize(
        self,
        size: tuple[int, int],
    ) -> "FakeImage":
        self.resize_calls.append(size)
        return FakeImage(size)


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

    adapter = DolphinV2Adapter(
        device=device,
        runtime_loader=loader,
        inference_runner=runner,
    )

    return adapter, loader, runner


def test_registry_reports_dolphin_v2() -> None:
    specs = {
        spec.name: spec
        for spec in list_table_ocr_adapters()
    }

    spec = specs["dolphin-v2"]

    assert spec.display_name == "Dolphin-v2"
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
    with pytest.raises(
        ValueError,
        match="gpu:<index>",
    ):
        DolphinV2Adapter(device="cpu")


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="gpu:<index>",
    ):
        DolphinV2Adapter(device="cuda:0")


def test_official_resize_matches_validated_crop_geometry() -> None:
    image = FakeImage((1012, 1903))

    resized = _dolphin_resize(image)

    assert image.resize_calls == [(850, 1600)]
    assert resized.size == (850, 1600)


def test_result_preserves_versions_and_native_evidence(
    tmp_path,
) -> None:
    adapter, *_ = make_adapter()

    result = adapter.extract(
        make_input(tmp_path, 12)
    )

    assert result.status == "ok"
    assert result.table_id == 12
    assert result.adapter_name == "dolphin-v2"
    assert result.adapter_version == "4.51.0"
    assert result.model_version == MODEL_VERSION
    assert result.result_count == 1

    native = result.native_json[0]["dolphin_v2"]

    assert native["model_repo"] == "ByteDance/Dolphin-v2"
    assert native["model_revision"] == (
        "c37c62768c644bb594da4283149c627765aa80f3"
    )
    assert native["backbone_architecture"] == "Qwen2.5-VL"
    assert native["model_class"] == (
        "Qwen2_5_VLForConditionalGeneration"
    )
    assert native["prompt"] == "Parse the table in the image."
    assert native["model_dtype"] == "bfloat16"
    assert native["resolved_model_dtype"] == "torch.bfloat16"
    assert native["max_new_tokens"] == 4096
    assert native["generation_do_sample"] is False
    assert native["generation_temperature"] is None
    assert native["execution_device"] == "cuda:0"
    assert native["transformers_version"] == "4.51.0"
    assert native["torch_version"] == "2.6.0"
    assert native["qwen_vl_utils_version"] == "0.0.14"
    assert native["source_image_size"] == [1012, 1903]
    assert native["model_input_image_size"] == [850, 1600]
    assert native["prompt_tokens"] == 1738
    assert native["generated_tokens"] == 2317
    assert native["raw_output"].endswith("<|im_end|>")
    assert native["clean_output"] == SIMPLE_HTML
    assert native["native_format"] == "html"
    assert native["special_tokens_removed_for_parsing"] is True
    assert native["normalization"] == "none"
    assert native["html_tables_detected"] == 1
    assert native["input_policy"] == "canonical_mineru_crop"
    assert native["image_preprocessing"] == {
        "rgb_conversion": True,
        "resize": "official_dolphin_resize_img",
        "max_size": 1600,
        "min_size": 28,
        "margin_crop": False,
    }
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


def test_empty_generated_output_is_explicit(
    tmp_path,
) -> None:
    payload = {
        "raw_output": "<|im_end|>",
        "clean_output": "",
        "source_image_size": [100, 50],
        "model_input_image_size": [100, 50],
        "prompt_tokens": 20,
        "generated_tokens": 1,
    }

    adapter, *_ = make_adapter(payload=payload)

    result = adapter.extract(
        make_input(tmp_path)
    )

    assert result.status == "empty"
    assert "no generated table output" in result.error


def test_non_html_output_is_explicit(
    tmp_path,
) -> None:
    payload = {
        "raw_output": "not a table<|im_end|>",
        "clean_output": "not a table",
        "source_image_size": [100, 50],
        "model_input_image_size": [100, 50],
        "prompt_tokens": 20,
        "generated_tokens": 4,
    }

    adapter, *_ = make_adapter(payload=payload)

    result = adapter.extract(
        make_input(tmp_path)
    )

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
        "raw_output": html + "<|im_end|>",
        "clean_output": html,
        "source_image_size": [100, 50],
        "model_input_image_size": [100, 50],
        "prompt_tokens": 20,
        "generated_tokens": 20,
    }

    adapter, *_ = make_adapter(payload=payload)

    result = adapter.extract(
        make_input(tmp_path)
    )

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
