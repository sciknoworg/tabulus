from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tabulus.table_ocr.base import TableOCRInput
from tabulus.table_ocr.nanonets_ocr_s import (
    MODEL_VERSION,
    NanonetsOCRSAdapter,
)
from tabulus.table_ocr.output import parse_result_tables
from tabulus.table_ocr.registry import list_table_ocr_adapters


SIMPLE_HTML = (
    "<table>"
    "<thead><tr><th>A</th><th>B</th></tr></thead>"
    "<tbody><tr><td>1</td><td>2</td></tr></tbody>"
    "</table>"
)

RICH_HTML = (
    "<table>"
    "<thead>"
    "<tr><th>Material</th><th>Reference</th></tr>"
    "</thead>"
    "<tbody>"
    "<tr>"
    '<td rowspan="2">Al<sub>2</sub>O<sub>3</sub></td>'
    "<td><sup>41</sup></td>"
    "</tr>"
    "<tr><td><sup>59</sup><br><sup>86</sup></td></tr>"
    "</tbody>"
    "</table>"
)


class FakeRuntimeLoader:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.runtime = SimpleNamespace(
            model_device=None,
            model_dtype="torch.bfloat16",
            model_class="Qwen2_5_VLForConditionalGeneration",
            transformers_version="4.52.4",
            tokenizers_version="0.21.4",
            flash_attn_version="2.7.3",
            torch_version="2.6.0",
            torchvision_version="0.21.0",
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
            "prompt_tokens": 2606,
            "generated_tokens": 3414,
            "raw_output_chars": len(SIMPLE_HTML) + 10,
            "clean_output_chars": len(SIMPLE_HTML),
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

    adapter = NanonetsOCRSAdapter(
        device=device,
        runtime_loader=loader,
        inference_runner=runner,
    )

    return adapter, loader, runner


def test_registry_reports_nanonets_ocr_s() -> None:
    specs = {
        spec.name: spec
        for spec in list_table_ocr_adapters()
    }

    spec = specs["nanonets-ocr-s"]

    assert spec.display_name == "Nanonets-OCR-s"
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
        NanonetsOCRSAdapter(device="cpu")


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="gpu:<index>",
    ):
        NanonetsOCRSAdapter(device="cuda:0")


def test_result_preserves_versions_and_native_evidence(
    tmp_path,
) -> None:
    adapter, *_ = make_adapter()

    result = adapter.extract(
        make_input(tmp_path, 12)
    )

    assert result.status == "ok"
    assert result.table_id == 12
    assert result.adapter_name == "nanonets-ocr-s"
    assert result.adapter_version == "4.52.4"
    assert result.model_version == MODEL_VERSION
    assert result.result_count == 1

    native = result.native_json[0]["nanonets_ocr_s"]

    assert native["model_repo"] == "nanonets/Nanonets-OCR-s"
    assert native["model_revision"] == (
        "3baad182cc87c65a1861f0c30357d3467e978172"
    )
    assert native["backbone_architecture"] == "Qwen2.5-VL"
    assert native["model_class"] == (
        "Qwen2_5_VLForConditionalGeneration"
    )
    assert native["system_prompt"] == "You are a helpful assistant."
    assert "Return the tables in html format." in native["prompt"]
    assert native["model_dtype"] == "bfloat16"
    assert native["resolved_model_dtype"] == "torch.bfloat16"
    assert native["attention_implementation"] == (
        "flash_attention_2"
    )
    assert native["processor_use_fast"] is False
    assert native["max_new_tokens"] == 15000
    assert native["generation_do_sample"] is False
    assert native["execution_device"] == "cuda:0"
    assert native["transformers_version"] == "4.52.4"
    assert native["tokenizers_version"] == "0.21.4"
    assert native["flash_attn_version"] == "2.7.3"
    assert native["torch_version"] == "2.6.0"
    assert native["torchvision_version"] == "0.21.0"
    assert native["source_image_size"] == [1012, 1903]
    assert native["prompt_tokens"] == 2606
    assert native["generated_tokens"] == 3414
    assert native["raw_output"].endswith("<|im_end|>")
    assert native["clean_output"] == SIMPLE_HTML
    assert native["native_format"] == "nanonets_document_markup"
    assert native["special_tokens_removed_for_parsing"] is True
    assert native["normalization"] == "none"
    assert native["parser_input"] == (
        "decoded_output_special_tokens_removed"
    )
    assert native["html_tables_detected"] == 1
    assert native["structured_tables_detected"] == 1
    assert native["parser_error"] is None
    assert native["input_policy"] == "canonical_mineru_crop"
    assert native["image_preprocessing"] == {
        "external": "rgb_conversion_only",
        "processor": "AutoProcessor",
        "processor_use_fast": False,
        "model_internal_resize": True,
    }
    assert native["layout_redetection"] is False
    assert native["recropping"] is False
    assert native["external_recropping"] is False


def test_shared_parser_accepts_rich_html_unchanged(
    tmp_path,
) -> None:
    payload = {
        "raw_output": RICH_HTML + "<|im_end|>",
        "clean_output": RICH_HTML,
        "source_image_size": [1012, 1903],
        "prompt_tokens": 2606,
        "generated_tokens": 200,
        "raw_output_chars": len(RICH_HTML) + 10,
        "clean_output_chars": len(RICH_HTML),
    }

    adapter, *_ = make_adapter(payload=payload)

    result = adapter.extract(
        make_input(tmp_path)
    )

    parsed = parse_result_tables(result)

    assert result.native_markdown == [RICH_HTML]
    assert len(parsed) == 1
    assert parsed[0].source == "html"
    assert parsed[0].rows == [
        ["Material", "Reference"],
        ["Al2O3", "41"],
        ["", "5986"],
    ]


def test_empty_generated_output_is_explicit(
    tmp_path,
) -> None:
    payload = {
        "raw_output": "<|im_end|>",
        "clean_output": "",
        "source_image_size": [100, 50],
        "prompt_tokens": 100,
        "generated_tokens": 1,
        "raw_output_chars": 10,
        "clean_output_chars": 0,
    }

    adapter, *_ = make_adapter(payload=payload)

    result = adapter.extract(
        make_input(tmp_path)
    )

    assert result.status == "empty"
    assert "no generated table output" in result.error


def test_non_structured_output_is_explicit(
    tmp_path,
) -> None:
    payload = {
        "raw_output": "not a table<|im_end|>",
        "clean_output": "not a table",
        "source_image_size": [100, 50],
        "prompt_tokens": 100,
        "generated_tokens": 4,
        "raw_output_chars": 21,
        "clean_output_chars": 11,
    }

    adapter, *_ = make_adapter(payload=payload)

    result = adapter.extract(
        make_input(tmp_path)
    )

    assert result.status == "empty"
    assert "usable structured table" in result.error


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
        "prompt_tokens": 100,
        "generated_tokens": 20,
        "raw_output_chars": len(html) + 10,
        "clean_output_chars": len(html),
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
