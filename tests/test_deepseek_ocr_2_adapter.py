from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tabulus.table_ocr.base import TableOCRInput
from tabulus.table_ocr.deepseek_ocr_2 import (
    DeepSeekOCR2Adapter,
    MODEL_VERSION,
)
from tabulus.table_ocr.output import parse_result_tables
from tabulus.table_ocr.registry import list_table_ocr_adapters


SIMPLE_HTML = (
    "<table>"
    "<tr><td>A</td><td>B</td></tr>"
    "<tr><td>1</td><td>2</td></tr>"
    "</table>"
)

GROUNDED_HTML = (
    "<|ref|>table<|/ref|>"
    "<|det|>[[0, 0, 999, 999]]<|/det|>\n"
    + SIMPLE_HTML
)


class FakeRuntimeLoader:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.runtime = SimpleNamespace(
            model_device=None,
            model_dtype="torch.bfloat16",
            model_class="DeepseekOCR2ForCausalLM",
            transformers_version="4.46.3",
            tokenizers_version="0.20.3",
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
            "raw_output": GROUNDED_HTML,
            "clean_output": GROUNDED_HTML,
            "source_image_size": [1012, 1903],
            "decoded_output_tokens": 1404,
            "output_chars": len(GROUNDED_HTML),
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

    adapter = DeepSeekOCR2Adapter(
        device=device,
        runtime_loader=loader,
        inference_runner=runner,
    )

    return adapter, loader, runner


def test_registry_reports_deepseek_ocr_2() -> None:
    specs = {
        spec.name: spec
        for spec in list_table_ocr_adapters()
    }

    spec = specs["deepseek-ocr-2"]

    assert spec.display_name == "DeepSeek-OCR-2"
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
        DeepSeekOCR2Adapter(device="cpu")


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="gpu:<index>",
    ):
        DeepSeekOCR2Adapter(device="cuda:0")


def test_result_preserves_versions_and_native_evidence(
    tmp_path,
) -> None:
    adapter, *_ = make_adapter()

    result = adapter.extract(
        make_input(tmp_path, 12)
    )

    assert result.status == "ok"
    assert result.table_id == 12
    assert result.adapter_name == "deepseek-ocr-2"
    assert result.adapter_version == "4.46.3"
    assert result.model_version == MODEL_VERSION
    assert result.result_count == 1

    native = result.native_json[0]["deepseek_ocr_2"]

    assert native["model_repo"] == "deepseek-ai/DeepSeek-OCR-2"
    assert native["model_revision"] == (
        "aaa02f3811945a91062062994c5c4a3f4c0af2b0"
    )
    assert native["model_class"] == "DeepseekOCR2ForCausalLM"
    assert native["prompt"] == (
        "<image>\n"
        "<|grounding|>Convert the document to markdown."
    )
    assert native["model_dtype"] == "bfloat16"
    assert native["resolved_model_dtype"] == "torch.bfloat16"
    assert native["attention_implementation"] == (
        "flash_attention_2"
    )
    assert native["max_new_tokens"] == 8192
    assert native["generation_do_sample"] is False
    assert native["generation_temperature"] == 0.0
    assert native["no_repeat_ngram_size"] == 35
    assert native["generation_use_cache"] is True
    assert native["official_infer_eval_mode"] is True
    assert native["official_infer_save_results"] is False
    assert native["execution_device"] == "cuda:0"
    assert native["transformers_version"] == "4.46.3"
    assert native["tokenizers_version"] == "0.20.3"
    assert native["flash_attn_version"] == "2.7.3"
    assert native["torch_version"] == "2.6.0"
    assert native["torchvision_version"] == "0.21.0"
    assert native["source_image_size"] == [1012, 1903]
    assert native["decoded_output_tokens"] == 1404
    assert native["output_chars"] == len(GROUNDED_HTML)
    assert native["raw_output"] == GROUNDED_HTML
    assert native["clean_output"] == GROUNDED_HTML
    assert native["native_format"] == (
        "deepseek_document_markdown_with_grounding"
    )
    assert native["normalization"] == "none"
    assert native["parser_input"] == (
        "model_infer_output_unchanged"
    )
    assert native["html_tables_detected"] == 1
    assert native["structured_tables_detected"] == 1
    assert native["parser_error"] is None
    assert native["input_policy"] == "canonical_mineru_crop"
    assert native["image_preprocessing"] == {
        "external": "none",
        "model_internal": "deepseek_ocr_2_dynamic_resolution",
        "base_size": 1024,
        "image_size": 768,
        "crop_mode": True,
        "model_internal_tiling": True,
    }
    assert native["layout_redetection"] is False
    assert native["recropping"] is False
    assert native["external_recropping"] is False
    assert native["trust_remote_code"] is True


def test_shared_parser_receives_grounded_output_unchanged(
    tmp_path,
) -> None:
    adapter, *_ = make_adapter()

    result = adapter.extract(
        make_input(tmp_path)
    )

    assert result.native_markdown == [GROUNDED_HTML]
    assert "<|ref|>table<|/ref|>" in result.native_markdown[0]

    parsed = parse_result_tables(result)

    assert len(parsed) == 1
    assert parsed[0].source == "html"
    assert parsed[0].rows == [
        ["A", "B"],
        ["1", "2"],
    ]


def test_markdown_table_output_is_accepted(
    tmp_path,
) -> None:
    markdown = "| A | B |\n| --- | --- |\n| 1 | 2 |"

    payload = {
        "raw_output": markdown,
        "clean_output": markdown,
        "source_image_size": [100, 50],
        "decoded_output_tokens": 16,
        "output_chars": len(markdown),
    }

    adapter, *_ = make_adapter(payload=payload)

    result = adapter.extract(
        make_input(tmp_path)
    )

    parsed = parse_result_tables(result)

    assert result.status == "ok"
    assert result.result_count == 1
    assert len(parsed) == 1
    assert parsed[0].rows == [
        ["A", "B"],
        ["1", "2"],
    ]


def test_empty_generated_output_is_explicit(
    tmp_path,
) -> None:
    payload = {
        "raw_output": "",
        "clean_output": "",
        "source_image_size": [100, 50],
        "decoded_output_tokens": 0,
        "output_chars": 0,
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
        "raw_output": "not a table",
        "clean_output": "not a table",
        "source_image_size": [100, 50],
        "decoded_output_tokens": 3,
        "output_chars": 11,
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
        "raw_output": html,
        "clean_output": html,
        "source_image_size": [100, 50],
        "decoded_output_tokens": 20,
        "output_chars": len(html),
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
