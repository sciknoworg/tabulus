from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tabulus.table_ocr.base import TableOCRInput
from tabulus.table_ocr.internvl3_5_8b import (
    InternVL35_8BAdapter,
    MAX_NEW_TOKENS,
    MODEL_VERSION,
    TABLE_PROMPT,
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
            model_class="InternVLForConditionalGeneration",
            model_type="internvl",
            text_model_type="qwen3",
            vision_model_type="internvl_vision",
            processor_class="InternVLProcessor",
            image_processor_class="GotOcr2ImageProcessorFast",
            tokenizer_class="Qwen2TokenizerFast",
            image_seq_length=256,
            attention_implementation="sdpa",
            text_attention_implementation="sdpa",
            vision_attention_implementation="sdpa",
            generation_do_sample=False,
            generation_num_beams=1,
            generation_temperature=1.0,
            generation_top_p=1.0,
            generation_repetition_penalty=1.0,
            generation_bos_token_id=151643,
            generation_eos_token_id=151645,
            transformers_version="4.55.0",
            accelerate_version="1.14.0",
            torch_version="2.7.0+cu128",
            torchvision_version="0.22.0+cu128",
        )

    def __call__(self, device: str) -> Any:
        self.calls.append(device)
        self.runtime.model_device = device
        return self.runtime


class FakeInferenceRunner:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {
            "output": SIMPLE_HTML,
            "image_size": [200, 100],
            "pixel_values_shape": [4, 3, 448, 448],
            "pixel_values_dtype": "torch.bfloat16",
            "prompt_tokens": 25,
            "generated_tokens": 30,
            "hit_token_ceiling": False,
        }
        self.calls: list[tuple[Any, Any]] = []

    def __call__(self, image_path, runtime: Any) -> dict[str, Any]:
        self.calls.append((image_path, runtime))
        return self.payload


def make_input(tmp_path, table_id: int = 7) -> TableOCRInput:
    image = tmp_path / "table.jpg"
    image.write_bytes(b"fake image")
    return TableOCRInput(
        table_id=table_id,
        image_path=image,
        provenance={"table_id": table_id, "source": "mineru"},
    )


def make_adapter(
    *, device: str = "gpu:0", payload: dict[str, Any] | None = None
):
    loader = FakeRuntimeLoader()
    runner = FakeInferenceRunner(payload)
    adapter = InternVL35_8BAdapter(
        device=device,
        runtime_loader=loader,
        inference_runner=runner,
    )
    return adapter, loader, runner


def test_registry_reports_internvl3_5_8b() -> None:
    specs = {spec.name: spec for spec in list_table_ocr_adapters()}
    spec = specs["internvl3-5-8b"]
    assert spec.display_name == "InternVL3.5-8B"
    assert spec.cpu_supported is False
    assert spec.gpu_supported is True


def test_gpu_device_translation_and_runtime_reuse(tmp_path) -> None:
    adapter, loader, runner = make_adapter()
    first = make_input(tmp_path, 1)
    second_path = tmp_path / "second.jpg"
    second_path.write_bytes(b"fake image")
    second = TableOCRInput(table_id=2, image_path=second_path)

    assert adapter.extract(first).status == "ok"
    assert adapter.extract(second).status == "ok"
    assert loader.calls == ["cuda:0"]
    assert len(runner.calls) == 2


def test_cpu_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="gpu:<index>"):
        InternVL35_8BAdapter(device="cpu")


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="gpu:<index>"):
        InternVL35_8BAdapter(device="cuda:0")


def test_result_preserves_frozen_runtime_and_method_provenance(tmp_path) -> None:
    adapter, *_ = make_adapter()
    result = adapter.extract(make_input(tmp_path, 12))

    assert result.status == "ok"
    assert result.table_id == 12
    assert result.adapter_name == "internvl3-5-8b"
    assert result.adapter_version == "4.55.0"
    assert result.model_version == MODEL_VERSION
    assert result.result_count == 1

    native = result.native_json[0]["internvl3_5_8b"]
    assert native["model_repo"] == "OpenGVLab/InternVL3_5-8B-HF"
    assert native["model_revision"] == (
        "741a7d03020411e666c6109218ab71e08151ef86"
    )
    assert native["model_class"] == "InternVLForConditionalGeneration"
    assert native["model_type"] == "internvl"
    assert native["text_model_type"] == "qwen3"
    assert native["vision_model_type"] == "internvl_vision"
    assert native["processor_class"] == "InternVLProcessor"
    assert native["image_processor_class"] == "GotOcr2ImageProcessorFast"
    assert native["tokenizer_class"] == "Qwen2TokenizerFast"
    assert native["image_seq_length"] == 256
    assert native["task"] == "table_to_html"
    assert native["prompt_source"] == "tabulus_defined"
    assert native["prompt"] == TABLE_PROMPT
    assert native["model_load_dtype"] == "bfloat16"
    assert native["resolved_model_dtype"] == "torch.bfloat16"
    assert native["attention_implementation"] == "sdpa"
    assert native["text_attention_implementation"] == "sdpa"
    assert native["vision_attention_implementation"] == "sdpa"
    assert native["max_new_tokens"] == MAX_NEW_TOKENS == 8192
    assert native["do_sample"] is False
    assert native["num_beams"] == 1
    assert native["temperature"] == 1.0
    assert native["top_p"] == 1.0
    assert native["repetition_penalty"] == 1.0
    defaults = native["resolved_generation_defaults"]
    assert defaults == {
        "do_sample": False,
        "num_beams": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
    }
    assert native["execution_device"] == "cuda:0"
    assert native["transformers_version"] == "4.55.0"
    assert native["accelerate_version"] == "1.14.0"
    assert native["torch_version"] == "2.7.0+cu128"
    assert native["torchvision_version"] == "0.22.0+cu128"
    assert native["local_files_only"] is True
    assert native["image_size"] == [200, 100]
    assert native["pixel_values_shape"] == [4, 3, 448, 448]
    assert native["pixel_values_dtype"] == "torch.bfloat16"
    assert native["prompt_tokens"] == 25
    assert native["generated_tokens"] == 30
    assert native["hit_token_ceiling"] is False
    assert native["output"] == SIMPLE_HTML
    assert native["native_format"] == "html"
    assert native["html_tables_detected"] == 1
    assert native["normalization"] == "none"
    assert native["model_native_image_processing"] is True
    assert native["official_hf_processor_only"] is True
    assert native["external_layout_redetection"] is False
    assert native["external_table_redetection"] is False
    assert native["external_recropping"] is False
    assert native["bbox_recropping"] is False
    assert native["tabulus_content_aware_tiling"] is False
    assert native["semantic_repair"] is False
    assert native["continued_table_merging"] is False
    assert native["reference_resolution"] is False
    assert native["input_policy"] == "canonical_mineru_crop"


def test_shared_parser_receives_native_html_unchanged(tmp_path) -> None:
    adapter, *_ = make_adapter()
    result = adapter.extract(make_input(tmp_path))
    parsed = parse_result_tables(result)

    assert result.status == "ok"
    assert result.native_markdown == [SIMPLE_HTML]
    assert len(parsed) == 1
    assert parsed[0].source == "html"
    assert parsed[0].rows == [["A", "B"], ["1", "2"]]


def test_empty_generated_output_is_explicit(tmp_path) -> None:
    payload = {
        "output": "",
        "image_size": [100, 50],
        "prompt_tokens": 20,
        "generated_tokens": 1,
        "hit_token_ceiling": False,
    }
    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))

    assert result.status == "empty"
    assert "no generated table output" in result.error
    assert result.native_markdown == []


def test_non_html_output_is_not_repaired(tmp_path) -> None:
    payload = {
        "output": "A | B\n--- | ---\n1 | 2",
        "image_size": [100, 50],
        "prompt_tokens": 20,
        "generated_tokens": 20,
        "hit_token_ceiling": False,
    }
    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))
    native = result.native_json[0]["internvl3_5_8b"]

    assert result.status == "empty"
    assert "did not contain a usable HTML table" in result.error
    assert result.native_markdown == []
    assert native["output"] == payload["output"]
    assert native["normalization"] == "none"
    assert native["semantic_repair"] is False


def test_token_ceiling_is_not_accepted_as_completed_output(tmp_path) -> None:
    payload = {
        "output": SIMPLE_HTML,
        "image_size": [100, 50],
        "prompt_tokens": 20,
        "generated_tokens": MAX_NEW_TOKENS,
        "hit_token_ceiling": True,
    }
    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))
    native = result.native_json[0]["internvl3_5_8b"]

    assert result.status == "empty"
    assert "generation ceiling" in result.error
    assert result.native_markdown == []
    assert native["hit_token_ceiling"] is True
    assert native["html_tables_detected"] == 1


def test_multiple_native_tables_are_preserved_without_selection(tmp_path) -> None:
    second_html = "<table><tr><td>C</td></tr></table>"
    output = SIMPLE_HTML + second_html
    payload = {
        "output": output,
        "image_size": [100, 100],
        "prompt_tokens": 20,
        "generated_tokens": 50,
        "hit_token_ceiling": False,
    }
    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))
    parsed = parse_result_tables(result)

    assert result.status == "ok"
    assert result.result_count == 2
    assert result.native_markdown == [output]
    assert len(parsed) == 2
    assert parsed[0].rows == [["A", "B"], ["1", "2"]]
    assert parsed[1].rows == [["C"]]


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
