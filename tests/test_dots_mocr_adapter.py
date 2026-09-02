from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from tabulus.table_ocr.base import TableOCRInput
from tabulus.table_ocr.dots_mocr import (
    DotsMOCRAdapter,
    LAYOUT_PROMPT,
    MAX_NEW_TOKENS,
    MODEL_VERSION,
    _extract_layout_objects,
)
from tabulus.table_ocr.output import parse_result_tables
from tabulus.table_ocr.registry import list_table_ocr_adapters


SIMPLE_HTML = (
    "<table>"
    "<tr><td>A</td><td>B</td></tr>"
    "<tr><td>1</td><td>2</td></tr>"
    "</table>"
)


def layout_output(*objects: dict[str, Any]) -> str:
    return json.dumps(list(objects), ensure_ascii=False)


class FakeRuntimeLoader:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.runtime = SimpleNamespace(
            model_device=None,
            model_dtype="torch.bfloat16",
            model_class="DotsOCRForCausalLM",
            model_type="dots_ocr",
            config_class="DotsOCRConfig",
            processor_class="DotsVLProcessor",
            image_processor_class="Qwen2VLImageProcessorFast",
            tokenizer_class="Qwen2TokenizerFast",
            attention_implementation="flash_attention_2",
            generation_do_sample=False,
            generation_num_beams=1,
            generation_temperature=1.0,
            generation_top_p=1.0,
            transformers_version="4.57.6",
            accelerate_version="1.14.0",
            torch_version="2.7.0+cu128",
            torchvision_version="0.22.0+cu128",
            qwen_vl_utils_version="0.0.14",
            flash_attn_version="2.8.0.post2",
        )

    def __call__(self, device: str) -> Any:
        self.calls.append(device)
        self.runtime.model_device = device
        return self.runtime


class FakeInferenceRunner:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        clean = layout_output(
            {
                "bbox": [0, 0, 100, 100],
                "category": "Table",
                "text": SIMPLE_HTML,
            }
        )
        self.payload = payload or {
            "raw_output": clean + "<|endofassistant|>",
            "clean_output": clean,
            "image_size": [200, 100],
            "prompt_tokens": 25,
            "generated_tokens": 30,
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
    adapter = DotsMOCRAdapter(
        device=device,
        runtime_loader=loader,
        inference_runner=runner,
    )
    return adapter, loader, runner


def test_registry_reports_dots_mocr() -> None:
    specs = {spec.name: spec for spec in list_table_ocr_adapters()}
    spec = specs["dots-mocr"]
    assert spec.display_name == "dots.mocr"
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
        DotsMOCRAdapter(device="cpu")


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="gpu:<index>"):
        DotsMOCRAdapter(device="cuda:0")


def test_result_preserves_versions_layout_and_bbox_provenance(tmp_path) -> None:
    adapter, *_ = make_adapter()
    result = adapter.extract(make_input(tmp_path, 12))

    assert result.status == "ok"
    assert result.table_id == 12
    assert result.adapter_name == "dots-mocr"
    assert result.adapter_version == "4.57.6"
    assert result.model_version == MODEL_VERSION
    assert result.result_count == 1

    native = result.native_json[0]["dots_mocr"]
    assert native["model_repo"] == "dots-studio/dots.mocr"
    assert native["model_revision"] == (
        "e539fbb52280393adc081b289ec597430a0f9031"
    )
    assert native["config_class"] == "DotsOCRConfig"
    assert native["model_class"] == "DotsOCRForCausalLM"
    assert native["model_type"] == "dots_ocr"
    assert native["processor_class"] == "DotsVLProcessor"
    assert native["image_processor_class"] == "Qwen2VLImageProcessorFast"
    assert native["tokenizer_class"] == "Qwen2TokenizerFast"
    assert native["prompt_mode"] == "prompt_layout_all_en"
    assert native["prompt"] == LAYOUT_PROMPT
    assert native["model_load_dtype"] == "bfloat16"
    assert native["resolved_model_dtype"] == "torch.bfloat16"
    assert native["attention_implementation"] == "flash_attention_2"
    assert native["max_new_tokens"] == MAX_NEW_TOKENS
    assert native["do_sample"] is False
    assert native["num_beams"] == 1
    assert native["execution_device"] == "cuda:0"
    assert native["transformers_version"] == "4.57.6"
    assert native["accelerate_version"] == "1.14.0"
    assert native["torch_version"] == "2.7.0+cu128"
    assert native["torchvision_version"] == "0.22.0+cu128"
    assert native["qwen_vl_utils_version"] == "0.0.14"
    assert native["flash_attn_version"] == "2.8.0.post2"
    assert native["image_size"] == [200, 100]
    assert native["prompt_tokens"] == 25
    assert native["generated_tokens"] == 30
    assert native["raw_output"].endswith("<|endofassistant|>")
    assert native["layout_json_parse_error"] is None
    assert native["layout_objects_detected"] == 1
    assert native["table_objects_detected"] == 1
    assert native["html_tables_detected"] == 1
    assert native["table_bboxes"] == [[0, 0, 100, 100]]
    assert native["bbox_policy"] == "provenance_only"
    assert native["model_native_layout_detection"] is True
    assert native["external_layout_redetection"] is False
    assert native["external_table_redetection"] is False
    assert native["recropping"] is False
    assert native["table_bboxes_used_for_recropping"] is False
    assert native["normalization"] == "none"
    assert native["json_repair"] is False
    assert native["semantic_repair"] is False
    assert native["continued_table_merging"] is False
    assert native["input_policy"] == "canonical_mineru_crop"


def test_shared_parser_receives_only_model_emitted_table_html(tmp_path) -> None:
    clean = layout_output(
        {
            "bbox": [0, 0, 20, 20],
            "category": "Text",
            "text": "not a table",
        },
        {
            "bbox": [20, 20, 100, 100],
            "category": "Table",
            "text": SIMPLE_HTML,
        },
    )
    payload = {
        "raw_output": clean + "<|endofassistant|>",
        "clean_output": clean,
        "image_size": [100, 100],
        "prompt_tokens": 20,
        "generated_tokens": 30,
    }
    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))
    parsed = parse_result_tables(result)

    assert result.status == "ok"
    assert result.native_markdown == [SIMPLE_HTML]
    assert len(parsed) == 1
    assert parsed[0].source == "html"
    assert parsed[0].rows == [["A", "B"], ["1", "2"]]


def test_json_list_and_wrapped_dict_layouts_are_supported() -> None:
    item = {"category": "Table", "bbox": [1, 2, 3, 4], "text": SIMPLE_HTML}
    assert _extract_layout_objects([item]) == [item]
    assert _extract_layout_objects({"elements": [item]}) == [item]


def test_empty_generated_output_is_explicit(tmp_path) -> None:
    payload = {
        "raw_output": "<|endofassistant|>",
        "clean_output": "",
        "image_size": [100, 50],
        "prompt_tokens": 20,
        "generated_tokens": 1,
    }
    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))
    assert result.status == "empty"
    assert "no generated layout output" in result.error


def test_malformed_json_is_not_repaired(tmp_path) -> None:
    payload = {
        "raw_output": "[not-json]<|endofassistant|>",
        "clean_output": "[not-json]",
        "image_size": [100, 50],
        "prompt_tokens": 20,
        "generated_tokens": 4,
    }
    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))
    native = result.native_json[0]["dots_mocr"]

    assert result.status == "empty"
    assert "not valid JSON layout output" in result.error
    assert native["json_repair"] is False
    assert native["layout_json"] is None
    assert native["layout_json_parse_error"]


def test_non_table_layout_objects_are_not_used_as_tables(tmp_path) -> None:
    clean = layout_output(
        {
            "bbox": [0, 0, 100, 100],
            "category": "Text",
            "text": SIMPLE_HTML,
        }
    )
    payload = {
        "raw_output": clean + "<|endofassistant|>",
        "clean_output": clean,
        "image_size": [100, 50],
        "prompt_tokens": 20,
        "generated_tokens": 20,
    }
    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))

    assert result.status == "empty"
    assert "no model-emitted Table objects" in result.error
    assert result.native_markdown == []


def test_table_object_without_html_is_explicit(tmp_path) -> None:
    clean = layout_output(
        {
            "bbox": [0, 0, 100, 100],
            "category": "Table",
            "text": "A | B\n--- | ---\n1 | 2",
        }
    )
    payload = {
        "raw_output": clean + "<|endofassistant|>",
        "clean_output": clean,
        "image_size": [100, 50],
        "prompt_tokens": 20,
        "generated_tokens": 20,
    }
    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))

    assert result.status == "empty"
    assert "did not contain usable HTML tables" in result.error
    assert result.native_markdown == []


def test_multiple_table_objects_are_preserved_independently(tmp_path) -> None:
    second_html = "<table><tr><td>C</td></tr></table>"
    clean = layout_output(
        {
            "bbox": [0, 0, 40, 40],
            "category": "Table",
            "text": SIMPLE_HTML,
        },
        {
            "bbox": [50, 50, 90, 90],
            "category": "Table",
            "text": second_html,
        },
    )
    payload = {
        "raw_output": clean + "<|endofassistant|>",
        "clean_output": clean,
        "image_size": [100, 100],
        "prompt_tokens": 20,
        "generated_tokens": 50,
    }
    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))
    parsed = parse_result_tables(result)

    assert result.status == "ok"
    assert result.result_count == 2
    assert result.native_markdown == [SIMPLE_HTML, second_html]
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
