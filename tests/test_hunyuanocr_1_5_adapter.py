from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tabulus.table_ocr.base import TableOCRInput
from tabulus.table_ocr.hunyuanocr_1_5 import (
    CLEANUP_MIN_REPEATS,
    HunyuanOCR15Adapter,
    MODEL_VERSION,
    _clean_repeated_substrings,
    _has_tail_repetition,
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
            model_class="HunYuanVLForConditionalGeneration",
            model_type="hunyuan_vl",
            transformers_version="5.13.0",
            accelerate_version="1.14.0",
            torch_version="2.11.0+cu130",
            torchvision_version="0.26.0+cu130",
        )

    def __call__(self, device: str) -> Any:
        self.calls.append(device)
        self.runtime.model_device = device
        return self.runtime


class FakeInferenceRunner:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {
            "raw_output": SIMPLE_HTML + "<|eos|>",
            "decoded_output": SIMPLE_HTML,
            "clean_output": SIMPLE_HTML,
            "image_size": [200, 100],
            "prompt_tokens": 25,
            "generated_tokens": 30,
            "tail_repetition_stop_triggered": False,
            "repetition_cleanup_changed_output": False,
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
    adapter = HunyuanOCR15Adapter(
        device=device,
        runtime_loader=loader,
        inference_runner=runner,
    )
    return adapter, loader, runner


def test_registry_reports_hunyuanocr_1_5() -> None:
    specs = {spec.name: spec for spec in list_table_ocr_adapters()}
    spec = specs["hunyuanocr-1-5"]
    assert spec.display_name == "HunyuanOCR-1.5"
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
        HunyuanOCR15Adapter(device="cpu")


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="gpu:<index>"):
        HunyuanOCR15Adapter(device="cuda:0")


def test_result_preserves_versions_and_native_evidence(tmp_path) -> None:
    adapter, *_ = make_adapter()
    result = adapter.extract(make_input(tmp_path, 12))

    assert result.status == "ok"
    assert result.table_id == 12
    assert result.adapter_name == "hunyuanocr-1-5"
    assert result.adapter_version == "5.13.0"
    assert result.model_version == MODEL_VERSION
    assert result.result_count == 1

    native = result.native_json[0]["hunyuanocr_1_5"]
    assert native["model_repo"] == "tencent/HunyuanOCR"
    assert native["model_revision"] == (
        "47644ecc4fc854efa4f505155158831f36773ee4"
    )
    assert native["model_class"] == "HunYuanVLForConditionalGeneration"
    assert native["model_type"] == "hunyuan_vl"
    assert native["task"] == "table"
    assert native["prompt"] == "把图中的表格解析为HTML。"
    assert native["model_load_dtype"] == "bfloat16"
    assert native["resolved_model_dtype"] == "torch.bfloat16"
    assert native["attention_implementation"] == "eager"
    assert native["max_new_tokens"] == 8192
    assert native["do_sample"] is False
    assert native["repetition_penalty"] == 1.08
    assert native["use_cache"] is True
    assert native["execution_device"] == "cuda:0"
    assert native["transformers_version"] == "5.13.0"
    assert native["accelerate_version"] == "1.14.0"
    assert native["torch_version"] == "2.11.0+cu130"
    assert native["torchvision_version"] == "0.26.0+cu130"
    assert native["image_size"] == [200, 100]
    assert native["prompt_tokens"] == 25
    assert native["generated_tokens"] == 30
    assert native["raw_output"].endswith("<|eos|>")
    assert native["decoded_output"] == SIMPLE_HTML
    assert native["clean_output"] == SIMPLE_HTML
    assert native["native_format"] == "html"
    assert native["special_tokens_removed_for_parsing"] is True
    assert native["normalization"] == "none"
    assert native["document_markdown_postprocessing"] is False
    assert native["html_tables_detected"] == 1
    assert native["input_policy"] == "canonical_mineru_crop"
    assert native["layout_redetection"] is False
    assert native["table_redetection"] is False
    assert native["recropping"] is False
    assert native["semantic_repair"] is False
    assert native["continued_table_merging"] is False

    repetition = native["official_repetition_controls"]
    assert repetition["tail_min_repeats"] == 8
    assert repetition["tail_max_unit"] == 256
    assert repetition["tail_check_start_chars"] == 4000
    assert repetition["tail_check_step_chars"] == 1000
    assert repetition["tail_token_probe_step"] == 64
    assert repetition["tail_window_chars"] == 8000
    assert repetition["cleanup_min_repeats"] == CLEANUP_MIN_REPEATS
    assert repetition["tail_stop_triggered"] is False
    assert repetition["cleanup_changed_output"] is False


def test_shared_parser_receives_clean_html(tmp_path) -> None:
    adapter, *_ = make_adapter()
    result = adapter.extract(make_input(tmp_path))
    parsed = parse_result_tables(result)
    assert len(parsed) == 1
    assert parsed[0].source == "html"
    assert parsed[0].rows == [["A", "B"], ["1", "2"]]


def test_tail_repetition_detection() -> None:
    assert _has_tail_repetition("prefix" + "abc" * 8)
    assert not _has_tail_repetition("abcdefg")


def test_official_cleanup_trims_repeated_suffix() -> None:
    prefix = "x" * 2100
    repeated = "YZ" * CLEANUP_MIN_REPEATS
    cleaned = _clean_repeated_substrings(prefix + repeated)
    assert cleaned == prefix + "YZ"


def test_empty_generated_output_is_explicit(tmp_path) -> None:
    payload = {
        "raw_output": "<|eos|>",
        "decoded_output": "",
        "clean_output": "",
        "image_size": [100, 50],
        "prompt_tokens": 20,
        "generated_tokens": 1,
        "tail_repetition_stop_triggered": False,
        "repetition_cleanup_changed_output": False,
    }
    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))
    assert result.status == "empty"
    assert "no generated table output" in result.error


def test_non_html_output_is_explicit(tmp_path) -> None:
    payload = {
        "raw_output": "not a table<|eos|>",
        "decoded_output": "not a table",
        "clean_output": "not a table",
        "image_size": [100, 50],
        "prompt_tokens": 20,
        "generated_tokens": 4,
        "tail_repetition_stop_triggered": False,
        "repetition_cleanup_changed_output": False,
    }
    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))
    assert result.status == "empty"
    assert "usable HTML table" in result.error


def test_multiple_html_tables_are_not_collapsed(tmp_path) -> None:
    html = (
        "<table><tr><td>A</td></tr></table>"
        "<table><tr><td>B</td></tr></table>"
    )
    payload = {
        "raw_output": html + "<|eos|>",
        "decoded_output": html,
        "clean_output": html,
        "image_size": [100, 50],
        "prompt_tokens": 20,
        "generated_tokens": 20,
        "tail_repetition_stop_triggered": False,
        "repetition_cleanup_changed_output": False,
    }
    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))
    parsed = parse_result_tables(result)
    assert result.status == "ok"
    assert result.result_count == 2
    assert len(parsed) == 2
    assert parsed[0].rows == [["A"]]
    assert parsed[1].rows == [["B"]]


def test_cleanup_intervention_is_preserved(tmp_path) -> None:
    payload = {
        "raw_output": SIMPLE_HTML + "junk<|eos|>",
        "decoded_output": SIMPLE_HTML + "junk",
        "clean_output": SIMPLE_HTML,
        "image_size": [100, 50],
        "prompt_tokens": 20,
        "generated_tokens": 40,
        "tail_repetition_stop_triggered": True,
        "repetition_cleanup_changed_output": True,
    }
    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))
    parsed = parse_result_tables(result)
    native = result.native_json[0]["hunyuanocr_1_5"]
    repetition = native["official_repetition_controls"]

    assert result.status == "ok"
    assert result.native_markdown == [SIMPLE_HTML]
    assert repetition["tail_stop_triggered"] is True
    assert repetition["cleanup_changed_output"] is True
    assert len(parsed) == 1


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
