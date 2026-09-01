from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tabulus.table_ocr.base import TableOCRInput
from tabulus.table_ocr.monkeyocrv2_b_parsing import (
    MODEL_VERSION,
    TABLE_MIN_PIXELS,
    MonkeyOCRv2BParsingAdapter,
)
from tabulus.table_ocr.output import parse_result_tables
from tabulus.table_ocr.registry import list_table_ocr_adapters


SIMPLE_OTSL = (
    "<fcel>A<fcel>B<nl>"
    "<fcel>1<fcel>2<nl>"
)

MERGED_OTSL = (
    "<fcel>Material<fcel>Reference<nl>"
    "<fcel>Al2O3<fcel>41<nl>"
    "<ucel><fcel>59<nl>"
)


class FakeRuntimeLoader:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.runtime = SimpleNamespace(
            model_device=None,
            model_dtype="torch.bfloat16",
            model_class="MonkeyOCRv2ForCausalLM",
            checkpoint_vision_attention="flash_attention_2",
            resolved_vision_attention="sdpa",
            processor_min_pixels=1003520,
            processor_max_pixels=11289600,
            transformers_version="4.57.1",
            accelerate_version="1.11.0",
            timm_version="1.0.27",
            einops_version="0.8.1",
            torch_version="2.6.0+cu124",
            torchvision_version="0.21.0+cu124",
            pillow_version="11.3.0",
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
            "raw_output": SIMPLE_OTSL + "<|im_end|>",
            "clean_output": SIMPLE_OTSL,
            "source_image_size": [200, 100],
            "prompt_tokens": 25,
            "generated_tokens": 12,
            "raw_output_chars": len(SIMPLE_OTSL) + 10,
            "clean_output_chars": len(SIMPLE_OTSL),
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

    adapter = MonkeyOCRv2BParsingAdapter(
        device=device,
        runtime_loader=loader,
        inference_runner=runner,
    )

    return adapter, loader, runner


def test_registry_reports_monkeyocrv2_b_parsing() -> None:
    specs = {
        spec.name: spec
        for spec in list_table_ocr_adapters()
    }

    spec = specs["monkeyocrv2-b-parsing"]

    assert spec.display_name == "MonkeyOCRv2-B-Parsing"
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
    with pytest.raises(ValueError, match="gpu:<index>"):
        MonkeyOCRv2BParsingAdapter(device="cpu")


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="gpu:<index>"):
        MonkeyOCRv2BParsingAdapter(device="cuda:0")


def test_result_preserves_versions_and_native_evidence(
    tmp_path,
) -> None:
    adapter, *_ = make_adapter()

    result = adapter.extract(
        make_input(tmp_path, 12)
    )

    assert result.status == "ok"
    assert result.table_id == 12
    assert result.adapter_name == "monkeyocrv2-b-parsing"
    assert result.adapter_version == "4.57.1"
    assert result.model_version == MODEL_VERSION
    assert result.result_count == 1

    native = result.native_json[0]["monkeyocrv2_b_parsing"]

    assert native["model_repo"] == "zenosai/MonkeyOCRv2-B-Parsing"
    assert native["model_revision"] == (
        "2419139b7bcd3fda2689b2a83167172afba91c8b"
    )
    assert native["model_class"] == "MonkeyOCRv2ForCausalLM"
    assert native["native_format"] == "otsl"
    assert native["model_dtype"] == "bfloat16"
    assert native["resolved_model_dtype"] == "torch.bfloat16"
    assert native["checkpoint_vision_attention"] == "flash_attention_2"
    assert native["attention_implementation"] == "sdpa"
    assert native["processor_use_fast"] is False
    assert native["table_min_pixels"] == TABLE_MIN_PIXELS
    assert native["processor_min_pixels"] == 1003520
    assert native["processor_max_pixels"] == 11289600
    assert native["max_new_tokens"] == 4096
    assert native["generation_do_sample"] is False
    assert native["generation_temperature"] is None
    assert native["generation_top_p"] is None
    assert native["generation_top_k"] is None
    assert native["execution_device"] == "cuda:0"
    assert native["transformers_version"] == "4.57.1"
    assert native["accelerate_version"] == "1.11.0"
    assert native["timm_version"] == "1.0.27"
    assert native["einops_version"] == "0.8.1"
    assert native["torch_version"] == "2.6.0+cu124"
    assert native["torchvision_version"] == "0.21.0+cu124"
    assert native["source_image_size"] == [200, 100]
    assert native["prompt_tokens"] == 25
    assert native["generated_tokens"] == 12
    assert native["raw_output"].endswith("<|im_end|>")
    assert native["clean_output"] == SIMPLE_OTSL
    assert native["special_tokens_removed_for_parsing"] is True
    assert native["normalization"].endswith("otsl_table_to_html")
    assert native["structured_tables_detected"] == 1
    assert native["parser_error"] is None
    assert native["input_policy"] == "canonical_mineru_crop"
    assert native["image_preprocessing"] == {
        "external": "rgb_conversion_only",
        "processor": "AutoProcessor",
        "processor_use_fast": False,
        "upstream_table_min_pixels": 1003520,
        "model_internal_resize": True,
    }
    assert native["layout_redetection"] is False
    assert native["recropping"] is False
    assert native["external_recropping"] is False
    assert native["semantic_repair"] is False


def test_shared_parser_receives_one_rectangular_table(
    tmp_path,
) -> None:
    adapter, *_ = make_adapter()

    result = adapter.extract(
        make_input(tmp_path)
    )
    parsed = parse_result_tables(result)

    assert len(parsed) == 1
    assert parsed[0].rows == [
        ["A", "B"],
        ["1", "2"],
    ]


def test_otsl_rowspan_is_preserved_without_semantic_repair(
    tmp_path,
) -> None:
    payload = {
        "raw_output": MERGED_OTSL + "<|im_end|>",
        "clean_output": MERGED_OTSL,
        "source_image_size": [200, 100],
        "prompt_tokens": 25,
        "generated_tokens": 15,
        "raw_output_chars": len(MERGED_OTSL) + 10,
        "clean_output_chars": len(MERGED_OTSL),
    }

    adapter, *_ = make_adapter(payload=payload)

    result = adapter.extract(
        make_input(tmp_path)
    )
    parsed = parse_result_tables(result)

    assert len(parsed) == 1
    assert parsed[0].rows == [
        ["Material", "Reference"],
        ["Al2O3", "41"],
        ["", "59"],
    ]


def test_empty_generated_output_is_explicit(
    tmp_path,
) -> None:
    payload = {
        "raw_output": "<|im_end|>",
        "clean_output": "",
        "source_image_size": [100, 50],
        "prompt_tokens": 20,
        "generated_tokens": 1,
        "raw_output_chars": 10,
        "clean_output_chars": 0,
    }

    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))

    assert result.status == "empty"
    assert "no generated OTSL output" in result.error


def test_unusable_otsl_is_explicit(
    tmp_path,
) -> None:
    payload = {
        "raw_output": "not an OTSL table<|im_end|>",
        "clean_output": "not an OTSL table",
        "source_image_size": [100, 50],
        "prompt_tokens": 20,
        "generated_tokens": 5,
        "raw_output_chars": 28,
        "clean_output_chars": 17,
    }

    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))

    assert result.status == "empty"
    assert "one usable OTSL table" in result.error


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
