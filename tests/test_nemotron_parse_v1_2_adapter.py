from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tabulus.table_ocr.base import TableOCRInput
from tabulus.table_ocr.nemotron_parse_v1_2 import (
    CRADIO_REVISION,
    MODEL_REVISION,
    MODEL_VERSION,
    NemotronParseV12Adapter,
)
from tabulus.table_ocr.output import parse_result_tables
from tabulus.table_ocr.registry import list_table_ocr_adapters


SIMPLE_HTML = """
<table>
  <tr><td>A</td><td>B</td></tr>
  <tr><td>1</td><td>2</td></tr>
</table>
""".strip()

SECOND_HTML = """
<table>
  <tr><td>C</td><td>D</td></tr>
  <tr><td>3</td><td>4</td></tr>
</table>
""".strip()

CLEAN_TABLE_OUTPUT = (
    "<x_0.1><y_0.1>\\begin{tabular}{cc}"
    "A & B \\\\ 1 & 2 \\\\ "
    "\\end{tabular}<x_0.9><y_0.9><class_Table>"
)


class FakeRuntimeLoader:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.runtime = SimpleNamespace(
            model_device=None,
            model_dtype="torch.bfloat16",
            model_class="NemotronParseForConditionalGeneration",
            encoder_class="RADIOModel",
            encoder_module=(
                "transformers_modules.nvidia.C_RADIOv2_H."
                f"{CRADIO_REVISION}.hf_model"
            ),
            encoder_source=(
                "/cache/C-RADIOv2-H/"
                f"{CRADIO_REVISION}/hf_model.py"
            ),
            cradio_revision_verified=True,
            attention_implementation="sdpa",
            image_size=[2048, 1664],
            transformers_version="5.6.1",
            accelerate_version="1.12.0",
            albumentations_version="2.0.8",
            timm_version="1.0.22",
            einops_version="0.8.2",
            open_clip_torch_version="3.3.0",
            opencv_version="5.0.0.93",
            beautifulsoup_version="4.15.0",
            torch_version="2.6.0+cu124",
            torchvision_version="0.21.0+cu124",
            pillow_version="11.3.0",
            huggingface_hub_version="0.36.0",
            safetensors_version="0.6.2",
            helper_dir=f"/cache/Nemotron/{MODEL_REVISION}",
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
            "raw_output": CLEAN_TABLE_OUTPUT + "</s>",
            "clean_output": CLEAN_TABLE_OUTPUT,
            "source_image_size": [200, 100],
            "prompt_tokens": 6,
            "generated_tokens": 42,
            "raw_output_chars": len(CLEAN_TABLE_OUTPUT) + 4,
            "clean_output_chars": len(CLEAN_TABLE_OUTPUT),
            "objects": [
                {
                    "class": "Table",
                    "bbox": [0.1, 0.1, 0.9, 0.9],
                    "text": "\\begin{tabular}{cc}...",
                }
            ],
            "html_tables": [SIMPLE_HTML],
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

    adapter = NemotronParseV12Adapter(
        device=device,
        runtime_loader=loader,
        inference_runner=runner,
    )

    return adapter, loader, runner


def test_registry_reports_nemotron_parse_v1_2() -> None:
    specs = {
        spec.name: spec
        for spec in list_table_ocr_adapters()
    }

    spec = specs["nemotron-parse-v1-2"]

    assert spec.display_name == "NVIDIA Nemotron Parse v1.2"
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
        NemotronParseV12Adapter(device="cpu")


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="gpu:<index>"):
        NemotronParseV12Adapter(device="cuda:0")


def test_result_preserves_versions_and_native_evidence(
    tmp_path,
) -> None:
    adapter, *_ = make_adapter()

    result = adapter.extract(
        make_input(tmp_path, 12)
    )

    assert result.status == "ok"
    assert result.table_id == 12
    assert result.adapter_name == "nemotron-parse-v1-2"
    assert result.adapter_version == "5.6.1"
    assert result.model_version == MODEL_VERSION
    assert result.result_count == 1

    native = result.native_json[0]["nemotron_parse_v1_2"]

    assert native["model_repo"] == (
        "nvidia/NVIDIA-Nemotron-Parse-v1.2"
    )
    assert native["model_revision"] == MODEL_REVISION
    assert native["model_class"] == (
        "NemotronParseForConditionalGeneration"
    )
    assert native["model_dtype"] == "bfloat16"
    assert native["resolved_model_dtype"] == "torch.bfloat16"
    assert native["image_size"] == [2048, 1664]
    assert native["attention_implementation"] == "sdpa"
    assert native["cradio_repo"] == "nvidia/C-RADIOv2-H"
    assert native["cradio_revision"] == CRADIO_REVISION
    assert native["cradio_revision_verified"] is True
    assert native["encoder_class"] == "RADIOModel"
    assert CRADIO_REVISION in native["encoder_module"]
    assert native["max_new_tokens"] == 9000
    assert native["generation_do_sample"] is False
    assert native["generation_num_beams"] == 1
    assert native["generation_repetition_penalty"] == 1.1
    assert native["execution_device"] == "cuda:0"
    assert native["transformers_version"] == "5.6.1"
    assert native["accelerate_version"] == "1.12.0"
    assert native["albumentations_version"] == "2.0.8"
    assert native["timm_version"] == "1.0.22"
    assert native["einops_version"] == "0.8.2"
    assert native["open_clip_torch_version"] == "3.3.0"
    assert native["opencv_version"] == "5.0.0.93"
    assert native["beautifulsoup_version"] == "4.15.0"
    assert native["source_image_size"] == [200, 100]
    assert native["prompt_tokens"] == 6
    assert native["generated_tokens"] == 42
    assert native["objects"][0]["class"] == "Table"
    assert native["table_objects"] == 1
    assert native["html_tables"] == [SIMPLE_HTML]
    assert native["normalization"].startswith(
        "pinned NVIDIA postprocessing.py"
    )
    assert native["structured_tables_detected"] == 1
    assert native["parser_errors"] == []
    assert native["input_policy"] == "canonical_mineru_crop"
    assert native["generated_bbox_usage"] == "provenance_only"
    assert native["layout_redetection"] is False
    assert native["recropping"] is False
    assert native["external_recropping"] is False
    assert native["semantic_repair"] is False
    assert native["continued_table_merging"] is False


def test_shared_parser_receives_html_table(
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


def test_multiple_table_objects_are_preserved_without_choice(
    tmp_path,
) -> None:
    payload = {
        "raw_output": "two tables",
        "clean_output": "two tables",
        "source_image_size": [200, 100],
        "prompt_tokens": 6,
        "generated_tokens": 50,
        "raw_output_chars": 10,
        "clean_output_chars": 10,
        "objects": [
            {
                "class": "Table",
                "bbox": [0.1, 0.1, 0.4, 0.4],
                "text": "table one",
            },
            {
                "class": "Table",
                "bbox": [0.5, 0.5, 0.9, 0.9],
                "text": "table two",
            },
        ],
        "html_tables": [SIMPLE_HTML, SECOND_HTML],
    }

    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))
    parsed = parse_result_tables(result)

    assert result.status == "ok"
    assert result.result_count == 2
    assert len(parsed) == 2
    assert parsed[0].rows[0] == ["A", "B"]
    assert parsed[1].rows[0] == ["C", "D"]


def test_non_table_object_is_explicitly_empty(
    tmp_path,
) -> None:
    payload = {
        "raw_output": "text",
        "clean_output": "text",
        "source_image_size": [100, 50],
        "prompt_tokens": 6,
        "generated_tokens": 5,
        "raw_output_chars": 4,
        "clean_output_chars": 4,
        "objects": [
            {
                "class": "Text",
                "bbox": [0.1, 0.1, 0.9, 0.9],
                "text": "text",
            }
        ],
        "html_tables": [],
    }

    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))

    assert result.status == "empty"
    assert "no Table-class object" in result.error


def test_table_object_without_usable_html_is_empty(
    tmp_path,
) -> None:
    payload = {
        "raw_output": "table",
        "clean_output": "table",
        "source_image_size": [100, 50],
        "prompt_tokens": 6,
        "generated_tokens": 5,
        "raw_output_chars": 5,
        "clean_output_chars": 5,
        "objects": [
            {
                "class": "Table",
                "bbox": [0.1, 0.1, 0.9, 0.9],
                "text": "table",
            }
        ],
        "html_tables": ["not a structured table"],
    }

    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))

    assert result.status == "empty"
    assert "did not yield a usable structured table" in result.error


def test_empty_generated_output_is_explicit(
    tmp_path,
) -> None:
    payload = {
        "raw_output": "",
        "clean_output": "",
        "source_image_size": [100, 50],
        "prompt_tokens": 6,
        "generated_tokens": 0,
        "raw_output_chars": 0,
        "clean_output_chars": 0,
        "objects": [],
        "html_tables": [],
    }

    adapter, *_ = make_adapter(payload=payload)
    result = adapter.extract(make_input(tmp_path))

    assert result.status == "empty"
    assert "no generated output" in result.error


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
