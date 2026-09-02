from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

from tabulus.table_ocr.base import (
    TableOCRCapabilities,
    TableOCRDependencyError,
    TableOCRInput,
    TableOCRResult,
)
from tabulus.table_ocr.parsing import extract_html_tables


VALIDATED_PYTHON_MAJOR_MINOR = (3, 12)
VALIDATED_TRANSFORMERS_VERSION = "4.57.6"
VALIDATED_ACCELERATE_VERSION = "1.14.0"
VALIDATED_TORCH_VERSION = "2.7.0+cu128"
VALIDATED_TORCHVISION_VERSION = "0.22.0+cu128"
VALIDATED_QWEN_VL_UTILS_VERSION = "0.0.14"
VALIDATED_FLASH_ATTN_VERSION = "2.8.0.post2"

MODEL_REPO = "dots-studio/dots.mocr"
MODEL_REVISION = "e539fbb52280393adc081b289ec597430a0f9031"
MODEL_VERSION = f"{MODEL_REPO}@{MODEL_REVISION}"
CONFIG_CLASS = "DotsOCRConfig"
MODEL_CLASS = "DotsOCRForCausalLM"
MODEL_TYPE = "dots_ocr"
PROCESSOR_CLASS = "DotsVLProcessor"
IMAGE_PROCESSOR_CLASS = "Qwen2VLImageProcessorFast"
TOKENIZER_CLASS = "Qwen2TokenizerFast"

PROMPT_MODE = "prompt_layout_all_en"
LAYOUT_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""

MODEL_DTYPE = "bfloat16"
ATTENTION_IMPLEMENTATION = "flash_attention_2"
MAX_NEW_TOKENS = 24000
GENERATION_DO_SAMPLE = False
GENERATION_NUM_BEAMS = 1


@dataclass(frozen=True)
class _DotsMOCRRuntime:
    torch: Any
    image_module: Any
    processor: Any
    model: Any
    process_vision_info: Callable[[Any], tuple[Any, Any]]
    model_device: str
    model_dtype: str
    model_class: str
    model_type: str
    config_class: str
    processor_class: str
    image_processor_class: str
    tokenizer_class: str
    attention_implementation: str
    generation_do_sample: bool
    generation_num_beams: int
    generation_temperature: float | None
    generation_top_p: float | None
    transformers_version: str
    accelerate_version: str
    torch_version: str
    torchvision_version: str
    qwen_vl_utils_version: str
    flash_attn_version: str


RuntimeLoader = Callable[[str], Any]
InferenceRunner = Callable[[Path, Any], dict[str, Any]]


def _installed_package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _require_validated_version(package_name: str, expected: str) -> str:
    found = _installed_package_version(package_name)
    if found != expected:
        actual = found or "not installed"
        raise TableOCRDependencyError(
            "The dots-mocr adapter is validated against "
            f"{package_name}=={expected}; found {actual}."
        )
    return found


def _dots_mocr_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized == "gpu":
        return "cuda:0"
    if normalized.startswith("gpu:"):
        index = normalized.split(":", maxsplit=1)[1]
        if index.isdigit():
            return f"cuda:{index}"
    raise ValueError("Use device='gpu' or 'gpu:<index>' for dots-mocr.")


def _dependency_error() -> TableOCRDependencyError:
    return TableOCRDependencyError(
        "The dots-mocr adapter requires Python 3.12, "
        "transformers==4.57.6, accelerate==1.14.0, "
        "torch==2.7.0+cu128, torchvision==0.22.0+cu128, "
        "qwen-vl-utils==0.0.14, flash-attn==2.8.0.post2, and Pillow "
        "in the active environment."
    )


def _extract_layout_objects(payload: Any) -> list[dict[str, Any]]:
    """Collect model-emitted layout elements without altering their contents."""

    objects: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if "category" in value:
                objects.append(value)
                return
            for child in value.values():
                if isinstance(child, (dict, list)):
                    visit(child)
            return

        if isinstance(value, list):
            for child in value:
                if isinstance(child, (dict, list)):
                    visit(child)

    visit(payload)
    return objects


def _default_runtime_loader(model_device: str) -> _DotsMOCRRuntime:
    if sys.version_info[:2] != VALIDATED_PYTHON_MAJOR_MINOR:
        found = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise TableOCRDependencyError(
            "The dots-mocr adapter is validated against "
            f"Python 3.12; found {found}."
        )

    transformers_version = _require_validated_version(
        "transformers", VALIDATED_TRANSFORMERS_VERSION
    )
    accelerate_version = _require_validated_version(
        "accelerate", VALIDATED_ACCELERATE_VERSION
    )
    torch_version = _require_validated_version("torch", VALIDATED_TORCH_VERSION)
    torchvision_version = _require_validated_version(
        "torchvision", VALIDATED_TORCHVISION_VERSION
    )
    qwen_vl_utils_version = _require_validated_version(
        "qwen-vl-utils", VALIDATED_QWEN_VL_UTILS_VERSION
    )
    flash_attn_version = _require_validated_version(
        "flash-attn", VALIDATED_FLASH_ATTN_VERSION
    )

    try:
        torch = importlib.import_module("torch")
        image_module = importlib.import_module("PIL.Image")
        transformers = importlib.import_module("transformers")
        qwen_vl_utils = importlib.import_module("qwen_vl_utils")
        importlib.import_module("torchvision")
        importlib.import_module("accelerate")
        importlib.import_module("flash_attn")
    except ImportError as exc:
        raise _dependency_error() from exc

    if not bool(torch.cuda.is_available()):
        raise RuntimeError(
            "dots-mocr requires CUDA in the validated Tabulus configuration, "
            "but CUDA is not available."
        )

    config = transformers.AutoConfig.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )

    resolved_config_class = type(config).__name__
    if resolved_config_class != CONFIG_CLASS:
        raise RuntimeError(
            "dots.mocr loaded an unexpected config class: "
            f"{resolved_config_class!r}."
        )

    resolved_config_type = str(getattr(config, "model_type", ""))
    if resolved_config_type != MODEL_TYPE:
        raise RuntimeError(
            "dots.mocr loaded an unexpected config model type: "
            f"{resolved_config_type!r}."
        )

    architectures = tuple(getattr(config, "architectures", ()) or ())
    if MODEL_CLASS not in architectures:
        raise RuntimeError(
            "dots.mocr config does not declare the expected architecture "
            f"{MODEL_CLASS!r}: {architectures!r}."
        )

    config_module = importlib.import_module(type(config).__module__)
    processor_type = getattr(config_module, PROCESSOR_CLASS, None)
    if processor_type is None:
        raise TableOCRDependencyError(
            "The pinned dots.mocr remote-code module does not expose "
            f"{PROCESSOR_CLASS}."
        )

    processor = processor_type.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        use_fast=True,
    )

    resolved_processor_class = type(processor).__name__
    if resolved_processor_class != PROCESSOR_CLASS:
        raise RuntimeError(
            "dots.mocr loaded an unexpected processor class: "
            f"{resolved_processor_class!r}."
        )

    image_processor = getattr(processor, "image_processor", None)
    resolved_image_processor_class = type(image_processor).__name__
    if resolved_image_processor_class != IMAGE_PROCESSOR_CLASS:
        raise RuntimeError(
            "dots.mocr loaded an unexpected image processor class: "
            f"{resolved_image_processor_class!r}."
        )

    tokenizer = getattr(processor, "tokenizer", None)
    resolved_tokenizer_class = type(tokenizer).__name__
    if resolved_tokenizer_class != TOKENIZER_CLASS:
        raise RuntimeError(
            "dots.mocr loaded an unexpected tokenizer class: "
            f"{resolved_tokenizer_class!r}."
        )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation=ATTENTION_IMPLEMENTATION,
        device_map={"": model_device},
    ).eval()

    resolved_device = str(next(model.parameters()).device)
    if resolved_device != model_device:
        raise RuntimeError(
            "dots.mocr loaded on an unexpected device: "
            f"requested {model_device}, resolved {resolved_device}."
        )

    resolved_dtype = str(next(model.parameters()).dtype)
    if resolved_dtype != "torch.bfloat16":
        raise RuntimeError(
            "dots.mocr loaded with an unexpected dtype: "
            f"{resolved_dtype!r}."
        )

    resolved_model_class = type(model).__name__
    if resolved_model_class != MODEL_CLASS:
        raise RuntimeError(
            "dots.mocr loaded an unexpected model class: "
            f"{resolved_model_class!r}."
        )

    resolved_model_type = str(getattr(model.config, "model_type", ""))
    if resolved_model_type != MODEL_TYPE:
        raise RuntimeError(
            "dots.mocr loaded an unexpected model type: "
            f"{resolved_model_type!r}."
        )

    resolved_attention = str(
        getattr(model.config, "_attn_implementation", "")
    )
    if resolved_attention != ATTENTION_IMPLEMENTATION:
        raise RuntimeError(
            "dots.mocr loaded with an unexpected attention implementation: "
            f"{resolved_attention!r}."
        )

    generation_config = model.generation_config
    resolved_do_sample = bool(getattr(generation_config, "do_sample", False))
    resolved_num_beams = int(getattr(generation_config, "num_beams", 1))
    if resolved_do_sample is not False or resolved_num_beams != 1:
        raise RuntimeError(
            "dots.mocr resolved unexpected generation defaults: "
            f"do_sample={resolved_do_sample!r}, num_beams={resolved_num_beams!r}."
        )

    temperature_value = getattr(generation_config, "temperature", None)
    top_p_value = getattr(generation_config, "top_p", None)

    return _DotsMOCRRuntime(
        torch=torch,
        image_module=image_module,
        processor=processor,
        model=model,
        process_vision_info=qwen_vl_utils.process_vision_info,
        model_device=model_device,
        model_dtype=resolved_dtype,
        model_class=resolved_model_class,
        model_type=resolved_model_type,
        config_class=resolved_config_class,
        processor_class=resolved_processor_class,
        image_processor_class=resolved_image_processor_class,
        tokenizer_class=resolved_tokenizer_class,
        attention_implementation=resolved_attention,
        generation_do_sample=resolved_do_sample,
        generation_num_beams=resolved_num_beams,
        generation_temperature=(
            float(temperature_value) if temperature_value is not None else None
        ),
        generation_top_p=(
            float(top_p_value) if top_p_value is not None else None
        ),
        transformers_version=transformers_version,
        accelerate_version=accelerate_version,
        torch_version=torch_version,
        torchvision_version=torchvision_version,
        qwen_vl_utils_version=qwen_vl_utils_version,
        flash_attn_version=flash_attn_version,
    )


def _default_inference_runner(
    image_path: Path,
    runtime: _DotsMOCRRuntime,
) -> dict[str, Any]:
    with runtime.image_module.open(image_path) as source_image:
        width, height = source_image.size
        image = source_image.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": LAYOUT_PROMPT},
            ],
        }
    ]

    chat_text = runtime.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = runtime.process_vision_info(messages)
    inputs = runtime.processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(runtime.model_device)

    input_ids = inputs["input_ids"]
    prompt_len = int(input_ids.shape[1])

    with runtime.torch.inference_mode():
        generated_ids = runtime.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=GENERATION_DO_SAMPLE,
            num_beams=GENERATION_NUM_BEAMS,
        )

    trimmed = [
        output_ids[len(source_ids):]
        for source_ids, output_ids in zip(input_ids, generated_ids)
    ]

    raw_values = runtime.processor.batch_decode(
        trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    clean_values = runtime.processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    raw_output = raw_values[0] if raw_values else ""
    clean_output = clean_values[0] if clean_values else ""

    return {
        "raw_output": raw_output,
        "clean_output": clean_output,
        "image_size": [int(width), int(height)],
        "prompt_tokens": prompt_len,
        "generated_tokens": int(trimmed[0].numel()) if trimmed else 0,
    }


class DotsMOCRAdapter:
    """dots.mocr reconstruction from canonical MinerU table crops."""

    NAME = "dots-mocr"
    DISPLAY_NAME = "dots.mocr"
    MODEL_VERSION = MODEL_VERSION

    _CAPABILITIES = TableOCRCapabilities(
        name=NAME,
        display_name=DISPLAY_NAME,
        cpu_supported=False,
        gpu_supported=True,
    )

    def __init__(
        self,
        *,
        device: str = "gpu:0",
        runtime_loader: RuntimeLoader | None = None,
        inference_runner: InferenceRunner | None = None,
    ) -> None:
        self.device = device
        self._model_device = _dots_mocr_device(device)
        self._runtime_loader = runtime_loader or _default_runtime_loader
        self._inference_runner = inference_runner or _default_inference_runner
        self._runtime: Any | None = None

    @property
    def capabilities(self) -> TableOCRCapabilities:
        return self._CAPABILITIES

    def _get_runtime(self) -> Any:
        if self._runtime is None:
            self._runtime = self._runtime_loader(self._model_device)
        return self._runtime

    def _result(
        self,
        table: TableOCRInput,
        *,
        status: str,
        adapter_version: str | None = None,
        result_count: int = 0,
        native_json: list[Any] | None = None,
        native_markdown: list[Any] | None = None,
        error: str | None = None,
    ) -> TableOCRResult:
        return TableOCRResult(
            table_id=table.table_id,
            adapter_name=self.NAME,
            adapter_version=adapter_version,
            model_version=self.MODEL_VERSION,
            device=self.device,
            source_image=table.image_path,
            status=status,  # type: ignore[arg-type]
            provenance=dict(table.provenance),
            result_count=result_count,
            native_json=native_json or [],
            native_markdown=native_markdown or [],
            error=error,
        )

    def extract(self, table: TableOCRInput) -> TableOCRResult:
        image_path = Path(table.image_path)
        if not image_path.is_file():
            return self._result(
                table,
                status="error",
                error=f"Table crop image not found: {image_path}",
            )

        try:
            runtime = self._get_runtime()
        except TableOCRDependencyError:
            raise
        except Exception as exc:
            return self._result(
                table,
                status="error",
                error=f"Could not initialize dots.mocr: {exc}",
            )

        adapter_version = getattr(runtime, "transformers_version", None)

        try:
            reconstruction = self._inference_runner(image_path, runtime)
        except TableOCRDependencyError:
            raise
        except Exception as exc:
            return self._result(
                table,
                status="error",
                adapter_version=adapter_version,
                error=f"dots.mocr table reconstruction failed: {exc}",
            )

        def as_text(key: str) -> str:
            value = reconstruction.get(key, "")
            return value if isinstance(value, str) else str(value or "")

        raw_output = as_text("raw_output")
        clean_output = as_text("clean_output")

        layout_payload: Any | None = None
        layout_parse_error: str | None = None
        layout_objects: list[dict[str, Any]] = []
        table_objects: list[dict[str, Any]] = []
        parser_inputs: list[str] = []
        html_table_count = 0

        if clean_output.strip():
            try:
                layout_payload = json.loads(clean_output)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                layout_parse_error = f"{type(exc).__name__}: {exc}"
            else:
                layout_objects = _extract_layout_objects(layout_payload)
                table_objects = [
                    item
                    for item in layout_objects
                    if item.get("category") == "Table"
                ]

                for item in table_objects:
                    text = item.get("text")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    html_tables = extract_html_tables(text)
                    if not html_tables:
                        continue
                    parser_inputs.append(text)
                    html_table_count += len(html_tables)

        native = {
            "dots_mocr": {
                "model_repo": MODEL_REPO,
                "model_revision": MODEL_REVISION,
                "config_class": getattr(runtime, "config_class", None),
                "model_class": getattr(runtime, "model_class", None),
                "model_type": getattr(runtime, "model_type", None),
                "processor_class": getattr(runtime, "processor_class", None),
                "image_processor_class": getattr(
                    runtime, "image_processor_class", None
                ),
                "tokenizer_class": getattr(runtime, "tokenizer_class", None),
                "prompt_mode": PROMPT_MODE,
                "prompt": LAYOUT_PROMPT,
                "model_load_dtype": MODEL_DTYPE,
                "resolved_model_dtype": getattr(runtime, "model_dtype", None),
                "attention_implementation": getattr(
                    runtime,
                    "attention_implementation",
                    ATTENTION_IMPLEMENTATION,
                ),
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": GENERATION_DO_SAMPLE,
                "num_beams": GENERATION_NUM_BEAMS,
                "resolved_generation_defaults": {
                    "do_sample": getattr(
                        runtime, "generation_do_sample", None
                    ),
                    "num_beams": getattr(
                        runtime, "generation_num_beams", None
                    ),
                    "temperature": getattr(
                        runtime, "generation_temperature", None
                    ),
                    "top_p": getattr(runtime, "generation_top_p", None),
                },
                "execution_device": getattr(
                    runtime, "model_device", self._model_device
                ),
                "transformers_version": adapter_version,
                "accelerate_version": getattr(
                    runtime, "accelerate_version", None
                ),
                "torch_version": getattr(runtime, "torch_version", None),
                "torchvision_version": getattr(
                    runtime, "torchvision_version", None
                ),
                "qwen_vl_utils_version": getattr(
                    runtime, "qwen_vl_utils_version", None
                ),
                "flash_attn_version": getattr(
                    runtime, "flash_attn_version", None
                ),
                "image_size": reconstruction.get("image_size", []),
                "prompt_tokens": reconstruction.get("prompt_tokens", 0),
                "generated_tokens": reconstruction.get(
                    "generated_tokens", 0
                ),
                "raw_output": raw_output,
                "clean_output": clean_output,
                "native_format": "json_layout_with_html_tables",
                "special_tokens_removed_for_json_parsing": True,
                "layout_json": layout_payload,
                "layout_json_parse_error": layout_parse_error,
                "layout_objects_detected": len(layout_objects),
                "table_objects": table_objects,
                "table_objects_detected": len(table_objects),
                "html_tables_detected": html_table_count,
                "table_bboxes": [
                    item.get("bbox") for item in table_objects
                ],
                "bbox_policy": "provenance_only",
                "model_native_layout_detection": True,
                "external_layout_redetection": False,
                "external_table_redetection": False,
                "recropping": False,
                "table_bboxes_used_for_recropping": False,
                "normalization": "none",
                "json_repair": False,
                "semantic_repair": False,
                "continued_table_merging": False,
                "parser": "tabulus.table_ocr.parsing:parse_table_text",
                "input_policy": "canonical_mineru_crop",
            }
        }

        if not clean_output.strip():
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=[native],
                error="dots.mocr returned no generated layout output.",
            )

        if layout_parse_error is not None:
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=[native],
                error="dots.mocr output was not valid JSON layout output.",
            )

        if not layout_objects:
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=[native],
                error="dots.mocr output contained no layout objects.",
            )

        if not table_objects:
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=[native],
                error="dots.mocr output contained no model-emitted Table objects.",
            )

        if not parser_inputs:
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=[native],
                error=(
                    "dots.mocr Table objects did not contain usable HTML tables."
                ),
            )

        return self._result(
            table,
            status="ok",
            adapter_version=adapter_version,
            result_count=html_table_count,
            native_json=[native],
            native_markdown=parser_inputs,
        )
