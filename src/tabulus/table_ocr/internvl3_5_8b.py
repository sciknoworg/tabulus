from __future__ import annotations

import importlib
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
VALIDATED_TRANSFORMERS_VERSION = "4.55.0"
VALIDATED_ACCELERATE_VERSION = "1.14.0"
VALIDATED_TORCH_VERSION = "2.7.0+cu128"
VALIDATED_TORCHVISION_VERSION = "0.22.0+cu128"

MODEL_REPO = "OpenGVLab/InternVL3_5-8B-HF"
MODEL_REVISION = "741a7d03020411e666c6109218ab71e08151ef86"
MODEL_VERSION = f"{MODEL_REPO}@{MODEL_REVISION}"
MODEL_CLASS = "InternVLForConditionalGeneration"
MODEL_TYPE = "internvl"
TEXT_MODEL_TYPE = "qwen3"
VISION_MODEL_TYPE = "internvl_vision"
PROCESSOR_CLASS = "InternVLProcessor"
IMAGE_PROCESSOR_CLASS = "GotOcr2ImageProcessorFast"
TOKENIZER_CLASS = "Qwen2TokenizerFast"
IMAGE_SEQ_LENGTH = 256

TABLE_PROMPT = (
    "Extract the table from this image and output only a valid HTML table. "
    "Return only the table markup, with no explanation, no markdown fences, "
    "and no extra text before or after the table."
)

MODEL_DTYPE = "bfloat16"
ATTENTION_IMPLEMENTATION = "sdpa"
MAX_NEW_TOKENS = 8192
GENERATION_DO_SAMPLE = False
GENERATION_NUM_BEAMS = 1
GENERATION_TEMPERATURE = 1.0
GENERATION_TOP_P = 1.0
GENERATION_REPETITION_PENALTY = 1.0


@dataclass(frozen=True)
class _InternVLRuntime:
    torch: Any
    image_module: Any
    processor: Any
    model: Any
    model_device: str
    model_dtype: str
    model_class: str
    model_type: str
    text_model_type: str
    vision_model_type: str
    processor_class: str
    image_processor_class: str
    tokenizer_class: str
    image_seq_length: int
    attention_implementation: str
    text_attention_implementation: str
    vision_attention_implementation: str
    generation_do_sample: bool
    generation_num_beams: int
    generation_temperature: float
    generation_top_p: float
    generation_repetition_penalty: float
    generation_bos_token_id: int | None
    generation_eos_token_id: int | None
    transformers_version: str
    accelerate_version: str
    torch_version: str
    torchvision_version: str


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
            "The internvl3-5-8b adapter is validated against "
            f"{package_name}=={expected}; found {actual}."
        )
    return found


def _internvl_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized == "gpu":
        return "cuda:0"
    if normalized.startswith("gpu:"):
        index = normalized.split(":", maxsplit=1)[1]
        if index.isdigit():
            return f"cuda:{index}"
    raise ValueError("Use device='gpu' or 'gpu:<index>' for internvl3-5-8b.")


def _dependency_error() -> TableOCRDependencyError:
    return TableOCRDependencyError(
        "The internvl3-5-8b adapter requires Python 3.12, "
        "transformers==4.55.0, accelerate==1.14.0, "
        "torch==2.7.0+cu128, torchvision==0.22.0+cu128, and Pillow "
        "in the active environment. The pinned model snapshot must already "
        "be present in the local Hugging Face cache."
    )


def _resolved_attention(config: Any) -> str:
    return str(getattr(config, "_attn_implementation", ""))


def _default_runtime_loader(model_device: str) -> _InternVLRuntime:
    if sys.version_info[:2] != VALIDATED_PYTHON_MAJOR_MINOR:
        found = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise TableOCRDependencyError(
            "The internvl3-5-8b adapter is validated against "
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

    try:
        torch = importlib.import_module("torch")
        image_module = importlib.import_module("PIL.Image")
        transformers = importlib.import_module("transformers")
        importlib.import_module("torchvision")
        importlib.import_module("accelerate")
    except ImportError as exc:
        raise _dependency_error() from exc

    if not bool(torch.cuda.is_available()):
        raise RuntimeError(
            "internvl3-5-8b requires CUDA in the validated Tabulus "
            "configuration, but CUDA is not available."
        )

    processor = transformers.AutoProcessor.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        local_files_only=True,
    )

    resolved_processor_class = type(processor).__name__
    if resolved_processor_class != PROCESSOR_CLASS:
        raise RuntimeError(
            "InternVL3.5-8B loaded an unexpected processor class: "
            f"{resolved_processor_class!r}."
        )

    image_processor = getattr(processor, "image_processor", None)
    resolved_image_processor_class = type(image_processor).__name__
    if resolved_image_processor_class != IMAGE_PROCESSOR_CLASS:
        raise RuntimeError(
            "InternVL3.5-8B loaded an unexpected image processor class: "
            f"{resolved_image_processor_class!r}."
        )

    tokenizer = getattr(processor, "tokenizer", None)
    resolved_tokenizer_class = type(tokenizer).__name__
    if resolved_tokenizer_class != TOKENIZER_CLASS:
        raise RuntimeError(
            "InternVL3.5-8B loaded an unexpected tokenizer class: "
            f"{resolved_tokenizer_class!r}."
        )

    model_class = getattr(transformers, MODEL_CLASS, None)
    if model_class is None:
        raise TableOCRDependencyError(
            f"transformers=={VALIDATED_TRANSFORMERS_VERSION} does not expose "
            f"{MODEL_CLASS}."
        )

    model = model_class.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTENTION_IMPLEMENTATION,
        low_cpu_mem_usage=True,
        device_map={"": model_device},
    ).eval()

    resolved_device = str(next(model.parameters()).device)
    if resolved_device != model_device:
        raise RuntimeError(
            "InternVL3.5-8B loaded on an unexpected device: "
            f"requested {model_device}, resolved {resolved_device}."
        )

    resolved_dtype = str(next(model.parameters()).dtype)
    if resolved_dtype != "torch.bfloat16":
        raise RuntimeError(
            "InternVL3.5-8B loaded with an unexpected dtype: "
            f"{resolved_dtype!r}."
        )

    resolved_model_class = type(model).__name__
    if resolved_model_class != MODEL_CLASS:
        raise RuntimeError(
            "InternVL3.5-8B loaded an unexpected model class: "
            f"{resolved_model_class!r}."
        )

    resolved_model_type = str(getattr(model.config, "model_type", ""))
    if resolved_model_type != MODEL_TYPE:
        raise RuntimeError(
            "InternVL3.5-8B loaded an unexpected model type: "
            f"{resolved_model_type!r}."
        )

    text_config = getattr(model.config, "text_config", None)
    vision_config = getattr(model.config, "vision_config", None)
    resolved_text_model_type = str(getattr(text_config, "model_type", ""))
    resolved_vision_model_type = str(getattr(vision_config, "model_type", ""))

    if resolved_text_model_type != TEXT_MODEL_TYPE:
        raise RuntimeError(
            "InternVL3.5-8B loaded an unexpected text model type: "
            f"{resolved_text_model_type!r}."
        )
    if resolved_vision_model_type != VISION_MODEL_TYPE:
        raise RuntimeError(
            "InternVL3.5-8B loaded an unexpected vision model type: "
            f"{resolved_vision_model_type!r}."
        )

    resolved_image_seq_length = int(
        getattr(model.config, "image_seq_length", 0)
    )
    if resolved_image_seq_length != IMAGE_SEQ_LENGTH:
        raise RuntimeError(
            "InternVL3.5-8B loaded an unexpected image_seq_length: "
            f"{resolved_image_seq_length!r}."
        )

    resolved_attention = _resolved_attention(model.config)
    resolved_text_attention = _resolved_attention(text_config)
    resolved_vision_attention = _resolved_attention(vision_config)
    attentions = (
        resolved_attention,
        resolved_text_attention,
        resolved_vision_attention,
    )
    if any(value != ATTENTION_IMPLEMENTATION for value in attentions):
        raise RuntimeError(
            "InternVL3.5-8B loaded with unexpected attention implementations: "
            f"top={resolved_attention!r}, text={resolved_text_attention!r}, "
            f"vision={resolved_vision_attention!r}."
        )

    generation_config = model.generation_config
    resolved_do_sample = bool(getattr(generation_config, "do_sample", False))
    resolved_num_beams = int(getattr(generation_config, "num_beams", 1))
    resolved_temperature = float(getattr(generation_config, "temperature", 1.0))
    resolved_top_p = float(getattr(generation_config, "top_p", 1.0))
    resolved_repetition_penalty = float(
        getattr(generation_config, "repetition_penalty", 1.0)
    )

    expected_generation = (
        resolved_do_sample is GENERATION_DO_SAMPLE
        and resolved_num_beams == GENERATION_NUM_BEAMS
        and resolved_temperature == GENERATION_TEMPERATURE
        and resolved_top_p == GENERATION_TOP_P
        and resolved_repetition_penalty == GENERATION_REPETITION_PENALTY
    )
    if not expected_generation:
        raise RuntimeError(
            "InternVL3.5-8B resolved unexpected generation defaults: "
            f"do_sample={resolved_do_sample!r}, "
            f"num_beams={resolved_num_beams!r}, "
            f"temperature={resolved_temperature!r}, "
            f"top_p={resolved_top_p!r}, "
            f"repetition_penalty={resolved_repetition_penalty!r}."
        )

    bos_token_id = getattr(generation_config, "bos_token_id", None)
    eos_token_id = getattr(generation_config, "eos_token_id", None)

    return _InternVLRuntime(
        torch=torch,
        image_module=image_module,
        processor=processor,
        model=model,
        model_device=model_device,
        model_dtype=resolved_dtype,
        model_class=resolved_model_class,
        model_type=resolved_model_type,
        text_model_type=resolved_text_model_type,
        vision_model_type=resolved_vision_model_type,
        processor_class=resolved_processor_class,
        image_processor_class=resolved_image_processor_class,
        tokenizer_class=resolved_tokenizer_class,
        image_seq_length=resolved_image_seq_length,
        attention_implementation=resolved_attention,
        text_attention_implementation=resolved_text_attention,
        vision_attention_implementation=resolved_vision_attention,
        generation_do_sample=resolved_do_sample,
        generation_num_beams=resolved_num_beams,
        generation_temperature=resolved_temperature,
        generation_top_p=resolved_top_p,
        generation_repetition_penalty=resolved_repetition_penalty,
        generation_bos_token_id=(
            int(bos_token_id) if bos_token_id is not None else None
        ),
        generation_eos_token_id=(
            int(eos_token_id) if eos_token_id is not None else None
        ),
        transformers_version=transformers_version,
        accelerate_version=accelerate_version,
        torch_version=torch_version,
        torchvision_version=torchvision_version,
    )


def _default_inference_runner(
    image_path: Path,
    runtime: _InternVLRuntime,
) -> dict[str, Any]:
    with runtime.image_module.open(image_path) as source_image:
        width, height = source_image.size
        image = source_image.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": TABLE_PROMPT},
            ],
        }
    ]

    chat_text = runtime.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = runtime.processor(
        text=[chat_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    for key, value in inputs.items():
        if not isinstance(value, runtime.torch.Tensor):
            continue
        if runtime.torch.is_floating_point(value):
            inputs[key] = value.to(
                device=runtime.model_device,
                dtype=runtime.torch.bfloat16,
            )
        else:
            inputs[key] = value.to(runtime.model_device)

    input_ids = inputs["input_ids"]
    prompt_len = int(input_ids.shape[1])
    pixel_values = inputs.get("pixel_values")

    with runtime.torch.inference_mode():
        generated = runtime.model.generate(
            **inputs,
            do_sample=GENERATION_DO_SAMPLE,
            num_beams=GENERATION_NUM_BEAMS,
            temperature=GENERATION_TEMPERATURE,
            top_p=GENERATION_TOP_P,
            repetition_penalty=GENERATION_REPETITION_PENALTY,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    new_tokens = generated[:, prompt_len:]
    values = runtime.processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
    )
    output = values[0] if values else ""
    generated_tokens = int(new_tokens.shape[1])

    return {
        "output": output,
        "image_size": [int(width), int(height)],
        "prompt_tokens": prompt_len,
        "generated_tokens": generated_tokens,
        "hit_token_ceiling": generated_tokens >= MAX_NEW_TOKENS,
        "pixel_values_shape": (
            [int(value) for value in pixel_values.shape]
            if pixel_values is not None
            else []
        ),
        "pixel_values_dtype": (
            str(pixel_values.dtype) if pixel_values is not None else None
        ),
    }


class InternVL35_8BAdapter:
    """InternVL3.5-8B table reconstruction from canonical MinerU crops."""

    NAME = "internvl3-5-8b"
    DISPLAY_NAME = "InternVL3.5-8B"
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
        self._model_device = _internvl_device(device)
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
                error=f"Could not initialize InternVL3.5-8B: {exc}",
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
                error=f"InternVL3.5-8B table reconstruction failed: {exc}",
            )

        value = reconstruction.get("output", "")
        output = value if isinstance(value, str) else str(value or "")
        html_tables = extract_html_tables(output)
        generated_tokens = int(reconstruction.get("generated_tokens", 0) or 0)
        hit_token_ceiling = bool(
            reconstruction.get(
                "hit_token_ceiling",
                generated_tokens >= MAX_NEW_TOKENS,
            )
        )

        native = {
            "internvl3_5_8b": {
                "model_repo": MODEL_REPO,
                "model_revision": MODEL_REVISION,
                "model_class": getattr(runtime, "model_class", None),
                "model_type": getattr(runtime, "model_type", None),
                "text_model_type": getattr(runtime, "text_model_type", None),
                "vision_model_type": getattr(runtime, "vision_model_type", None),
                "processor_class": getattr(runtime, "processor_class", None),
                "image_processor_class": getattr(
                    runtime, "image_processor_class", None
                ),
                "tokenizer_class": getattr(runtime, "tokenizer_class", None),
                "image_seq_length": getattr(runtime, "image_seq_length", None),
                "task": "table_to_html",
                "prompt_source": "tabulus_defined",
                "prompt": TABLE_PROMPT,
                "model_load_dtype": MODEL_DTYPE,
                "resolved_model_dtype": getattr(runtime, "model_dtype", None),
                "attention_implementation": getattr(
                    runtime,
                    "attention_implementation",
                    ATTENTION_IMPLEMENTATION,
                ),
                "text_attention_implementation": getattr(
                    runtime,
                    "text_attention_implementation",
                    None,
                ),
                "vision_attention_implementation": getattr(
                    runtime,
                    "vision_attention_implementation",
                    None,
                ),
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": GENERATION_DO_SAMPLE,
                "num_beams": GENERATION_NUM_BEAMS,
                "temperature": GENERATION_TEMPERATURE,
                "top_p": GENERATION_TOP_P,
                "repetition_penalty": GENERATION_REPETITION_PENALTY,
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
                    "repetition_penalty": getattr(
                        runtime, "generation_repetition_penalty", None
                    ),
                    "bos_token_id": getattr(
                        runtime, "generation_bos_token_id", None
                    ),
                    "eos_token_id": getattr(
                        runtime, "generation_eos_token_id", None
                    ),
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
                "local_files_only": True,
                "image_size": reconstruction.get("image_size", []),
                "pixel_values_shape": reconstruction.get(
                    "pixel_values_shape", []
                ),
                "pixel_values_dtype": reconstruction.get(
                    "pixel_values_dtype", None
                ),
                "prompt_tokens": reconstruction.get("prompt_tokens", 0),
                "generated_tokens": generated_tokens,
                "hit_token_ceiling": hit_token_ceiling,
                "output": output,
                "native_format": "html",
                "special_tokens_removed_for_parsing": True,
                "html_tables_detected": len(html_tables),
                "normalization": "none",
                "model_native_image_processing": True,
                "official_hf_processor_only": True,
                "external_layout_redetection": False,
                "external_table_redetection": False,
                "external_recropping": False,
                "bbox_recropping": False,
                "tabulus_content_aware_tiling": False,
                "semantic_repair": False,
                "continued_table_merging": False,
                "reference_resolution": False,
                "parser": "tabulus.table_ocr.parsing:parse_table_text",
                "input_policy": "canonical_mineru_crop",
            }
        }

        if not output.strip():
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=[native],
                error="InternVL3.5-8B returned no generated table output.",
            )

        if hit_token_ceiling:
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=[native],
                error=(
                    "InternVL3.5-8B reached the max_new_tokens generation "
                    "ceiling before a natural stop."
                ),
            )

        if not html_tables:
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=[native],
                error=(
                    "InternVL3.5-8B output did not contain a usable HTML table."
                ),
            )

        return self._result(
            table,
            status="ok",
            adapter_version=adapter_version,
            result_count=len(html_tables),
            native_json=[native],
            native_markdown=[output],
        )
