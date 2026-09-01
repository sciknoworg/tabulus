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
from tabulus.table_ocr.parsing import (
    otsl_table_to_html,
    parse_table_text,
)


VALIDATED_PYTHON_MAJOR_MINOR = (3, 11)
VALIDATED_TRANSFORMERS_VERSION = "4.57.1"
VALIDATED_ACCELERATE_VERSION = "1.11.0"
VALIDATED_TIMM_VERSION = "1.0.27"
VALIDATED_EINOPS_VERSION = "0.8.1"

MODEL_REPO = "zenosai/MonkeyOCRv2-B-Parsing"
MODEL_REVISION = "2419139b7bcd3fda2689b2a83167172afba91c8b"
MODEL_VERSION = f"{MODEL_REPO}@{MODEL_REVISION}"

MODEL_CLASS = "MonkeyOCRv2ForCausalLM"
MODEL_DTYPE = "bfloat16"
PROCESSOR_USE_FAST = False
ATTENTION_IMPLEMENTATION = "sdpa"
CHECKPOINT_VISION_ATTENTION = "flash_attention_2"

TABLE_PROMPT = (
    "Please extract the table from the image and represent it in OTSL format."
)
TABLE_MIN_PIXELS = 1003520
MAX_NEW_TOKENS = 4096
GENERATION_DO_SAMPLE = False


@dataclass(frozen=True)
class _MonkeyOCRv2Runtime:
    torch: Any
    image_module: Any
    processor: Any
    model: Any
    model_device: str
    model_dtype: str
    model_class: str
    checkpoint_vision_attention: str | None
    resolved_vision_attention: str | None
    processor_min_pixels: int | None
    processor_max_pixels: int | None
    transformers_version: str | None
    accelerate_version: str | None
    timm_version: str | None
    einops_version: str | None
    torch_version: str | None
    torchvision_version: str | None
    pillow_version: str | None


RuntimeLoader = Callable[[str], Any]
InferenceRunner = Callable[[Path, Any], dict[str, Any]]


def _installed_package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _require_validated_version(
    package_name: str,
    expected: str,
) -> str:
    found = _installed_package_version(package_name)

    if found != expected:
        actual = found or "not installed"
        raise TableOCRDependencyError(
            "The monkeyocrv2-b-parsing adapter is validated against "
            f"{package_name}=={expected}; found {actual}."
        )

    return found


def _monkeyocrv2_device(device: str) -> str:
    normalized = device.strip().lower()

    if normalized == "gpu":
        return "cuda:0"

    if normalized.startswith("gpu:"):
        index = normalized.split(":", maxsplit=1)[1]
        if index.isdigit():
            return f"cuda:{index}"

    raise ValueError(
        "Use device='gpu' or 'gpu:<index>' for monkeyocrv2-b-parsing."
    )


def _dependency_error() -> TableOCRDependencyError:
    return TableOCRDependencyError(
        "The monkeyocrv2-b-parsing adapter requires Python 3.11, "
        "transformers==4.57.1, accelerate==1.11.0, timm==1.0.27, "
        "einops==0.8.1, PyTorch, torchvision, and Pillow. "
        "FlashAttention is not required because the validated "
        "configuration uses explicit SDPA."
    )


def _default_runtime_loader(
    model_device: str,
) -> _MonkeyOCRv2Runtime:
    if sys.version_info[:2] != VALIDATED_PYTHON_MAJOR_MINOR:
        found = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise TableOCRDependencyError(
            "The monkeyocrv2-b-parsing adapter is validated against "
            f"Python 3.11; found {found}."
        )

    transformers_version = _require_validated_version(
        "transformers",
        VALIDATED_TRANSFORMERS_VERSION,
    )
    accelerate_version = _require_validated_version(
        "accelerate",
        VALIDATED_ACCELERATE_VERSION,
    )
    timm_version = _require_validated_version(
        "timm",
        VALIDATED_TIMM_VERSION,
    )
    einops_version = _require_validated_version(
        "einops",
        VALIDATED_EINOPS_VERSION,
    )

    try:
        torch = importlib.import_module("torch")
        image_module = importlib.import_module("PIL.Image")
        transformers = importlib.import_module("transformers")

        importlib.import_module("torchvision")
        importlib.import_module("accelerate")
        importlib.import_module("timm")
        importlib.import_module("einops")
    except ImportError as exc:
        raise _dependency_error() from exc

    if not bool(torch.cuda.is_available()):
        raise RuntimeError(
            "monkeyocrv2-b-parsing requires CUDA in the validated "
            "Tabulus configuration, but CUDA is not available."
        )

    processor = transformers.AutoProcessor.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        use_fast=PROCESSOR_USE_FAST,
    )

    image_processor = processor.image_processor
    image_processor.min_pixels = TABLE_MIN_PIXELS

    config = transformers.AutoConfig.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )

    checkpoint_attention = getattr(
        config.vision_config,
        "attn_implementation",
        None,
    )
    config.vision_config.attn_implementation = ATTENTION_IMPLEMENTATION

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": model_device},
        attn_implementation=ATTENTION_IMPLEMENTATION,
    )
    model.eval()

    resolved_device = str(next(model.parameters()).device)

    if resolved_device != model_device:
        raise RuntimeError(
            "MonkeyOCRv2-B-Parsing loaded on an unexpected device: "
            f"requested {model_device}, resolved {resolved_device}."
        )

    resolved_attention = getattr(
        model.config.vision_config,
        "attn_implementation",
        None,
    )

    return _MonkeyOCRv2Runtime(
        torch=torch,
        image_module=image_module,
        processor=processor,
        model=model,
        model_device=model_device,
        model_dtype=str(next(model.parameters()).dtype),
        model_class=type(model).__name__,
        checkpoint_vision_attention=(
            str(checkpoint_attention)
            if checkpoint_attention is not None
            else None
        ),
        resolved_vision_attention=(
            str(resolved_attention)
            if resolved_attention is not None
            else None
        ),
        processor_min_pixels=getattr(image_processor, "min_pixels", None),
        processor_max_pixels=getattr(image_processor, "max_pixels", None),
        transformers_version=transformers_version,
        accelerate_version=accelerate_version,
        timm_version=timm_version,
        einops_version=einops_version,
        torch_version=_installed_package_version("torch"),
        torchvision_version=_installed_package_version("torchvision"),
        pillow_version=_installed_package_version("Pillow"),
    )


def _default_inference_runner(
    image_path: Path,
    runtime: _MonkeyOCRv2Runtime,
) -> dict[str, Any]:
    with runtime.image_module.open(image_path) as opened_image:
        image = opened_image.convert("RGB")
        source_width, source_height = image.size

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": TABLE_PROMPT},
            ],
        }
    ]

    text = runtime.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = runtime.processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(runtime.model_device)

    prompt_tokens = int(inputs["input_ids"].shape[-1])

    with runtime.torch.inference_mode():
        output_ids = runtime.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=GENERATION_DO_SAMPLE,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]

    raw_output = runtime.processor.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).strip()

    clean_output = runtime.processor.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    return {
        "raw_output": raw_output,
        "clean_output": clean_output,
        "source_image_size": [int(source_width), int(source_height)],
        "prompt_tokens": prompt_tokens,
        "generated_tokens": int(generated_ids.numel()),
        "raw_output_chars": len(raw_output),
        "clean_output_chars": len(clean_output),
    }


class MonkeyOCRv2BParsingAdapter:
    """MonkeyOCRv2-B-Parsing reconstruction for canonical MinerU crops."""

    NAME = "monkeyocrv2-b-parsing"
    DISPLAY_NAME = "MonkeyOCRv2-B-Parsing"
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
        self._model_device = _monkeyocrv2_device(device)
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
                error=f"Could not initialize MonkeyOCRv2-B-Parsing: {exc}",
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
                error=(
                    "MonkeyOCRv2-B-Parsing table reconstruction "
                    f"failed: {exc}"
                ),
            )

        raw_value = reconstruction.get("raw_output", "")
        clean_value = reconstruction.get("clean_output", "")
        raw_output = raw_value if isinstance(raw_value, str) else str(raw_value or "")
        clean_output = (
            clean_value if isinstance(clean_value, str) else str(clean_value or "")
        )

        html_table = otsl_table_to_html(clean_output)
        parser_error: str | None = None

        try:
            parsed_tables = parse_table_text(html_table) if html_table else []
        except Exception as exc:
            parsed_tables = []
            parser_error = str(exc)

        usable_table = bool(
            len(parsed_tables) == 1
            and parsed_tables[0].rows
            and parsed_tables[0].n_cols > 0
        )

        native = {
            "monkeyocrv2_b_parsing": {
                "model_repo": MODEL_REPO,
                "model_revision": MODEL_REVISION,
                "model_class": getattr(runtime, "model_class", MODEL_CLASS),
                "prompt": TABLE_PROMPT,
                "native_format": "otsl",
                "model_dtype": MODEL_DTYPE,
                "resolved_model_dtype": getattr(runtime, "model_dtype", None),
                "checkpoint_vision_attention": getattr(
                    runtime,
                    "checkpoint_vision_attention",
                    CHECKPOINT_VISION_ATTENTION,
                ),
                "attention_implementation": getattr(
                    runtime,
                    "resolved_vision_attention",
                    ATTENTION_IMPLEMENTATION,
                ),
                "processor_use_fast": PROCESSOR_USE_FAST,
                "table_min_pixels": TABLE_MIN_PIXELS,
                "processor_min_pixels": getattr(
                    runtime, "processor_min_pixels", None
                ),
                "processor_max_pixels": getattr(
                    runtime, "processor_max_pixels", None
                ),
                "max_new_tokens": MAX_NEW_TOKENS,
                "generation_do_sample": GENERATION_DO_SAMPLE,
                "generation_temperature": None,
                "generation_top_p": None,
                "generation_top_k": None,
                "execution_device": getattr(
                    runtime, "model_device", self._model_device
                ),
                "python_version": (
                    f"{sys.version_info.major}.{sys.version_info.minor}."
                    f"{sys.version_info.micro}"
                ),
                "transformers_version": adapter_version,
                "accelerate_version": getattr(runtime, "accelerate_version", None),
                "timm_version": getattr(runtime, "timm_version", None),
                "einops_version": getattr(runtime, "einops_version", None),
                "torch_version": getattr(runtime, "torch_version", None),
                "torchvision_version": getattr(
                    runtime, "torchvision_version", None
                ),
                "pillow_version": getattr(runtime, "pillow_version", None),
                "source_image_size": reconstruction.get("source_image_size", []),
                "prompt_tokens": reconstruction.get("prompt_tokens", 0),
                "generated_tokens": reconstruction.get("generated_tokens", 0),
                "raw_output_chars": reconstruction.get(
                    "raw_output_chars", len(raw_output)
                ),
                "clean_output_chars": reconstruction.get(
                    "clean_output_chars", len(clean_output)
                ),
                "raw_output": raw_output,
                "clean_output": clean_output,
                "special_tokens_removed_for_parsing": True,
                "normalization": (
                    "tabulus.table_ocr.parsing:otsl_table_to_html"
                ),
                "parser": "tabulus.table_ocr.parsing:parse_table_text",
                "structured_tables_detected": len(parsed_tables),
                "parser_error": parser_error,
                "input_policy": "canonical_mineru_crop",
                "image_preprocessing": {
                    "external": "rgb_conversion_only",
                    "processor": "AutoProcessor",
                    "processor_use_fast": PROCESSOR_USE_FAST,
                    "upstream_table_min_pixels": TABLE_MIN_PIXELS,
                    "model_internal_resize": True,
                },
                "layout_redetection": False,
                "recropping": False,
                "external_recropping": False,
                "semantic_repair": False,
            }
        }

        if not clean_output.strip():
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=[native],
                error=(
                    "MonkeyOCRv2-B-Parsing returned no generated OTSL output."
                ),
            )

        if not usable_table:
            error = (
                "MonkeyOCRv2-B-Parsing output did not contain one usable "
                "OTSL table."
            )
            if parser_error:
                error += f" Shared parser error: {parser_error}"

            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=[native],
                error=error,
            )

        return self._result(
            table,
            status="ok",
            adapter_version=adapter_version,
            result_count=1,
            native_json=[native],
            native_markdown=[html_table],
        )
