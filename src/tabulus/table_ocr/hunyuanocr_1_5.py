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
VALIDATED_TRANSFORMERS_VERSION = "5.13.0"
VALIDATED_ACCELERATE_VERSION = "1.14.0"
VALIDATED_TORCH_VERSION = "2.11.0+cu130"
VALIDATED_TORCHVISION_VERSION = "0.26.0+cu130"

MODEL_REPO = "tencent/HunyuanOCR"
MODEL_REVISION = "47644ecc4fc854efa4f505155158831f36773ee4"
MODEL_VERSION = f"{MODEL_REPO}@{MODEL_REVISION}"
MODEL_CLASS = "HunYuanVLForConditionalGeneration"
MODEL_TYPE = "hunyuan_vl"

TABLE_PROMPT = "把图中的表格解析为HTML。"
MODEL_DTYPE = "bfloat16"
ATTENTION_IMPLEMENTATION = "eager"
MAX_NEW_TOKENS = 8192
GENERATION_DO_SAMPLE = False
GENERATION_REPETITION_PENALTY = 1.08
GENERATION_USE_CACHE = True

TAIL_MIN_REPEATS = 8
TAIL_MAX_UNIT = 256
TAIL_CHECK_START_CHARS = 4000
TAIL_CHECK_STEP_CHARS = 1000
TAIL_TOKEN_PROBE_STEP = 64
TAIL_WINDOW_CHARS = 8000
CLEANUP_MIN_REPEATS = 10


@dataclass(frozen=True)
class _HunyuanOCRRuntime:
    torch: Any
    image_module: Any
    transformers: Any
    processor: Any
    model: Any
    model_device: str
    model_dtype: str
    model_class: str
    model_type: str
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
            "The hunyuanocr-1-5 adapter is validated against "
            f"{package_name}=={expected}; found {actual}."
        )
    return found


def _hunyuanocr_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized == "gpu":
        return "cuda:0"
    if normalized.startswith("gpu:"):
        index = normalized.split(":", maxsplit=1)[1]
        if index.isdigit():
            return f"cuda:{index}"
    raise ValueError("Use device='gpu' or 'gpu:<index>' for hunyuanocr-1-5.")


def _dependency_error() -> TableOCRDependencyError:
    return TableOCRDependencyError(
        "The hunyuanocr-1-5 adapter requires Python 3.12, "
        "transformers==5.13.0, accelerate==1.14.0, "
        "torch==2.11.0+cu130, torchvision==0.26.0+cu130, "
        "and Pillow in the active environment."
    )


def _has_tail_repetition(
    text: str,
    *,
    min_repeats: int = TAIL_MIN_REPEATS,
    max_unit: int = TAIL_MAX_UNIT,
) -> bool:
    n_chars = len(text)
    if n_chars < min_repeats * 2:
        return False

    upper = min(max_unit, n_chars // min_repeats)
    for length in range(1, upper + 1):
        unit = text[-length:]
        if not unit.strip():
            continue

        repeated = True
        for repeat_index in range(2, min_repeats + 1):
            start = -length * repeat_index
            end = -length * (repeat_index - 1)
            if text[start:end] != unit:
                repeated = False
                break
        if repeated:
            return True

    return False


def _clean_repeated_substrings(
    text: str,
    *,
    min_repeats: int = CLEANUP_MIN_REPEATS,
) -> str:
    n_chars = len(text)
    if n_chars < 2000:
        return text

    for length in range(2, n_chars // min_repeats + 1):
        candidate = text[-length:]
        count = 0
        index = n_chars - length

        while index >= 0 and text[index : index + length] == candidate:
            count += 1
            index -= length

        if count >= min_repeats:
            return text[: n_chars - length * (count - 1)]

    return text


def _build_tail_repetition_stop(
    runtime: _HunyuanOCRRuntime,
    *,
    prompt_len: int,
) -> Any:
    tokenizer = runtime.processor.tokenizer
    stopping_base = runtime.transformers.StoppingCriteria

    class TailRepetitionStop(stopping_base):
        def __init__(self) -> None:
            self.next_check_at_chars = TAIL_CHECK_START_CHARS
            self.last_probe_tokens = 0
            self.triggered = False

        def __call__(
            self,
            input_ids: Any,
            scores: Any,
            **kwargs: Any,
        ) -> bool:
            del scores, kwargs

            if self.triggered:
                return True

            new_tokens = input_ids[0, prompt_len:]
            n_new = int(new_tokens.numel())
            if n_new - self.last_probe_tokens < TAIL_TOKEN_PROBE_STEP:
                return False

            self.last_probe_tokens = n_new
            try:
                decoded = tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            except Exception:
                return False

            decoded_chars = len(decoded)
            if decoded_chars < self.next_check_at_chars:
                return False

            self.next_check_at_chars = decoded_chars + TAIL_CHECK_STEP_CHARS

            if _has_tail_repetition(
                decoded[-TAIL_WINDOW_CHARS:],
                min_repeats=TAIL_MIN_REPEATS,
                max_unit=TAIL_MAX_UNIT,
            ):
                self.triggered = True
                return True

            return False

    return TailRepetitionStop()


def _default_runtime_loader(model_device: str) -> _HunyuanOCRRuntime:
    if sys.version_info[:2] != VALIDATED_PYTHON_MAJOR_MINOR:
        found = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise TableOCRDependencyError(
            "The hunyuanocr-1-5 adapter is validated against "
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
            "hunyuanocr-1-5 requires CUDA in the validated "
            "Tabulus configuration, but CUDA is not available."
        )

    processor = transformers.AutoProcessor.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        use_fast=False,
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
        dtype=torch.bfloat16,
        attn_implementation=ATTENTION_IMPLEMENTATION,
    ).to(model_device).eval()

    resolved_device = str(next(model.parameters()).device)
    if resolved_device != model_device:
        raise RuntimeError(
            "HunyuanOCR-1.5 loaded on an unexpected device: "
            f"requested {model_device}, resolved {resolved_device}."
        )

    resolved_dtype = str(next(model.parameters()).dtype)
    if resolved_dtype != "torch.bfloat16":
        raise RuntimeError(
            "HunyuanOCR-1.5 loaded with an unexpected dtype: "
            f"{resolved_dtype!r}."
        )

    resolved_model_class = type(model).__name__
    if resolved_model_class != MODEL_CLASS:
        raise RuntimeError(
            "HunyuanOCR-1.5 loaded an unexpected model class: "
            f"{resolved_model_class!r}."
        )

    resolved_model_type = str(getattr(model.config, "model_type", ""))
    if resolved_model_type != MODEL_TYPE:
        raise RuntimeError(
            "HunyuanOCR-1.5 loaded an unexpected model type: "
            f"{resolved_model_type!r}."
        )

    return _HunyuanOCRRuntime(
        torch=torch,
        image_module=image_module,
        transformers=transformers,
        processor=processor,
        model=model,
        model_device=model_device,
        model_dtype=resolved_dtype,
        model_class=resolved_model_class,
        model_type=resolved_model_type,
        transformers_version=transformers_version,
        accelerate_version=accelerate_version,
        torch_version=torch_version,
        torchvision_version=torchvision_version,
    )


def _default_inference_runner(
    image_path: Path,
    runtime: _HunyuanOCRRuntime,
) -> dict[str, Any]:
    with runtime.image_module.open(image_path) as source_image:
        width, height = source_image.size
        image = source_image.convert("RGB")

    messages = [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": TABLE_PROMPT},
            ],
        },
    ]

    chat_text = runtime.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = runtime.processor(
        text=[chat_text],
        images=image,
        padding=True,
        return_tensors="pt",
    ).to(runtime.model_device)

    input_ids = inputs["input_ids"]
    prompt_len = int(input_ids.shape[1])

    tail_stop = _build_tail_repetition_stop(runtime, prompt_len=prompt_len)
    stopping_criteria = runtime.transformers.StoppingCriteriaList([tail_stop])

    tokenizer = runtime.processor.tokenizer
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or eos_token_id

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": GENERATION_DO_SAMPLE,
        "repetition_penalty": GENERATION_REPETITION_PENALTY,
        "use_cache": GENERATION_USE_CACHE,
        "stopping_criteria": stopping_criteria,
    }
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id

    with runtime.torch.inference_mode():
        generated_ids = runtime.model.generate(**inputs, **generation_kwargs)

    trimmed = [
        output_ids[len(source_ids):]
        for source_ids, output_ids in zip(input_ids, generated_ids)
    ]

    raw_values = runtime.processor.batch_decode(
        trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    decoded_values = runtime.processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    raw_output = raw_values[0] if raw_values else ""
    decoded_output = decoded_values[0] if decoded_values else ""
    clean_output = _clean_repeated_substrings(decoded_output)

    return {
        "raw_output": raw_output,
        "decoded_output": decoded_output,
        "clean_output": clean_output,
        "image_size": [int(width), int(height)],
        "prompt_tokens": prompt_len,
        "generated_tokens": int(trimmed[0].numel()) if trimmed else 0,
        "tail_repetition_stop_triggered": bool(tail_stop.triggered),
        "repetition_cleanup_changed_output": clean_output != decoded_output,
    }


class HunyuanOCR15Adapter:
    """HunyuanOCR-1.5 table reconstruction for canonical MinerU crops."""

    NAME = "hunyuanocr-1-5"
    DISPLAY_NAME = "HunyuanOCR-1.5"
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
        self._model_device = _hunyuanocr_device(device)
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
                error=f"Could not initialize HunyuanOCR-1.5: {exc}",
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
                error=f"HunyuanOCR-1.5 table reconstruction failed: {exc}",
            )

        def as_text(key: str) -> str:
            value = reconstruction.get(key, "")
            return value if isinstance(value, str) else str(value or "")

        raw_output = as_text("raw_output")
        decoded_output = as_text("decoded_output")
        clean_output = as_text("clean_output")
        html_tables = extract_html_tables(clean_output)

        native = {
            "hunyuanocr_1_5": {
                "model_repo": MODEL_REPO,
                "model_revision": MODEL_REVISION,
                "model_class": getattr(runtime, "model_class", None),
                "model_type": getattr(runtime, "model_type", None),
                "task": "table",
                "prompt": TABLE_PROMPT,
                "model_load_dtype": MODEL_DTYPE,
                "resolved_model_dtype": getattr(runtime, "model_dtype", None),
                "attention_implementation": ATTENTION_IMPLEMENTATION,
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": GENERATION_DO_SAMPLE,
                "repetition_penalty": GENERATION_REPETITION_PENALTY,
                "use_cache": GENERATION_USE_CACHE,
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
                "image_size": reconstruction.get("image_size", []),
                "prompt_tokens": reconstruction.get("prompt_tokens", 0),
                "generated_tokens": reconstruction.get("generated_tokens", 0),
                "raw_output": raw_output,
                "decoded_output": decoded_output,
                "clean_output": clean_output,
                "native_format": "html",
                "special_tokens_removed_for_parsing": True,
                "official_repetition_controls": {
                    "tail_min_repeats": TAIL_MIN_REPEATS,
                    "tail_max_unit": TAIL_MAX_UNIT,
                    "tail_check_start_chars": TAIL_CHECK_START_CHARS,
                    "tail_check_step_chars": TAIL_CHECK_STEP_CHARS,
                    "tail_token_probe_step": TAIL_TOKEN_PROBE_STEP,
                    "tail_window_chars": TAIL_WINDOW_CHARS,
                    "cleanup_min_repeats": CLEANUP_MIN_REPEATS,
                    "tail_stop_triggered": reconstruction.get(
                        "tail_repetition_stop_triggered", False
                    ),
                    "cleanup_changed_output": reconstruction.get(
                        "repetition_cleanup_changed_output", False
                    ),
                },
                "normalization": "none",
                "document_markdown_postprocessing": False,
                "parser": "tabulus.table_ocr.parsing:parse_table_text",
                "html_tables_detected": len(html_tables),
                "input_policy": "canonical_mineru_crop",
                "layout_redetection": False,
                "table_redetection": False,
                "recropping": False,
                "semantic_repair": False,
                "continued_table_merging": False,
            }
        }

        if not clean_output.strip():
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=[native],
                error="HunyuanOCR-1.5 returned no generated table output.",
            )

        if not html_tables:
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=[native],
                error=(
                    "HunyuanOCR-1.5 output did not contain a usable HTML table."
                ),
            )

        return self._result(
            table,
            status="ok",
            adapter_version=adapter_version,
            result_count=len(html_tables),
            native_json=[native],
            native_markdown=[clean_output],
        )
