from __future__ import annotations

import importlib
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


VALIDATED_TRANSFORMERS_VERSION = "4.51.0"

MODEL_REPO = "ByteDance/Dolphin-v2"
MODEL_REVISION = "c37c62768c644bb594da4283149c627765aa80f3"
MODEL_VERSION = f"{MODEL_REPO}@{MODEL_REVISION}"

BACKBONE_ARCHITECTURE = "Qwen2.5-VL"
MODEL_CLASS = "Qwen2_5_VLForConditionalGeneration"
TABLE_PROMPT = "Parse the table in the image."
MODEL_DTYPE = "bfloat16"
MAX_NEW_TOKENS = 4096
GENERATION_DO_SAMPLE = False
GENERATION_TEMPERATURE = None

DOLPHIN_RESIZE_MAX_SIZE = 1600
DOLPHIN_RESIZE_MIN_SIZE = 28


@dataclass(frozen=True)
class _DolphinV2Runtime:
    torch: Any
    image_module: Any
    processor: Any
    model: Any
    process_vision_info: Callable[[Any], tuple[Any, Any]]
    model_device: str
    model_dtype: str
    model_class: str
    transformers_version: str | None
    torch_version: str | None
    qwen_vl_utils_version: str | None
    pillow_version: str | None


RuntimeLoader = Callable[[str], Any]
InferenceRunner = Callable[[Path, Any], dict[str, Any]]


def _installed_package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _dolphin_v2_device(device: str) -> str:
    normalized = device.strip().lower()

    if normalized == "gpu":
        return "cuda"

    if normalized.startswith("gpu:"):
        index = normalized.split(":", maxsplit=1)[1]
        if index.isdigit():
            return f"cuda:{index}"

    raise ValueError(
        "Use device='gpu' or 'gpu:<index>' for dolphin-v2."
    )


def _dependency_error() -> TableOCRDependencyError:
    return TableOCRDependencyError(
        "The dolphin-v2 adapter requires transformers==4.51.0, "
        "PyTorch, Pillow, and qwen-vl-utils in the active environment."
    )


def _default_runtime_loader(
    model_device: str,
) -> _DolphinV2Runtime:
    transformers_version = _installed_package_version("transformers")

    if transformers_version != VALIDATED_TRANSFORMERS_VERSION:
        found = transformers_version or "not installed"
        raise TableOCRDependencyError(
            "The dolphin-v2 adapter is validated against "
            f"transformers=={VALIDATED_TRANSFORMERS_VERSION}; "
            f"found {found}."
        )

    try:
        torch = importlib.import_module("torch")
        image_module = importlib.import_module("PIL.Image")
        transformers = importlib.import_module("transformers")
        qwen_vl_utils = importlib.import_module("qwen_vl_utils")

        model_class = getattr(
            transformers,
            MODEL_CLASS,
        )
        process_vision_info = getattr(
            qwen_vl_utils,
            "process_vision_info",
        )
    except (ImportError, AttributeError) as exc:
        raise _dependency_error() from exc

    if not bool(torch.cuda.is_available()):
        raise RuntimeError(
            "dolphin-v2 requires CUDA in the validated Tabulus "
            "configuration, but CUDA is not available."
        )

    processor = transformers.AutoProcessor.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
    )

    model = model_class.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
    )
    model.eval()
    model = model.to(model_device)
    model = model.bfloat16()

    return _DolphinV2Runtime(
        torch=torch,
        image_module=image_module,
        processor=processor,
        model=model,
        process_vision_info=process_vision_info,
        model_device=model_device,
        model_dtype=str(getattr(model, "dtype", "unknown")),
        model_class=type(model).__name__,
        transformers_version=transformers_version,
        torch_version=_installed_package_version("torch"),
        qwen_vl_utils_version=_installed_package_version(
            "qwen-vl-utils"
        ),
        pillow_version=_installed_package_version("Pillow"),
    )


def _dolphin_resize(
    image: Any,
    *,
    max_size: int = DOLPHIN_RESIZE_MAX_SIZE,
    min_size: int = DOLPHIN_RESIZE_MIN_SIZE,
) -> Any:
    """Apply Dolphin's official element-level resize policy."""

    width, height = image.size

    if (
        max(width, height) < max_size
        and min(width, height) >= min_size
    ):
        return image

    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        image = image.resize((new_width, new_height))
        width, height = image.size

    if min(width, height) < min_size:
        if width < height:
            new_width = min_size
            new_height = int(height * (min_size / width))
        else:
            new_height = min_size
            new_width = int(width * (min_size / height))

        image = image.resize((new_width, new_height))

    return image


def _default_inference_runner(
    image_path: Path,
    runtime: _DolphinV2Runtime,
) -> dict[str, Any]:
    with runtime.image_module.open(image_path) as opened_image:
        source_image = opened_image.convert("RGB")
        source_width, source_height = source_image.size

        model_image = _dolphin_resize(source_image)
        model_width, model_height = model_image.size

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": model_image,
                    },
                    {
                        "type": "text",
                        "text": TABLE_PROMPT,
                    },
                ],
            }
        ]

        text = runtime.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = (
            runtime.process_vision_info(messages)
        )

        inputs = runtime.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(runtime.model_device)

    prompt_tokens = int(inputs.input_ids.shape[-1])

    with runtime.torch.inference_mode():
        generated_ids = runtime.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=GENERATION_DO_SAMPLE,
            temperature=GENERATION_TEMPERATURE,
        )

    generated = generated_ids[
        0,
        inputs.input_ids.shape[1] :,
    ]

    raw_output = runtime.processor.decode(
        generated,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).strip()

    clean_output = runtime.processor.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    return {
        "raw_output": raw_output,
        "clean_output": clean_output,
        "source_image_size": [
            int(source_width),
            int(source_height),
        ],
        "model_input_image_size": [
            int(model_width),
            int(model_height),
        ],
        "prompt_tokens": prompt_tokens,
        "generated_tokens": int(generated.numel()),
    }


class DolphinV2Adapter:
    """Dolphin-v2 table reconstruction for canonical MinerU crops."""

    NAME = "dolphin-v2"
    DISPLAY_NAME = "Dolphin-v2"
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
        self._model_device = _dolphin_v2_device(device)
        self._runtime_loader = (
            runtime_loader or _default_runtime_loader
        )
        self._inference_runner = (
            inference_runner or _default_inference_runner
        )
        self._runtime: Any | None = None

    @property
    def capabilities(self) -> TableOCRCapabilities:
        return self._CAPABILITIES

    def _get_runtime(self) -> Any:
        if self._runtime is None:
            self._runtime = self._runtime_loader(
                self._model_device
            )

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

    def extract(
        self,
        table: TableOCRInput,
    ) -> TableOCRResult:
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
                error=f"Could not initialize Dolphin-v2: {exc}",
            )

        adapter_version = getattr(
            runtime,
            "transformers_version",
            None,
        )

        try:
            reconstruction = self._inference_runner(
                image_path,
                runtime,
            )
        except TableOCRDependencyError:
            raise
        except Exception as exc:
            return self._result(
                table,
                status="error",
                adapter_version=adapter_version,
                error=(
                    "Dolphin-v2 table reconstruction failed: "
                    f"{exc}"
                ),
            )

        raw_value = reconstruction.get("raw_output", "")
        clean_value = reconstruction.get("clean_output", "")

        raw_output = (
            raw_value
            if isinstance(raw_value, str)
            else str(raw_value or "")
        )
        clean_output = (
            clean_value
            if isinstance(clean_value, str)
            else str(clean_value or "")
        )

        html_tables = extract_html_tables(clean_output)

        native = {
            "dolphin_v2": {
                "model_repo": MODEL_REPO,
                "model_revision": MODEL_REVISION,
                "backbone_architecture": BACKBONE_ARCHITECTURE,
                "model_class": getattr(
                    runtime,
                    "model_class",
                    MODEL_CLASS,
                ),
                "prompt": TABLE_PROMPT,
                "model_dtype": MODEL_DTYPE,
                "resolved_model_dtype": getattr(
                    runtime,
                    "model_dtype",
                    None,
                ),
                "max_new_tokens": MAX_NEW_TOKENS,
                "generation_do_sample": GENERATION_DO_SAMPLE,
                "generation_temperature": GENERATION_TEMPERATURE,
                "execution_device": getattr(
                    runtime,
                    "model_device",
                    self._model_device,
                ),
                "transformers_version": adapter_version,
                "torch_version": getattr(
                    runtime,
                    "torch_version",
                    None,
                ),
                "qwen_vl_utils_version": getattr(
                    runtime,
                    "qwen_vl_utils_version",
                    None,
                ),
                "pillow_version": getattr(
                    runtime,
                    "pillow_version",
                    None,
                ),
                "source_image_size": reconstruction.get(
                    "source_image_size",
                    [],
                ),
                "model_input_image_size": reconstruction.get(
                    "model_input_image_size",
                    [],
                ),
                "prompt_tokens": reconstruction.get(
                    "prompt_tokens",
                    0,
                ),
                "generated_tokens": reconstruction.get(
                    "generated_tokens",
                    0,
                ),
                "raw_output": raw_output,
                "clean_output": clean_output,
                "native_format": "html",
                "special_tokens_removed_for_parsing": True,
                "normalization": "none",
                "parser": (
                    "tabulus.table_ocr.parsing:"
                    "parse_table_text"
                ),
                "html_tables_detected": len(html_tables),
                "input_policy": "canonical_mineru_crop",
                "image_preprocessing": {
                    "rgb_conversion": True,
                    "resize": "official_dolphin_resize_img",
                    "max_size": DOLPHIN_RESIZE_MAX_SIZE,
                    "min_size": DOLPHIN_RESIZE_MIN_SIZE,
                    "margin_crop": False,
                },
                "layout_redetection": False,
                "recropping": False,
            }
        }

        if not clean_output.strip():
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=[native],
                error=(
                    "Dolphin-v2 returned no generated table output."
                ),
            )

        if not html_tables:
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=[native],
                error=(
                    "Dolphin-v2 output did not contain "
                    "a usable HTML table."
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
