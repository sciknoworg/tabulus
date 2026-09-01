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
from tabulus.table_ocr.parsing import (
    extract_html_tables,
    parse_table_text,
)


VALIDATED_TRANSFORMERS_VERSION = "4.52.4"
VALIDATED_TOKENIZERS_VERSION = "0.21.4"
VALIDATED_FLASH_ATTN_VERSION = "2.7.3"

MODEL_REPO = "nanonets/Nanonets-OCR-s"
MODEL_REVISION = "3baad182cc87c65a1861f0c30357d3467e978172"
MODEL_VERSION = f"{MODEL_REPO}@{MODEL_REVISION}"

BACKBONE_ARCHITECTURE = "Qwen2.5-VL"
MODEL_CLASS = "Qwen2_5_VLForConditionalGeneration"
MODEL_DTYPE = "bfloat16"
ATTN_IMPLEMENTATION = "flash_attention_2"
PROCESSOR_USE_FAST = False

SYSTEM_PROMPT = "You are a helpful assistant."
TABLE_PROMPT = (
    "Extract the text from the above document as if you were reading it "
    "naturally. Return the tables in html format. Return the equations in "
    "LaTeX representation. If there is an image in the document and image "
    "caption is not present, add a small description of the image inside "
    "the <img></img> tag; otherwise, add the image caption inside "
    "<img></img>. Watermarks should be wrapped in brackets. Ex: "
    "<watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped "
    "in brackets.\nEx: <page_number>14</page_number> or "
    "<page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."
)

MAX_NEW_TOKENS = 15000
GENERATION_DO_SAMPLE = False


@dataclass(frozen=True)
class _NanonetsOCRSRuntime:
    torch: Any
    image_module: Any
    processor: Any
    model: Any
    model_device: str
    model_dtype: str
    model_class: str
    transformers_version: str | None
    tokenizers_version: str | None
    flash_attn_version: str | None
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


def _nanonets_ocr_s_device(device: str) -> str:
    normalized = device.strip().lower()

    if normalized == "gpu":
        return "cuda:0"

    if normalized.startswith("gpu:"):
        index = normalized.split(":", maxsplit=1)[1]
        if index.isdigit():
            return f"cuda:{index}"

    raise ValueError(
        "Use device='gpu' or 'gpu:<index>' for nanonets-ocr-s."
    )


def _dependency_error() -> TableOCRDependencyError:
    return TableOCRDependencyError(
        "The nanonets-ocr-s adapter requires "
        "transformers==4.52.4, tokenizers==0.21.4, "
        "flash-attn==2.7.3, PyTorch, torchvision, Pillow, "
        "and accelerate in the active environment."
    )


def _require_validated_version(
    package_name: str,
    expected: str,
) -> str:
    found = _installed_package_version(package_name)

    if found != expected:
        actual = found or "not installed"
        raise TableOCRDependencyError(
            "The nanonets-ocr-s adapter is validated against "
            f"{package_name}=={expected}; found {actual}."
        )

    return found


def _default_runtime_loader(
    model_device: str,
) -> _NanonetsOCRSRuntime:
    transformers_version = _require_validated_version(
        "transformers",
        VALIDATED_TRANSFORMERS_VERSION,
    )
    tokenizers_version = _require_validated_version(
        "tokenizers",
        VALIDATED_TOKENIZERS_VERSION,
    )
    flash_attn_version = _require_validated_version(
        "flash-attn",
        VALIDATED_FLASH_ATTN_VERSION,
    )

    try:
        torch = importlib.import_module("torch")
        image_module = importlib.import_module("PIL.Image")
        transformers = importlib.import_module("transformers")

        importlib.import_module("torchvision")
        importlib.import_module("flash_attn")
        importlib.import_module("accelerate")
    except ImportError as exc:
        raise _dependency_error() from exc

    if not bool(torch.cuda.is_available()):
        raise RuntimeError(
            "nanonets-ocr-s requires CUDA in the validated Tabulus "
            "configuration, but CUDA is not available."
        )

    processor = transformers.AutoProcessor.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        use_fast=PROCESSOR_USE_FAST,
    )

    model = transformers.AutoModelForImageTextToText.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map={"": model_device},
        attn_implementation=ATTN_IMPLEMENTATION,
    )
    model.eval()

    resolved_device = str(next(model.parameters()).device)

    if resolved_device != model_device:
        raise RuntimeError(
            "Nanonets-OCR-s loaded on an unexpected device: "
            f"requested {model_device}, resolved {resolved_device}."
        )

    return _NanonetsOCRSRuntime(
        torch=torch,
        image_module=image_module,
        processor=processor,
        model=model,
        model_device=model_device,
        model_dtype=str(next(model.parameters()).dtype),
        model_class=type(model).__name__,
        transformers_version=transformers_version,
        tokenizers_version=tokenizers_version,
        flash_attn_version=flash_attn_version,
        torch_version=_installed_package_version("torch"),
        torchvision_version=_installed_package_version(
            "torchvision"
        ),
        pillow_version=_installed_package_version("Pillow"),
    )


def _default_inference_runner(
    image_path: Path,
    runtime: _NanonetsOCRSRuntime,
) -> dict[str, Any]:
    with runtime.image_module.open(image_path) as opened_image:
        image = opened_image.convert("RGB")
        source_width, source_height = image.size

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"file://{image_path}",
                    },
                    {
                        "type": "text",
                        "text": TABLE_PROMPT,
                    },
                ],
            },
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
    prompt_tokens = int(inputs.input_ids.shape[-1])

    with runtime.torch.inference_mode():
        output_ids = runtime.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=GENERATION_DO_SAMPLE,
        )

    generated_ids = output_ids[
        0,
        inputs.input_ids.shape[-1] :,
    ]

    raw_output = runtime.processor.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).strip()

    clean_output = runtime.processor.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    return {
        "raw_output": raw_output,
        "clean_output": clean_output,
        "source_image_size": [
            int(source_width),
            int(source_height),
        ],
        "prompt_tokens": prompt_tokens,
        "generated_tokens": int(generated_ids.numel()),
        "raw_output_chars": len(raw_output),
        "clean_output_chars": len(clean_output),
    }


class NanonetsOCRSAdapter:
    """Nanonets-OCR-s reconstruction for canonical MinerU crops."""

    NAME = "nanonets-ocr-s"
    DISPLAY_NAME = "Nanonets-OCR-s"
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
        self._model_device = _nanonets_ocr_s_device(device)
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
                error=(
                    "Could not initialize Nanonets-OCR-s: "
                    f"{exc}"
                ),
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
                    "Nanonets-OCR-s table reconstruction failed: "
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

        parser_error: str | None = None

        try:
            parsed_tables = parse_table_text(clean_output)
        except Exception as exc:
            parsed_tables = []
            parser_error = str(exc)

        html_tables = extract_html_tables(clean_output)

        native = {
            "nanonets_ocr_s": {
                "model_repo": MODEL_REPO,
                "model_revision": MODEL_REVISION,
                "backbone_architecture": BACKBONE_ARCHITECTURE,
                "model_class": getattr(
                    runtime,
                    "model_class",
                    MODEL_CLASS,
                ),
                "system_prompt": SYSTEM_PROMPT,
                "prompt": TABLE_PROMPT,
                "model_dtype": MODEL_DTYPE,
                "resolved_model_dtype": getattr(
                    runtime,
                    "model_dtype",
                    None,
                ),
                "attention_implementation": ATTN_IMPLEMENTATION,
                "processor_use_fast": PROCESSOR_USE_FAST,
                "max_new_tokens": MAX_NEW_TOKENS,
                "generation_do_sample": GENERATION_DO_SAMPLE,
                "execution_device": getattr(
                    runtime,
                    "model_device",
                    self._model_device,
                ),
                "transformers_version": adapter_version,
                "tokenizers_version": getattr(
                    runtime,
                    "tokenizers_version",
                    None,
                ),
                "flash_attn_version": getattr(
                    runtime,
                    "flash_attn_version",
                    None,
                ),
                "torch_version": getattr(
                    runtime,
                    "torch_version",
                    None,
                ),
                "torchvision_version": getattr(
                    runtime,
                    "torchvision_version",
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
                "prompt_tokens": reconstruction.get(
                    "prompt_tokens",
                    0,
                ),
                "generated_tokens": reconstruction.get(
                    "generated_tokens",
                    0,
                ),
                "raw_output_chars": reconstruction.get(
                    "raw_output_chars",
                    len(raw_output),
                ),
                "clean_output_chars": reconstruction.get(
                    "clean_output_chars",
                    len(clean_output),
                ),
                "raw_output": raw_output,
                "clean_output": clean_output,
                "native_format": "nanonets_document_markup",
                "special_tokens_removed_for_parsing": True,
                "normalization": "none",
                "parser_input": "decoded_output_special_tokens_removed",
                "parser": (
                    "tabulus.table_ocr.parsing:"
                    "parse_table_text"
                ),
                "html_tables_detected": len(html_tables),
                "structured_tables_detected": len(parsed_tables),
                "parser_error": parser_error,
                "input_policy": "canonical_mineru_crop",
                "image_preprocessing": {
                    "external": "rgb_conversion_only",
                    "processor": "AutoProcessor",
                    "processor_use_fast": PROCESSOR_USE_FAST,
                    "model_internal_resize": True,
                },
                "layout_redetection": False,
                "recropping": False,
                "external_recropping": False,
            }
        }

        if not clean_output.strip():
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=[native],
                error=(
                    "Nanonets-OCR-s returned no generated "
                    "table output."
                ),
            )

        if not parsed_tables:
            error = (
                "Nanonets-OCR-s output did not contain a "
                "usable structured table."
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
            result_count=len(parsed_tables),
            native_json=[native],
            native_markdown=[clean_output],
        )
