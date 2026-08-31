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


VALIDATED_TRANSFORMERS_VERSION = "5.16.1"

MODEL_REPO = "zai-org/GLM-OCR"
MODEL_REVISION = "ca5d8b3e287e52589e37c28385d9655ee4372f9d"

TABLE_PROMPT = "Table Recognition:"
MODEL_LOAD_DTYPE = "auto"
MAX_NEW_TOKENS = 8192

MODEL_VERSION = f"{MODEL_REPO}@{MODEL_REVISION}"


@dataclass(frozen=True)
class _GLMOCRRuntime:
    torch: Any
    image_module: Any
    processor: Any
    model: Any
    model_device: str
    model_dtype: str
    transformers_version: str | None
    torch_version: str | None
    accelerate_version: str | None


RuntimeLoader = Callable[[str], Any]
InferenceRunner = Callable[[Path, Any], dict[str, Any]]


def _installed_package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _glm_ocr_device(device: str) -> str:
    normalized = device.strip().lower()

    if normalized == "gpu":
        return "cuda"

    if normalized.startswith("gpu:"):
        index = normalized.split(":", maxsplit=1)[1]
        if index.isdigit():
            return f"cuda:{index}"

    raise ValueError(
        "Use device='gpu' or 'gpu:<index>' for glm-ocr."
    )


def _dependency_error() -> TableOCRDependencyError:
    return TableOCRDependencyError(
        "The glm-ocr adapter requires transformers==5.16.1, "
        "PyTorch, Accelerate, and Pillow in the active environment."
    )


def _default_runtime_loader(
    model_device: str,
) -> _GLMOCRRuntime:
    transformers_version = _installed_package_version("transformers")

    if transformers_version != VALIDATED_TRANSFORMERS_VERSION:
        found = transformers_version or "not installed"
        raise TableOCRDependencyError(
            "The glm-ocr adapter is validated against "
            f"transformers=={VALIDATED_TRANSFORMERS_VERSION}; "
            f"found {found}."
        )

    accelerate_version = _installed_package_version("accelerate")
    if accelerate_version is None:
        raise _dependency_error()

    try:
        torch = importlib.import_module("torch")
        image_module = importlib.import_module("PIL.Image")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise _dependency_error() from exc

    if not bool(torch.cuda.is_available()):
        raise RuntimeError(
            "glm-ocr requires CUDA in the validated Tabulus "
            "configuration, but CUDA is not available."
        )

    processor = transformers.AutoProcessor.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
    )

    model = transformers.AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path=MODEL_REPO,
        revision=MODEL_REVISION,
        torch_dtype=MODEL_LOAD_DTYPE,
        device_map=model_device,
    )
    model.eval()

    return _GLMOCRRuntime(
        torch=torch,
        image_module=image_module,
        processor=processor,
        model=model,
        model_device=model_device,
        model_dtype=str(getattr(model, "dtype", "unknown")),
        transformers_version=transformers_version,
        torch_version=_installed_package_version("torch"),
        accelerate_version=accelerate_version,
    )


def _default_inference_runner(
    image_path: Path,
    runtime: _GLMOCRRuntime,
) -> dict[str, Any]:
    with runtime.image_module.open(image_path) as source_image:
        width, height = source_image.size

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": str(image_path),
                },
                {
                    "type": "text",
                    "text": TABLE_PROMPT,
                },
            ],
        }
    ]

    inputs = runtime.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(runtime.model_device)

    # GLM-OCR's official direct Transformers example removes this field
    # before generation.
    inputs.pop("token_type_ids", None)

    prompt_tokens = int(inputs["input_ids"].shape[-1])

    with runtime.torch.inference_mode():
        output_ids = runtime.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    generated = output_ids[0, prompt_tokens:]

    raw_output = runtime.processor.decode(
        generated,
        skip_special_tokens=False,
    ).strip()

    clean_output = runtime.processor.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    return {
        "raw_output": raw_output,
        "clean_output": clean_output,
        "image_size": [int(width), int(height)],
        "prompt_tokens": prompt_tokens,
        "generated_tokens": int(generated.numel()),
    }


class GLMOCRAdapter:
    """GLM-OCR table reconstruction for canonical MinerU crops."""

    NAME = "glm-ocr"
    DISPLAY_NAME = "GLM-OCR"
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
        self._model_device = _glm_ocr_device(device)
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
                error=f"Could not initialize GLM-OCR: {exc}",
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
                error=f"GLM-OCR table reconstruction failed: {exc}",
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
            "glm_ocr": {
                "model_repo": MODEL_REPO,
                "model_revision": MODEL_REVISION,
                "prompt": TABLE_PROMPT,
                "model_load_dtype": MODEL_LOAD_DTYPE,
                "resolved_model_dtype": getattr(
                    runtime,
                    "model_dtype",
                    None,
                ),
                "max_new_tokens": MAX_NEW_TOKENS,
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
                "accelerate_version": getattr(
                    runtime,
                    "accelerate_version",
                    None,
                ),
                "image_size": reconstruction.get(
                    "image_size",
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
                error="GLM-OCR returned no generated table output.",
            )

        if not html_tables:
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=[native],
                error=(
                    "GLM-OCR output did not contain a usable "
                    "HTML table."
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
