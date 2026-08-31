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
from tabulus.table_ocr.parsing import otsl_table_to_html


VALIDATED_TRANSFORMERS_VERSION = "5.16.1"

MODEL_REPO = "opendatalab/TRivia-3B"
MODEL_REVISION = "fcf890f3869afaa9fc768a14e72ab1ff46bfc813"

TABLE_PROMPT = (
    "You are an AI specialized in recognizing and extracting table from images. "
    "Your mission is to analyze the table image and generate the result in OTSL "
    "format using specified tags. Output only the results without any other words "
    "and explanation."
)

MODEL_DTYPE = "bfloat16"
MAX_NEW_TOKENS = 8192
REPETITION_PENALTY = 1.05

MODEL_VERSION = f"{MODEL_REPO}@{MODEL_REVISION}"


@dataclass(frozen=True)
class _TRiviaRuntime:
    torch: Any
    image_module: Any
    processor: Any
    model: Any
    model_device: str
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


def _trivia_device(device: str) -> str:
    normalized = device.strip().lower()

    if normalized == "gpu":
        return "cuda"

    if normalized.startswith("gpu:"):
        index = normalized.split(":", maxsplit=1)[1]
        if index.isdigit():
            return f"cuda:{index}"

    raise ValueError(
        "Use device='gpu' or 'gpu:<index>' for trivia."
    )


def _dependency_error() -> TableOCRDependencyError:
    return TableOCRDependencyError(
        "The trivia adapter requires transformers==5.16.1, "
        "PyTorch, Accelerate, and Pillow in the active environment."
    )


def _default_runtime_loader(
    model_device: str,
) -> _TRiviaRuntime:
    transformers_version = _installed_package_version("transformers")

    if transformers_version != VALIDATED_TRANSFORMERS_VERSION:
        found = transformers_version or "not installed"
        raise TableOCRDependencyError(
            "The trivia adapter is validated against "
            f"transformers=={VALIDATED_TRANSFORMERS_VERSION}; "
            f"found {found}."
        )

    try:
        torch = importlib.import_module("torch")
        image_module = importlib.import_module("PIL.Image")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise _dependency_error() from exc

    if not bool(torch.cuda.is_available()):
        raise RuntimeError(
            "trivia requires CUDA in the validated Tabulus "
            "configuration, but CUDA is not available."
        )

    processor = transformers.AutoProcessor.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
    )

    model = transformers.AutoModelForMultimodalLM.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        device_map=model_device,
    )
    model.eval()

    return _TRiviaRuntime(
        torch=torch,
        image_module=image_module,
        processor=processor,
        model=model,
        model_device=model_device,
        transformers_version=transformers_version,
        torch_version=_installed_package_version("torch"),
        accelerate_version=_installed_package_version("accelerate"),
    )


def _default_inference_runner(
    image_path: Path,
    runtime: _TRiviaRuntime,
) -> dict[str, Any]:
    with runtime.image_module.open(image_path) as source_image:
        image = source_image.convert("RGB")

    width, height = image.size

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
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
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(runtime.model_device)

    prompt_tokens = int(inputs["input_ids"].shape[-1])

    with runtime.torch.inference_mode():
        output_ids = runtime.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=REPETITION_PENALTY,
            use_cache=True,
        )

    generated = output_ids[0, prompt_tokens:]

    raw_output = runtime.processor.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    return {
        "raw_output": raw_output,
        "image_size": [int(width), int(height)],
        "prompt_tokens": prompt_tokens,
        "generated_tokens": int(generated.numel()),
    }


class TRiviaAdapter:
    """TRivia-3B table reconstruction for canonical MinerU crops."""

    NAME = "trivia"
    DISPLAY_NAME = "TRivia-3B"
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
        self._model_device = _trivia_device(device)
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
                error=f"Could not initialize TRivia-3B: {exc}",
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
                error=f"TRivia-3B table reconstruction failed: {exc}",
            )

        raw_output = str(
            reconstruction.get("raw_output", "")
        )

        native = {
            "trivia": {
                "model_repo": MODEL_REPO,
                "model_revision": MODEL_REVISION,
                "prompt": TABLE_PROMPT,
                "dtype": MODEL_DTYPE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": False,
                "repetition_penalty": REPETITION_PENALTY,
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
                "raw_otsl": raw_output,
                "normalization": (
                    "tabulus.table_ocr.parsing:"
                    "otsl_table_to_html"
                ),
            }
        }

        if not raw_output.strip():
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=[native],
                error="TRivia-3B returned no generated OTSL output.",
            )

        html_table = otsl_table_to_html(raw_output)

        if not html_table:
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=[native],
                error=(
                    "TRivia-3B output did not contain a usable "
                    "OTSL table."
                ),
            )

        return self._result(
            table,
            status="ok",
            adapter_version=adapter_version,
            result_count=1,
            native_json=[native],
            native_markdown=[html_table],
        )
