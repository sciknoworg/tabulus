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


MODEL_ID = "numind/NuExtract3"
MODE = "markdown"
ENABLE_THINKING = False
MAX_NEW_TOKENS = 8192


@dataclass(frozen=True)
class _NuExtractRuntime:
    torch: Any
    image_module: Any
    processor: Any
    model: Any


RuntimeLoader = Callable[[str], Any]
InferenceRunner = Callable[[Path, Any], str]


def _installed_package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _torch_device(device: str) -> str:
    """Translate the validated Tabulus GPU device names to PyTorch devices."""

    normalized = device.strip().lower()

    if normalized == "gpu":
        return "cuda"

    if normalized.startswith("gpu:"):
        index = normalized.split(":", maxsplit=1)[1]
        if index.isdigit():
            return f"cuda:{index}"

    raise ValueError(
        "NuExtract3 is GPU-only in the validated Tabulus configuration. "
        "Use gpu or gpu:<index>."
    )


def _dependency_error(exc: ImportError) -> TableOCRDependencyError:
    return TableOCRDependencyError(
        "The nuextract3 adapter requires PyTorch, Transformers, Accelerate, "
        "and Pillow in a dedicated model environment."
    )


def _default_runtime_loader(torch_device: str) -> _NuExtractRuntime:
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        importlib.import_module("accelerate")
        image_module = importlib.import_module("PIL.Image")
    except ImportError as exc:
        raise _dependency_error(exc) from exc

    if not bool(torch.cuda.is_available()):
        raise RuntimeError(
            "NuExtract3 requires a visible CUDA GPU in the validated Tabulus "
            "configuration."
        )

    processor = transformers.AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map={"": torch_device},
        trust_remote_code=True,
    ).eval()

    return _NuExtractRuntime(
        torch=torch,
        image_module=image_module,
        processor=processor,
        model=model,
    )


def _default_inference_runner(
    image_path: Path,
    runtime: _NuExtractRuntime,
) -> str:
    with runtime.image_module.open(image_path) as source_image:
        image = source_image.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                }
            ],
        }
    ]

    inputs = runtime.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        mode=MODE,
        enable_thinking=ENABLE_THINKING,
    ).to(runtime.model.device)

    prompt_length = inputs["input_ids"].shape[1]

    with runtime.torch.inference_mode():
        generated_ids = runtime.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    answer_ids = generated_ids[:, prompt_length:]
    decoded = runtime.processor.batch_decode(
        answer_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    if not decoded:
        return ""

    return str(decoded[0]).strip()


class NuExtract3Adapter:
    """NuExtract3 document-VLM reconstruction for MinerU table crops."""

    NAME = "nuextract3"
    DISPLAY_NAME = "NuExtract3"
    MODEL_VERSION = MODEL_ID

    _CAPABILITIES = TableOCRCapabilities(
        name=NAME,
        display_name=DISPLAY_NAME,
        cpu_supported=False,
        gpu_supported=True,
    )

    def __init__(
        self,
        *,
        device: str = "gpu",
        runtime_loader: RuntimeLoader | None = None,
        inference_runner: InferenceRunner | None = None,
    ) -> None:
        self.device = device
        self._torch_device = _torch_device(device)
        self._runtime_loader = runtime_loader or _default_runtime_loader
        self._inference_runner = inference_runner or _default_inference_runner
        self._runtime: Any | None = None

    @property
    def capabilities(self) -> TableOCRCapabilities:
        return self._CAPABILITIES

    def _get_runtime(self) -> Any:
        if self._runtime is None:
            self._runtime = self._runtime_loader(self._torch_device)

        return self._runtime

    def _result(
        self,
        table: TableOCRInput,
        *,
        status: str,
        result_count: int = 0,
        native_json: list[Any] | None = None,
        native_markdown: list[Any] | None = None,
        error: str | None = None,
    ) -> TableOCRResult:
        return TableOCRResult(
            table_id=table.table_id,
            adapter_name=self.NAME,
            adapter_version=_installed_package_version("transformers"),
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
                error=f"Could not initialize NuExtract3: {exc}",
            )

        try:
            raw = self._inference_runner(image_path, runtime)
        except TableOCRDependencyError:
            raise
        except Exception as exc:
            return self._result(
                table,
                status="error",
                error=f"NuExtract3 inference failed: {exc}",
            )

        if not isinstance(raw, str) or not raw.strip():
            return self._result(
                table,
                status="empty",
                result_count=1,
                error="NuExtract3 returned empty generated content.",
            )

        raw = raw.strip()
        native_json = [
            {
                "raw": raw,
                "mode": MODE,
                "enable_thinking": ENABLE_THINKING,
                "max_new_tokens": MAX_NEW_TOKENS,
            }
        ]

        return self._result(
            table,
            status="ok",
            result_count=1,
            native_json=native_json,
            native_markdown=[raw],
        )
