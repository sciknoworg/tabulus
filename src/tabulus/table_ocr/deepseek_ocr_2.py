from __future__ import annotations

import importlib
import tempfile
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


VALIDATED_TRANSFORMERS_VERSION = "4.46.3"
VALIDATED_TOKENIZERS_VERSION = "0.20.3"
VALIDATED_FLASH_ATTN_VERSION = "2.7.3"

MODEL_REPO = "deepseek-ai/DeepSeek-OCR-2"
MODEL_REVISION = "aaa02f3811945a91062062994c5c4a3f4c0af2b0"
MODEL_VERSION = f"{MODEL_REPO}@{MODEL_REVISION}"

MODEL_CLASS = "DeepseekOCR2ForCausalLM"
TABLE_PROMPT = (
    "<image>\n"
    "<|grounding|>Convert the document to markdown."
)
MODEL_DTYPE = "bfloat16"
ATTN_IMPLEMENTATION = "flash_attention_2"

BASE_SIZE = 1024
IMAGE_SIZE = 768
CROP_MODE = True
SAVE_RESULTS = False
EVAL_MODE = True

MAX_NEW_TOKENS = 8192
GENERATION_DO_SAMPLE = False
GENERATION_TEMPERATURE = 0.0
NO_REPEAT_NGRAM_SIZE = 35
GENERATION_USE_CACHE = True


@dataclass(frozen=True)
class _DeepSeekOCR2Runtime:
    torch: Any
    image_module: Any
    tokenizer: Any
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


def _deepseek_ocr_2_device(device: str) -> str:
    normalized = device.strip().lower()

    if normalized == "gpu":
        return "cuda"

    if normalized.startswith("gpu:"):
        index = normalized.split(":", maxsplit=1)[1]
        if index.isdigit():
            return f"cuda:{index}"

    raise ValueError(
        "Use device='gpu' or 'gpu:<index>' for deepseek-ocr-2."
    )


def _dependency_error() -> TableOCRDependencyError:
    return TableOCRDependencyError(
        "The deepseek-ocr-2 adapter requires "
        "transformers==4.46.3, tokenizers==0.20.3, "
        "flash-attn==2.7.3, PyTorch, torchvision, "
        "Pillow, einops, addict, and easydict in the "
        "active environment."
    )


def _require_validated_version(
    package_name: str,
    expected: str,
) -> str:
    found = _installed_package_version(package_name)

    if found != expected:
        actual = found or "not installed"
        raise TableOCRDependencyError(
            "The deepseek-ocr-2 adapter is validated against "
            f"{package_name}=={expected}; found {actual}."
        )

    return found


def _default_runtime_loader(
    model_device: str,
) -> _DeepSeekOCR2Runtime:
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

        # Import the validated optional dependencies here so failures are
        # reported at adapter initialization rather than from remote model
        # code later in the load path.
        importlib.import_module("torchvision")
        importlib.import_module("flash_attn")
        importlib.import_module("einops")
        importlib.import_module("addict")
        importlib.import_module("easydict")
    except ImportError as exc:
        raise _dependency_error() from exc

    if not bool(torch.cuda.is_available()):
        raise RuntimeError(
            "deepseek-ocr-2 requires CUDA in the validated Tabulus "
            "configuration, but CUDA is not available."
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )

    model = transformers.AutoModel.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        use_safetensors=True,
        _attn_implementation=ATTN_IMPLEMENTATION,
    )

    if not callable(getattr(model, "infer", None)):
        raise RuntimeError(
            "Loaded DeepSeek-OCR-2 model does not expose the "
            "validated infer() method."
        )

    model = model.eval().to(model_device).to(torch.bfloat16)

    return _DeepSeekOCR2Runtime(
        torch=torch,
        image_module=image_module,
        tokenizer=tokenizer,
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
    runtime: _DeepSeekOCR2Runtime,
) -> dict[str, Any]:
    with runtime.image_module.open(image_path) as opened_image:
        source_width, source_height = opened_image.size

    with tempfile.TemporaryDirectory(
        prefix="tabulus-deepseek-ocr-2-"
    ) as scratch_dir:
        with runtime.torch.cuda.device(runtime.model_device):
            output = runtime.model.infer(
                runtime.tokenizer,
                prompt=TABLE_PROMPT,
                image_file=str(image_path),
                output_path=scratch_dir,
                base_size=BASE_SIZE,
                image_size=IMAGE_SIZE,
                crop_mode=CROP_MODE,
                save_results=SAVE_RESULTS,
                eval_mode=EVAL_MODE,
            )

    if not isinstance(output, str):
        output = str(output or "")

    decoded_output_tokens = len(
        runtime.tokenizer.encode(
            output,
            add_special_tokens=False,
        )
    )

    return {
        "raw_output": output,
        "clean_output": output,
        "source_image_size": [
            int(source_width),
            int(source_height),
        ],
        "decoded_output_tokens": int(decoded_output_tokens),
        "output_chars": len(output),
    }


class DeepSeekOCR2Adapter:
    """DeepSeek-OCR-2 reconstruction for canonical MinerU crops."""

    NAME = "deepseek-ocr-2"
    DISPLAY_NAME = "DeepSeek-OCR-2"
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
        self._model_device = _deepseek_ocr_2_device(device)
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
                    "Could not initialize DeepSeek-OCR-2: "
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
                    "DeepSeek-OCR-2 table reconstruction failed: "
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
            "deepseek_ocr_2": {
                "model_repo": MODEL_REPO,
                "model_revision": MODEL_REVISION,
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
                "attention_implementation": ATTN_IMPLEMENTATION,
                "max_new_tokens": MAX_NEW_TOKENS,
                "generation_do_sample": GENERATION_DO_SAMPLE,
                "generation_temperature": (
                    GENERATION_TEMPERATURE
                ),
                "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
                "generation_use_cache": GENERATION_USE_CACHE,
                "official_infer_eval_mode": EVAL_MODE,
                "official_infer_save_results": SAVE_RESULTS,
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
                "decoded_output_tokens": reconstruction.get(
                    "decoded_output_tokens",
                    0,
                ),
                "output_chars": reconstruction.get(
                    "output_chars",
                    len(clean_output),
                ),
                "raw_output": raw_output,
                "clean_output": clean_output,
                "native_format": (
                    "deepseek_document_markdown_with_grounding"
                ),
                "normalization": "none",
                "parser_input": "model_infer_output_unchanged",
                "parser": (
                    "tabulus.table_ocr.parsing:"
                    "parse_table_text"
                ),
                "html_tables_detected": len(html_tables),
                "structured_tables_detected": len(parsed_tables),
                "parser_error": parser_error,
                "input_policy": "canonical_mineru_crop",
                "image_preprocessing": {
                    "external": "none",
                    "model_internal": (
                        "deepseek_ocr_2_dynamic_resolution"
                    ),
                    "base_size": BASE_SIZE,
                    "image_size": IMAGE_SIZE,
                    "crop_mode": CROP_MODE,
                    "model_internal_tiling": True,
                },
                "layout_redetection": False,
                "recropping": False,
                "external_recropping": False,
                "trust_remote_code": True,
            }
        }

        if not clean_output.strip():
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=[native],
                error=(
                    "DeepSeek-OCR-2 returned no generated "
                    "table output."
                ),
            )

        if not parsed_tables:
            error = (
                "DeepSeek-OCR-2 output did not contain a "
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
