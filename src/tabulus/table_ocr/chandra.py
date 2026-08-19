from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

from tabulus.table_ocr.base import (
    TableOCRCapabilities,
    TableOCRDependencyError,
    TableOCRInput,
    TableOCRResult,
)


ModelLoader = Callable[[str], Any]
Generator = Callable[[list[Any], Any], list[Any]]
InputFactory = Callable[..., Any]
ImageLoader = Callable[[Path], Any]


def _installed_package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _torch_device(device: str) -> str:
    """Translate Tabulus device names to PyTorch device names."""

    normalized = device.strip().lower()

    if normalized == "gpu":
        return "cuda"

    if normalized.startswith("gpu:"):
        return f"cuda:{normalized.split(':', maxsplit=1)[1]}"

    if normalized.startswith("cpu"):
        return "cpu"

    raise ValueError(
        f"Unsupported Chandra device {device!r}. Use cpu, gpu, or gpu:<index>."
    )


def _dependency_error(exc: ImportError) -> TableOCRDependencyError:
    return TableOCRDependencyError(
        "The chandra adapter requires the Hugging Face Chandra runtime. "
        'Install it in a dedicated environment with '
        '`pip install "chandra-ocr[hf]==0.2.0"`.'
    )


def _default_model_loader(torch_device: str) -> Any:
    try:
        hf = importlib.import_module("chandra.model.hf")
        settings_module = importlib.import_module("chandra.settings")
    except ImportError as exc:
        raise _dependency_error(exc) from exc

    settings = settings_module.settings
    previous_device = settings.TORCH_DEVICE
    settings.TORCH_DEVICE = torch_device

    try:
        return hf.load_model()
    except ImportError as exc:
        raise _dependency_error(exc) from exc
    finally:
        settings.TORCH_DEVICE = previous_device


def _default_generator(batch: list[Any], model: Any) -> list[Any]:
    try:
        hf = importlib.import_module("chandra.model.hf")
    except ImportError as exc:
        raise _dependency_error(exc) from exc

    return list(hf.generate_hf(batch, model))


def _default_input_factory(**kwargs: Any) -> Any:
    try:
        schema = importlib.import_module("chandra.model.schema")
    except ImportError as exc:
        raise _dependency_error(exc) from exc

    return schema.BatchInputItem(**kwargs)


def _default_image_loader(path: Path) -> Any:
    try:
        image_module = importlib.import_module("PIL.Image")
    except ImportError as exc:
        raise _dependency_error(exc) from exc

    with image_module.open(path) as image:
        return image.convert("RGB")


class ChandraAdapter:
    """Chandra OCR 2 reconstruction for MinerU-generated table crops."""

    NAME = "chandra"
    DISPLAY_NAME = "Chandra OCR 2"
    MODEL_VERSION = "datalab-to/chandra-ocr-2"
    PROMPT_TYPE = "ocr"

    _CAPABILITIES = TableOCRCapabilities(
        name=NAME,
        display_name=DISPLAY_NAME,
        cpu_supported=True,
        gpu_supported=True,
    )

    def __init__(
        self,
        *,
        device: str = "cpu",
        model_loader: ModelLoader | None = None,
        generator: Generator | None = None,
        input_factory: InputFactory | None = None,
        image_loader: ImageLoader | None = None,
    ) -> None:
        self.device = device
        self._torch_device = _torch_device(device)
        self._model_loader = model_loader or _default_model_loader
        self._generator = generator or _default_generator
        self._input_factory = input_factory or _default_input_factory
        self._image_loader = image_loader or _default_image_loader
        self._model: Any | None = None

    @property
    def capabilities(self) -> TableOCRCapabilities:
        return self._CAPABILITIES

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = self._model_loader(self._torch_device)

        return self._model

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
            adapter_version=_installed_package_version("chandra-ocr"),
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
            model = self._get_model()
        except TableOCRDependencyError:
            raise
        except Exception as exc:
            return self._result(
                table,
                status="error",
                error=f"Could not initialize Chandra OCR 2: {exc}",
            )

        try:
            image = self._image_loader(image_path)
            batch_item = self._input_factory(
                image=image,
                prompt_type=self.PROMPT_TYPE,
            )
            results = list(self._generator([batch_item], model))
        except TableOCRDependencyError:
            raise
        except Exception as exc:
            return self._result(
                table,
                status="error",
                error=f"Chandra OCR 2 inference failed: {exc}",
            )

        if not results:
            return self._result(
                table,
                status="empty",
                error="Chandra OCR 2 returned no generation results.",
            )

        native_json = [
            {
                "raw": getattr(result, "raw", ""),
                "token_count": getattr(result, "token_count", 0),
                "error": bool(getattr(result, "error", False)),
            }
            for result in results
        ]
        native_markdown = [
            getattr(result, "raw", "")
            for result in results
        ]

        if len(results) != 1:
            return self._result(
                table,
                status="error",
                result_count=len(results),
                native_json=native_json,
                native_markdown=native_markdown,
                error=(
                    "Chandra OCR 2 returned multiple generation results "
                    "for one canonical table crop."
                ),
            )

        result = results[0]

        if bool(getattr(result, "error", False)):
            return self._result(
                table,
                status="error",
                result_count=1,
                native_json=native_json,
                native_markdown=native_markdown,
                error="Chandra OCR 2 reported an inference error.",
            )

        raw = getattr(result, "raw", "")

        if not isinstance(raw, str) or not raw.strip():
            return self._result(
                table,
                status="empty",
                result_count=1,
                native_json=native_json,
                native_markdown=native_markdown,
                error="Chandra OCR 2 returned empty generated content.",
            )

        return self._result(
            table,
            status="ok",
            result_count=1,
            native_json=native_json,
            native_markdown=native_markdown,
        )
