from __future__ import annotations

import importlib
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

from tabulus.table_ocr.base import (
    TableOCRCapabilities,
    TableOCRDependencyError,
    TableOCRInput,
    TableOCRResult,
)


PipelineFactory = Callable[..., Any]


def _installed_package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _default_pipeline_factory(**kwargs: Any) -> Any:
    try:
        paddleocr = importlib.import_module("paddleocr")
    except ImportError as exc:
        raise TableOCRDependencyError(
            "The paddleocr-vl adapter requires PaddleOCR and PaddlePaddle. "
            "Install them in a dedicated PaddleOCR environment."
        ) from exc

    paddleocr_vl = getattr(paddleocr, "PaddleOCRVL", None)

    if paddleocr_vl is None:
        raise TableOCRDependencyError(
            "The installed PaddleOCR package does not expose PaddleOCRVL. "
            "Install a PaddleOCR version that includes PaddleOCR-VL support."
        )

    return paddleocr_vl(**kwargs)


def _to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): _to_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item) for item in value]

    tolist = getattr(value, "tolist", None)

    if callable(tolist):
        return _to_json_safe(tolist())

    item = getattr(value, "item", None)

    if callable(item):
        try:
            return _to_json_safe(item())
        except ValueError:
            pass

    return repr(value)


def _read_public_result_attribute(result: Any, name: str) -> Any:
    value = getattr(result, name, None)

    if callable(value):
        value = value()

    return _to_json_safe(value)


class PaddleOCRVLAdapter:
    """PaddleOCR-VL table reconstruction for MinerU-generated table crops."""

    NAME = "paddleocr-vl"
    DISPLAY_NAME = "PaddleOCR-VL"
    PIPELINE_VERSION = "v1.6"

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
        engine: str = "paddle",
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self.device = device
        self.engine = engine
        self._pipeline_factory = (
            pipeline_factory
            if pipeline_factory is not None
            else _default_pipeline_factory
        )
        self._pipeline: Any | None = None

    @property
    def capabilities(self) -> TableOCRCapabilities:
        return self._CAPABILITIES

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            self._pipeline = self._pipeline_factory(
                pipeline_version=self.PIPELINE_VERSION,
                device=self.device,
                engine=self.engine,
                use_layout_detection=False,
            )

        return self._pipeline

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
            adapter_version=_installed_package_version("paddleocr"),
            model_version=f"PaddleOCR-VL {self.PIPELINE_VERSION}",
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
            pipeline = self._get_pipeline()
        except TableOCRDependencyError:
            raise
        except Exception as exc:
            return self._result(
                table,
                status="error",
                error=f"Could not initialize PaddleOCR-VL: {exc}",
            )

        try:
            output = pipeline.predict(
                str(image_path),
                use_layout_detection=False,
                prompt_label="table",
            )
            results = [] if output is None else list(output)
        except Exception as exc:
            return self._result(
                table,
                status="error",
                error=f"PaddleOCR-VL inference failed: {exc}",
            )

        if not results:
            return self._result(
                table,
                status="empty",
                error="PaddleOCR-VL returned no result objects.",
            )

        try:
            native_json = [
                _read_public_result_attribute(result, "json")
                for result in results
            ]
            native_markdown = [
                _read_public_result_attribute(result, "markdown")
                for result in results
            ]
        except Exception as exc:
            return self._result(
                table,
                status="error",
                result_count=len(results),
                error=f"Could not read PaddleOCR-VL result data: {exc}",
            )

        has_content = any(
            value not in (None, "", [], {})
            for value in [*native_json, *native_markdown]
        )

        return self._result(
            table,
            status="ok" if has_content else "empty",
            result_count=len(results),
            native_json=native_json,
            native_markdown=native_markdown,
            error=None if has_content else "PaddleOCR-VL returned empty result data.",
        )
