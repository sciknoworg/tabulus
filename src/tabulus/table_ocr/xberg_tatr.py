from __future__ import annotations

import asyncio
import importlib
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

from tabulus.table_ocr.base import (
    TableOCRCapabilities,
    TableOCRDependencyError,
    TableOCRInput,
    TableOCRResult,
)


ConfigFactory = Callable[[], Any]
InputFactory = Callable[..., Any]
Extractor = Callable[[Any, Any], Any]


def _installed_package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _dependency_error(exc: ImportError) -> TableOCRDependencyError:
    return TableOCRDependencyError(
        "The xberg-tatr adapter requires Xberg. Install it in a dedicated "
        "environment with `pip install xberg==1.0.14` and make sure the "
        "Tesseract executable is available in that environment."
    )


def _load_xberg() -> Any:
    try:
        return importlib.import_module("xberg")
    except ImportError as exc:
        raise _dependency_error(exc) from exc


def _default_config_factory() -> Any:
    xberg = _load_xberg()

    return xberg.ExtractionConfig(
        use_cache=False,
        ocr=xberg.OcrConfig(
            backend="tesseract",
            language=["eng"],
            tesseract_config=xberg.TesseractConfig(
                enable_table_detection=True,
            ),
        ),
        force_ocr=True,
        layout=xberg.LayoutDetectionConfig(
            strategy="always",
            apply_heuristics=True,
            table_model="tatr",
        ),
        pages=xberg.PageConfig(extract_pages=True),
    )


def _default_input_factory(**kwargs: Any) -> Any:
    xberg = _load_xberg()
    return xberg.ExtractInput(**kwargs)


def _default_extractor(extract_input: Any, config: Any) -> Any:
    xberg = _load_xberg()

    async def run() -> Any:
        return await xberg.extract(extract_input, config)

    return asyncio.run(run())


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

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _to_json_safe(to_dict())
        except (TypeError, ValueError):
            pass

    return repr(value)


def _read_public_attribute(value: Any, name: str, default: Any = None) -> Any:
    attribute = getattr(value, name, default)

    if callable(attribute):
        attribute = attribute()

    return attribute


def _normalized_cells(value: Any) -> list[list[str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []

    rows: list[list[str]] = []

    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            return []

        rows.append([
            "" if cell is None else str(cell)
            for cell in row
        ])

    return rows


def _markdown_from_cells(cells: list[list[str]]) -> str:
    if not cells:
        return ""

    max_cols = max((len(row) for row in cells), default=0)
    if max_cols == 0:
        return ""

    rows = [
        row + [""] * (max_cols - len(row))
        for row in cells
    ]

    def clean(cell: str) -> str:
        return " ".join(cell.replace("|", " ").split())

    def line(row: list[str]) -> str:
        return "| " + " | ".join(clean(cell) for cell in row) + " |"

    header = line(rows[0])
    separator = "| " + " | ".join("---" for _ in range(max_cols)) + " |"
    body = [line(row) for row in rows[1:]]

    return "\n".join([header, separator, *body])


def _table_markdown(table: Any) -> str:
    markdown = _read_public_attribute(table, "markdown", "")

    if isinstance(markdown, str) and markdown.strip():
        return markdown.strip()

    cells = _normalized_cells(
        _read_public_attribute(table, "cells", [])
    )
    return _markdown_from_cells(cells)


def _serialize_table(table: Any) -> dict[str, Any]:
    cells = _normalized_cells(
        _read_public_attribute(table, "cells", [])
    )

    return {
        "page_number": _to_json_safe(
            _read_public_attribute(table, "page_number")
        ),
        "markdown": _table_markdown(table),
        "cells": cells,
    }


class XbergTATRAdapter:
    """Xberg/TATR reconstruction for canonical MinerU table crops."""

    NAME = "xberg-tatr"
    DISPLAY_NAME = "Xberg/TATR (Tesseract OCR)"
    MODEL_VERSION = "RT-DETR v2 + TATR"

    _CAPABILITIES = TableOCRCapabilities(
        name=NAME,
        display_name=DISPLAY_NAME,
        cpu_supported=True,
        gpu_supported=False,
    )

    def __init__(
        self,
        *,
        device: str = "cpu",
        config_factory: ConfigFactory | None = None,
        input_factory: InputFactory | None = None,
        extractor: Extractor | None = None,
    ) -> None:
        if not device.strip().lower().startswith("cpu"):
            raise ValueError(
                "The validated Xberg/TATR configuration is CPU-only. "
                "Use device='cpu'."
            )

        self.device = device
        self._config_factory = config_factory or _default_config_factory
        self._input_factory = input_factory or _default_input_factory
        self._extractor = extractor or _default_extractor
        self._config: Any | None = None

    @property
    def capabilities(self) -> TableOCRCapabilities:
        return self._CAPABILITIES

    def _get_config(self) -> Any:
        if self._config is None:
            self._config = self._config_factory()

        return self._config

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
            adapter_version=_installed_package_version("xberg"),
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
            config = self._get_config()
            extract_input = self._input_factory(
                kind="uri",
                uri=str(image_path),
            )
        except TableOCRDependencyError:
            raise
        except Exception as exc:
            return self._result(
                table,
                status="error",
                error=f"Could not initialize Xberg/TATR: {exc}",
            )

        try:
            output = self._extractor(extract_input, config)
        except TableOCRDependencyError:
            raise
        except Exception as exc:
            return self._result(
                table,
                status="error",
                error=f"Xberg/TATR extraction failed: {exc}",
            )

        errors = list(_read_public_attribute(output, "errors", []) or [])
        documents = list(_read_public_attribute(output, "results", []) or [])

        if errors:
            return self._result(
                table,
                status="error",
                native_json=[{"errors": _to_json_safe(errors)}],
                error=(
                    "Xberg/TATR returned extraction errors for the "
                    "canonical table crop."
                ),
            )

        if not documents:
            return self._result(
                table,
                status="empty",
                error="Xberg/TATR returned no document result.",
            )

        if len(documents) != 1:
            return self._result(
                table,
                status="error",
                result_count=len(documents),
                error=(
                    "Xberg/TATR returned multiple document results for one "
                    "canonical table crop."
                ),
            )

        document = documents[0]
        tables = list(_read_public_attribute(document, "tables", []) or [])
        serialized_tables = [
            _serialize_table(extracted_table)
            for extracted_table in tables
        ]
        native_json = [{
            "content": _to_json_safe(
                _read_public_attribute(document, "content", "")
            ),
            "tables": serialized_tables,
        }]
        native_markdown = [
            serialized_table["markdown"]
            for serialized_table in serialized_tables
            if serialized_table["markdown"]
        ]

        if not tables:
            return self._result(
                table,
                status="empty",
                result_count=0,
                native_json=native_json,
                error="Xberg/TATR detected no structured table in the crop.",
            )

        if len(tables) != 1:
            return self._result(
                table,
                status="error",
                result_count=len(tables),
                native_json=native_json,
                native_markdown=native_markdown,
                error=(
                    "Xberg/TATR detected multiple structured tables in one "
                    "canonical table crop."
                ),
            )

        if not native_markdown:
            return self._result(
                table,
                status="empty",
                result_count=1,
                native_json=native_json,
                error=(
                    "Xberg/TATR returned one table but no usable structured "
                    "Markdown or cells."
                ),
            )

        return self._result(
            table,
            status="ok",
            result_count=1,
            native_json=native_json,
            native_markdown=native_markdown,
        )
