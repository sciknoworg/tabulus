from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class TableOCRAdapterSpec:
    """Registry metadata that can be inspected without loading ML libraries."""

    name: str
    display_name: str
    entrypoint: str
    cpu_supported: bool
    gpu_supported: bool


_ADAPTERS: dict[str, TableOCRAdapterSpec] = {
    "chandra": TableOCRAdapterSpec(
        name="chandra",
        display_name="Chandra OCR 2",
        entrypoint="tabulus.table_ocr.chandra:ChandraAdapter",
        cpu_supported=True,
        gpu_supported=True,
    ),
    "paddleocr-vl": TableOCRAdapterSpec(
        name="paddleocr-vl",
        display_name="PaddleOCR-VL",
        entrypoint="tabulus.table_ocr.paddleocr_vl:PaddleOCRVLAdapter",
        cpu_supported=True,
        gpu_supported=True,
    ),
    "nuextract3": TableOCRAdapterSpec(
        name="nuextract3",
        display_name="NuExtract3",
        entrypoint="tabulus.table_ocr.nuextract3:NuExtract3Adapter",
        cpu_supported=False,
        gpu_supported=True,
    ),
    "tesseract-tatr": TableOCRAdapterSpec(
        name="tesseract-tatr",
        display_name="Tesseract + Table Transformer",
        entrypoint="tabulus.table_ocr.tesseract_tatr:TesseractTATRAdapter",
        cpu_supported=True,
        gpu_supported=True,
    ),
}


def list_table_ocr_adapters() -> tuple[TableOCRAdapterSpec, ...]:
    """Return registered adapters without importing their ML dependencies."""

    return tuple(_ADAPTERS[name] for name in sorted(_ADAPTERS))


def get_table_ocr_adapter_class(name: str) -> type[Any]:
    """Load and return a registered adapter class."""

    try:
        spec = _ADAPTERS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_ADAPTERS))
        raise ValueError(
            f"Unknown table OCR adapter {name!r}. Available adapters: {available}"
        ) from exc

    module_name, class_name = spec.entrypoint.split(":", maxsplit=1)
    module = import_module(module_name)

    try:
        adapter_class = getattr(module, class_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"Table OCR adapter entrypoint is invalid: {spec.entrypoint}"
        ) from exc

    return adapter_class


def create_table_ocr_adapter(name: str, **kwargs: Any) -> Any:
    """Instantiate a registered table OCR adapter."""

    adapter_class = get_table_ocr_adapter_class(name)
    return adapter_class(**kwargs)
