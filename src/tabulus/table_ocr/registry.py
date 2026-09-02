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
    "deepseek-ocr-2": TableOCRAdapterSpec(
        name="deepseek-ocr-2",
        display_name="DeepSeek-OCR-2",
        entrypoint=(
            "tabulus.table_ocr.deepseek_ocr_2:DeepSeekOCR2Adapter"
        ),
        cpu_supported=False,
        gpu_supported=True,
    ),
    "dolphin-v2": TableOCRAdapterSpec(
        name="dolphin-v2",
        display_name="Dolphin-v2",
        entrypoint="tabulus.table_ocr.dolphin_v2:DolphinV2Adapter",
        cpu_supported=False,
        gpu_supported=True,
    ),
    "dots-mocr": TableOCRAdapterSpec(
        name="dots-mocr",
        display_name="dots.mocr",
        entrypoint="tabulus.table_ocr.dots_mocr:DotsMOCRAdapter",
        cpu_supported=False,
        gpu_supported=True,
    ),
    "paddleocr-vl": TableOCRAdapterSpec(
        name="paddleocr-vl",
        display_name="PaddleOCR-VL",
        entrypoint="tabulus.table_ocr.paddleocr_vl:PaddleOCRVLAdapter",
        cpu_supported=True,
        gpu_supported=True,
    ),
    "granite-vision-table": TableOCRAdapterSpec(
        name="granite-vision-table",
        display_name="Granite Vision 4.1 4B",
        entrypoint=(
            "tabulus.table_ocr.granite_vision_table:GraniteVisionTableAdapter"
        ),
        cpu_supported=False,
        gpu_supported=True,
    ),
    "glm-ocr": TableOCRAdapterSpec(
        name="glm-ocr",
        display_name="GLM-OCR",
        entrypoint="tabulus.table_ocr.glm_ocr:GLMOCRAdapter",
        cpu_supported=False,
        gpu_supported=True,
    ),
    "hunyuanocr-1-5": TableOCRAdapterSpec(
        name="hunyuanocr-1-5",
        display_name="HunyuanOCR-1.5",
        entrypoint=(
            "tabulus.table_ocr.hunyuanocr_1_5:"
            "HunyuanOCR15Adapter"
        ),
        cpu_supported=False,
        gpu_supported=True,
    ),
    "internvl3-5-8b": TableOCRAdapterSpec(
        name="internvl3-5-8b",
        display_name="InternVL3.5-8B",
        entrypoint=(
            "tabulus.table_ocr.internvl3_5_8b:InternVL35_8BAdapter"
        ),
        cpu_supported=False,
        gpu_supported=True,
    ),
    "nanonets-ocr-s": TableOCRAdapterSpec(
        name="nanonets-ocr-s",
        display_name="Nanonets-OCR-s",
        entrypoint=(
            "tabulus.table_ocr.nanonets_ocr_s:NanonetsOCRSAdapter"
        ),
        cpu_supported=False,
        gpu_supported=True,
    ),
    "monkeyocrv2-b-parsing": TableOCRAdapterSpec(
        name="monkeyocrv2-b-parsing",
        display_name="MonkeyOCRv2-B-Parsing",
        entrypoint=(
            "tabulus.table_ocr.monkeyocrv2_b_parsing:"
            "MonkeyOCRv2BParsingAdapter"
        ),
        cpu_supported=False,
        gpu_supported=True,
    ),
    "nemotron-parse-v1-2": TableOCRAdapterSpec(
        name="nemotron-parse-v1-2",
        display_name="NVIDIA Nemotron Parse v1.2",
        entrypoint=(
            "tabulus.table_ocr.nemotron_parse_v1_2:"
            "NemotronParseV12Adapter"
        ),
        cpu_supported=False,
        gpu_supported=True,
    ),
    "nuextract3": TableOCRAdapterSpec(
        name="nuextract3",
        display_name="NuExtract3",
        entrypoint="tabulus.table_ocr.nuextract3:NuExtract3Adapter",
        cpu_supported=False,
        gpu_supported=True,
    ),
    "rapidocr-tableformer": TableOCRAdapterSpec(
        name="rapidocr-tableformer",
        display_name="RapidOCR + Docling TableFormer",
        entrypoint=(
            "tabulus.table_ocr.rapidocr_tableformer:RapidOCRTableFormerAdapter"
        ),
        cpu_supported=True,
        gpu_supported=True,
    ),
    "trivia": TableOCRAdapterSpec(
        name="trivia",
        display_name="TRivia-3B",
        entrypoint="tabulus.table_ocr.trivia:TRiviaAdapter",
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
