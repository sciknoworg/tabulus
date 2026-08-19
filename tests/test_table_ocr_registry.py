from __future__ import annotations

import pytest

from tabulus.table_ocr import (
    get_table_ocr_adapter_class,
    list_table_ocr_adapters,
)


def test_paddleocr_vl_is_registered() -> None:
    specs = {
        spec.name: spec
        for spec in list_table_ocr_adapters()
    }

    spec = specs["paddleocr-vl"]

    assert spec.display_name == "PaddleOCR-VL"
    assert spec.cpu_supported is True
    assert spec.gpu_supported is True


def test_chandra_is_registered() -> None:
    specs = {
        spec.name: spec
        for spec in list_table_ocr_adapters()
    }

    spec = specs["chandra"]

    assert spec.display_name == "Chandra OCR 2"
    assert spec.cpu_supported is True
    assert spec.gpu_supported is True


def test_registry_loads_paddle_adapter_without_loading_ml_runtime() -> None:
    adapter_class = get_table_ocr_adapter_class("paddleocr-vl")

    assert adapter_class.__name__ == "PaddleOCRVLAdapter"


def test_registry_loads_chandra_adapter_without_loading_ml_runtime() -> None:
    adapter_class = get_table_ocr_adapter_class("chandra")

    assert adapter_class.__name__ == "ChandraAdapter"


def test_unknown_adapter_lists_available_names() -> None:
    with pytest.raises(ValueError, match="chandra.*paddleocr-vl"):
        get_table_ocr_adapter_class("does-not-exist")
