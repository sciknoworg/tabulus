from __future__ import annotations

import html
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


VALIDATED_DOCLING_VERSION = "2.123.1"
MODEL_REPO = "docling-project/docling-models"
MODEL_REVISION = "v2.3.0"
TABLEFORMER_MODE = "accurate"
RAPIDOCR_LANGUAGE = "en"
RAPIDOCR_BACKEND = "onnxruntime"
RAPIDOCR_SCALE = 1.0
RAPIDOCR_TEXT_SCORE = 0.5

MODEL_VERSION = (
    f"{MODEL_REPO}@{MODEL_REVISION}:tableformer/{TABLEFORMER_MODE}"
)


@dataclass(frozen=True)
class _RapidOCRTableFormerRuntime:
    numpy: Any
    image_module: Any
    BoundingBox: Any
    BoundingRectangle: Any
    CoordOrigin: Any
    DocItemLabel: Any
    TextCell: Any
    Cluster: Any
    ocr_model: Any
    table_model: Any
    table_device: str
    docling_version: str | None
    docling_core_version: str | None
    docling_ibm_models_version: str | None
    rapidocr_version: str | None
    onnxruntime_version: str | None


RuntimeLoader = Callable[[str], Any]
InferenceRunner = Callable[[Path, Any], dict[str, Any]]


def _installed_package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _tableformer_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized == "gpu":
        return "cuda"
    if normalized.startswith("gpu:"):
        index = normalized.split(":", maxsplit=1)[1]
        if index.isdigit():
            return f"cuda:{index}"
    raise ValueError(
        "Use device='cpu', 'gpu', or 'gpu:<index>' for rapidocr-tableformer."
    )


def _dependency_error() -> TableOCRDependencyError:
    return TableOCRDependencyError(
        "The rapidocr-tableformer adapter requires docling==2.123.1, "
        "docling-core, docling-ibm-models, RapidOCR, ONNX Runtime, "
        "PyTorch, Pillow, and NumPy in the active environment."
    )


def _default_runtime_loader(
    table_device: str,
) -> _RapidOCRTableFormerRuntime:
    docling_version = _installed_package_version("docling")
    if docling_version != VALIDATED_DOCLING_VERSION:
        found = docling_version or "not installed"
        raise TableOCRDependencyError(
            "The rapidocr-tableformer adapter is validated against "
            f"docling=={VALIDATED_DOCLING_VERSION}; found {found}."
        )

    try:
        numpy = importlib.import_module("numpy")
        image_module = importlib.import_module("PIL.Image")
        torch = importlib.import_module("torch")

        doc_types = importlib.import_module("docling_core.types.doc")
        page_types = importlib.import_module("docling_core.types.doc.page")
        base_models = importlib.import_module("docling.datamodel.base_models")
        accelerator = importlib.import_module(
            "docling.datamodel.accelerator_options"
        )
        pipeline_options = importlib.import_module(
            "docling.datamodel.pipeline_options"
        )
        rapidocr_module = importlib.import_module(
            "docling.models.stages.ocr.rapid_ocr_model"
        )
        tableformer_module = importlib.import_module(
            "docling.models.stages.table_structure.table_structure_model"
        )
    except ImportError as exc:
        raise _dependency_error() from exc

    if table_device.startswith("cuda") and not bool(torch.cuda.is_available()):
        raise RuntimeError(
            "rapidocr-tableformer requested a GPU device but CUDA is "
            "not available."
        )

    try:
        ocr_options = pipeline_options.RapidOcrOptions(
            lang=[RAPIDOCR_LANGUAGE],
            backend=RAPIDOCR_BACKEND,
            scale=RAPIDOCR_SCALE,
            text_score=RAPIDOCR_TEXT_SCORE,
        )
        ocr_model = rapidocr_module.RapidOcrModel(
            enabled=True,
            artifacts_path=None,
            options=ocr_options,
            accelerator_options=accelerator.AcceleratorOptions(
                device="cpu"
            ),
        )

        table_options = pipeline_options.TableStructureOptions(
            mode=pipeline_options.TableFormerMode.ACCURATE,
            do_cell_matching=True,
        )
        table_model = tableformer_module.TableStructureModel(
            enabled=True,
            artifacts_path=None,
            options=table_options,
            accelerator_options=accelerator.AcceleratorOptions(
                device=table_device
            ),
        )
    except ImportError as exc:
        raise _dependency_error() from exc

    return _RapidOCRTableFormerRuntime(
        numpy=numpy,
        image_module=image_module,
        BoundingBox=doc_types.BoundingBox,
        BoundingRectangle=page_types.BoundingRectangle,
        CoordOrigin=doc_types.CoordOrigin,
        DocItemLabel=doc_types.DocItemLabel,
        TextCell=page_types.TextCell,
        Cluster=base_models.Cluster,
        ocr_model=ocr_model,
        table_model=table_model,
        table_device=table_device,
        docling_version=docling_version,
        docling_core_version=_installed_package_version("docling-core"),
        docling_ibm_models_version=_installed_package_version(
            "docling-ibm-models"
        ),
        rapidocr_version=_installed_package_version("rapidocr"),
        onnxruntime_version=_installed_package_version("onnxruntime"),
    )


def _cells_to_html(
    cells: list[dict[str, Any]],
    num_rows: int,
    num_cols: int,
) -> str:
    if num_rows <= 0 or num_cols <= 0:
        return ""

    anchors: dict[tuple[int, int], dict[str, Any]] = {}
    covered: set[tuple[int, int]] = set()

    for cell in cells:
        row_start = int(cell["start_row_offset_idx"])
        row_end = int(cell["end_row_offset_idx"])
        col_start = int(cell["start_col_offset_idx"])
        col_end = int(cell["end_col_offset_idx"])

        if not (
            0 <= row_start < row_end <= num_rows
            and 0 <= col_start < col_end <= num_cols
        ):
            raise ValueError(
                "Docling returned a table cell outside the reported "
                "table dimensions."
            )

        anchor = (row_start, col_start)
        if anchor in anchors:
            raise ValueError(
                "Docling returned multiple table cells with the same anchor."
            )
        anchors[anchor] = cell

        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                if (row, col) != anchor:
                    covered.add((row, col))

    output = ["<table>"]
    for row in range(num_rows):
        output.append("<tr>")
        for col in range(num_cols):
            if (row, col) in covered:
                continue

            cell = anchors.get((row, col))
            if cell is None:
                output.append("<td></td>")
                continue

            row_start = int(cell["start_row_offset_idx"])
            row_end = int(cell["end_row_offset_idx"])
            col_start = int(cell["start_col_offset_idx"])
            col_end = int(cell["end_col_offset_idx"])

            is_header = bool(
                cell.get("column_header")
                or cell.get("row_header")
                or cell.get("row_section")
            )
            tag = "th" if is_header else "td"

            attributes: list[str] = []
            if row_end - row_start > 1:
                attributes.append(
                    f'rowspan="{row_end - row_start}"'
                )
            if col_end - col_start > 1:
                attributes.append(
                    f'colspan="{col_end - col_start}"'
                )

            attribute_text = (
                " " + " ".join(attributes) if attributes else ""
            )
            text = html.escape(str(cell.get("text", "")), quote=False)
            output.append(
                f"<{tag}{attribute_text}>{text}</{tag}>"
            )

        output.append("</tr>")

    output.append("</table>")
    return "".join(output)


def _empty_reconstruction(
    *,
    image_size: list[int] | None = None,
    ocr_tokens: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    width = image_size[0] if image_size else 0
    height = image_size[1] if image_size else 0
    return {
        "html": "",
        "image_size": image_size or [],
        "table_box": (
            [0.0, 0.0, float(width), float(height)]
            if image_size
            else []
        ),
        "ocr_tokens": ocr_tokens or [],
        "otsl_seq": [],
        "num_rows": 0,
        "num_cols": 0,
        "cells": [],
    }


def _default_inference_runner(
    image_path: Path,
    runtime: _RapidOCRTableFormerRuntime,
) -> dict[str, Any]:
    with runtime.image_module.open(image_path) as source_image:
        image = source_image.convert("RGB")

    width, height = image.size
    image_size = [int(width), int(height)]
    image_array = runtime.numpy.asarray(image)

    ocr_result = runtime.ocr_model.reader(image_array)
    if ocr_result is None or ocr_result.boxes is None:
        return _empty_reconstruction(image_size=image_size)

    boxes = ocr_result.boxes.tolist()
    texts = list(ocr_result.txts)
    scores = list(ocr_result.scores)

    text_cells = []
    native_tokens: list[dict[str, Any]] = []

    for index, (box, text, score) in enumerate(
        zip(boxes, texts, scores)
    ):
        text = str(text)
        if not text.strip():
            continue

        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        coordinates = (
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        )

        bbox = runtime.BoundingBox.from_tuple(
            coord=coordinates,
            origin=runtime.CoordOrigin.TOPLEFT,
        )

        text_cells.append(
            runtime.TextCell(
                index=index,
                text=text,
                orig=text,
                confidence=float(score),
                from_ocr=True,
                rect=runtime.BoundingRectangle.from_bounding_box(bbox),
            )
        )

        native_tokens.append(
            {
                "id": index,
                "text": text,
                "confidence": float(score),
                "bbox": [float(value) for value in coordinates],
            }
        )

    if not text_cells:
        return _empty_reconstruction(
            image_size=image_size,
            ocr_tokens=native_tokens,
        )

    table_bbox = runtime.BoundingBox.from_tuple(
        coord=(0.0, 0.0, float(width), float(height)),
        origin=runtime.CoordOrigin.TOPLEFT,
    )
    table_cluster = runtime.Cluster(
        id=0,
        label=runtime.DocItemLabel.TABLE,
        bbox=table_bbox,
        cells=text_cells,
    )

    # Intentionally use Docling's bare-crop TableFormer path. The entire
    # canonical MinerU crop is declared as the table region; there is no
    # Docling layout detection or candidate-specific recropping.
    table = runtime.table_model._do_prediction_on_image_to_table(
        table_image=image,
        table_cluster=table_cluster,
        page_no=0,
    )

    cells = [
        cell.model_dump(mode="json")
        for cell in table.table_cells
    ]
    num_rows = int(table.num_rows)
    num_cols = int(table.num_cols)

    return {
        "html": _cells_to_html(cells, num_rows, num_cols),
        "image_size": image_size,
        "table_box": [
            0.0,
            0.0,
            float(width),
            float(height),
        ],
        "ocr_tokens": native_tokens,
        "otsl_seq": [str(token) for token in table.otsl_seq],
        "num_rows": num_rows,
        "num_cols": num_cols,
        "cells": cells,
    }


class RapidOCRTableFormerAdapter:
    """RapidOCR + Docling TableFormer reconstruction for MinerU crops."""

    NAME = "rapidocr-tableformer"
    DISPLAY_NAME = "RapidOCR + Docling TableFormer"
    MODEL_VERSION = MODEL_VERSION

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
        runtime_loader: RuntimeLoader | None = None,
        inference_runner: InferenceRunner | None = None,
    ) -> None:
        self.device = device
        self._table_device = _tableformer_device(device)
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
                self._table_device
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
                error=(
                    "Could not initialize RapidOCR + Docling "
                    f"TableFormer: {exc}"
                ),
            )

        adapter_version = getattr(
            runtime, "docling_version", None
        )

        try:
            reconstruction = self._inference_runner(
                image_path, runtime
            )
        except TableOCRDependencyError:
            raise
        except Exception as exc:
            return self._result(
                table,
                status="error",
                adapter_version=adapter_version,
                error=(
                    "RapidOCR + Docling TableFormer reconstruction "
                    f"failed: {exc}"
                ),
            )

        ocr_tokens = list(
            reconstruction.get("ocr_tokens", [])
        )
        otsl_seq = list(
            reconstruction.get("otsl_seq", [])
        )
        cells = list(reconstruction.get("cells", []))
        num_rows = int(reconstruction.get("num_rows", 0))
        num_cols = int(reconstruction.get("num_cols", 0))
        html_table = str(
            reconstruction.get("html", "")
        ).strip()

        native_json = [
            {
                "rapidocr": {
                    "package_version": getattr(
                        runtime, "rapidocr_version", None
                    ),
                    "onnxruntime_version": getattr(
                        runtime, "onnxruntime_version", None
                    ),
                    "backend": RAPIDOCR_BACKEND,
                    "language": RAPIDOCR_LANGUAGE,
                    "scale": RAPIDOCR_SCALE,
                    "text_score": RAPIDOCR_TEXT_SCORE,
                    "execution_device": "cpu",
                    "tokens": ocr_tokens,
                },
                "tableformer": {
                    "docling_version": adapter_version,
                    "docling_core_version": getattr(
                        runtime, "docling_core_version", None
                    ),
                    "docling_ibm_models_version": getattr(
                        runtime,
                        "docling_ibm_models_version",
                        None,
                    ),
                    "model_repo": MODEL_REPO,
                    "model_revision": MODEL_REVISION,
                    "mode": TABLEFORMER_MODE,
                    "do_cell_matching": True,
                    "execution_device": getattr(
                        runtime,
                        "table_device",
                        self._table_device,
                    ),
                    "image_size": reconstruction.get(
                        "image_size", []
                    ),
                    "table_box": reconstruction.get(
                        "table_box", []
                    ),
                    "otsl_seq": otsl_seq,
                    "num_rows": num_rows,
                    "num_cols": num_cols,
                    "cells": cells,
                },
            }
        ]

        if not ocr_tokens:
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=native_json,
                error=(
                    "RapidOCR returned no OCR text tokens for the "
                    "canonical crop."
                ),
            )

        if (
            num_rows <= 0
            or num_cols <= 0
            or not cells
            or not html_table
        ):
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=native_json,
                error=(
                    "Docling TableFormer returned no usable structured "
                    "table for the canonical crop."
                ),
            )

        if not any(
            str(cell.get("text", "")).strip()
            for cell in cells
        ):
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                result_count=1,
                native_json=native_json,
                native_markdown=[html_table],
                error=(
                    "RapidOCR + Docling TableFormer produced table "
                    "structure but no populated cells."
                ),
            )

        return self._result(
            table,
            status="ok",
            adapter_version=adapter_version,
            result_count=1,
            native_json=native_json,
            native_markdown=[html_table],
        )
