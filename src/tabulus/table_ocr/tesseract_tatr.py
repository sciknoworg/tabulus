from __future__ import annotations

import csv
import html
import importlib
import io
import shutil
import subprocess
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


MODEL_ID = "microsoft/table-transformer-structure-recognition-v1.1-all"
TESSERACT_LANGUAGE = "eng"
TESSERACT_PSM = 6
TATR_THRESHOLD = 0.5
TATR_MAX_SIZE = 1000


@dataclass(frozen=True)
class _TesseractTATRRuntime:
    torch: Any
    transforms: Any
    image_module: Any
    model: Any
    postprocess: Any
    torch_device: str
    tesseract_executable: str
    tesseract_version: str


RuntimeLoader = Callable[[str], Any]
InferenceRunner = Callable[[Path, Any], dict[str, Any]]


def _installed_package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _torch_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized == "gpu":
        return "cuda"
    if normalized.startswith("gpu:"):
        index = normalized.split(":", maxsplit=1)[1]
        if index.isdigit():
            return f"cuda:{index}"
    raise ValueError("Use device='cpu', 'gpu', or 'gpu:<index>' for tesseract-tatr.")


def _dependency_error(exc: ImportError | None = None) -> TableOCRDependencyError:
    message = (
        "The tesseract-tatr adapter requires the Tesseract executable plus "
        "PyTorch, torchvision, Transformers 4.x, Pillow, timm, and PyMuPDF "
        "in the active environment."
    )
    return TableOCRDependencyError(message)


def _read_tesseract_version(executable: str) -> str:
    completed = subprocess.run(
        [executable, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    first_line = (completed.stdout or completed.stderr).splitlines()[0]
    return first_line.strip()


def _default_runtime_loader(torch_device: str) -> _TesseractTATRRuntime:
    executable = shutil.which("tesseract")
    if executable is None:
        raise TableOCRDependencyError(
            "The tesseract-tatr adapter could not find the Tesseract executable on PATH."
        )

    try:
        torch = importlib.import_module("torch")
        torchvision = importlib.import_module("torchvision")
        transformers = importlib.import_module("transformers")
        importlib.import_module("timm")
        image_module = importlib.import_module("PIL.Image")
        postprocess = importlib.import_module("tabulus.table_ocr.tatr_postprocess")
    except ImportError as exc:
        raise _dependency_error(exc) from exc

    transformers_version = _installed_package_version("transformers")
    if transformers_version is not None:
        try:
            major = int(transformers_version.split(".", maxsplit=1)[0])
        except ValueError:
            major = 0
        if major >= 5:
            raise TableOCRDependencyError(
                "The validated tesseract-tatr model configuration requires Transformers 4.x; "
                f"found {transformers_version}. The validated environment uses 4.57.6."
            )

    if torch_device.startswith("cuda") and not bool(torch.cuda.is_available()):
        raise RuntimeError("tesseract-tatr requested a GPU device but CUDA is not available.")

    model = transformers.AutoModelForObjectDetection.from_pretrained(MODEL_ID)
    model = model.to(torch.device(torch_device)).eval()

    return _TesseractTATRRuntime(
        torch=torch,
        transforms=torchvision.transforms,
        image_module=image_module,
        model=model,
        postprocess=postprocess,
        torch_device=torch_device,
        tesseract_executable=executable,
        tesseract_version=_read_tesseract_version(executable),
    )


def _parse_tesseract_tsv(tsv: str) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t"):
        if row.get("level") != "5":
            continue
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            x = float(row["left"])
            y = float(row["top"])
            width = float(row["width"])
            height = float(row["height"])
            confidence = float(row["conf"])
            block_num = int(row["block_num"])
            line_num = int(row["line_num"])
            span_num = int(row["word_num"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Tesseract TSV contains an invalid word record.") from exc
        tokens.append(
            {
                "text": text,
                "bbox": [x, y, x + width, y + height],
                "block_num": block_num,
                "line_num": line_num,
                "span_num": span_num,
                "confidence": confidence,
            }
        )
    return tokens


def _run_tesseract(image_path: Path, runtime: _TesseractTATRRuntime) -> tuple[str, list[dict[str, Any]]]:
    command = [
        runtime.tesseract_executable,
        str(image_path),
        "stdout",
        "-l",
        TESSERACT_LANGUAGE,
        "--psm",
        str(TESSERACT_PSM),
        "tsv",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return " ".join(command), _parse_tesseract_tsv(completed.stdout)


def _tatr_objects(image: Any, runtime: _TesseractTATRRuntime) -> tuple[dict[int, str], list[dict[str, Any]]]:
    width, height = image.size
    scale = TATR_MAX_SIZE / max(width, height)
    resized = image.resize((int(round(width * scale)), int(round(height * scale))))

    transform = runtime.transforms.Compose(
        [
            runtime.transforms.ToTensor(),
            runtime.transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )
    pixel_values = transform(resized).unsqueeze(0).to(runtime.torch.device(runtime.torch_device))

    with runtime.torch.no_grad():
        outputs = runtime.model(pixel_values=pixel_values)

    probabilities = outputs.logits.softmax(-1)[0]
    scores, labels = probabilities.max(-1)
    boxes = outputs.pred_boxes[0]
    class_names = {int(key): str(value) for key, value in runtime.model.config.id2label.items()}

    objects: list[dict[str, Any]] = []
    for score, label, box in zip(scores, labels, boxes):
        label_id = int(label)
        if label_id not in class_names:
            continue
        score_value = float(score)
        if score_value < TATR_THRESHOLD:
            continue
        cx, cy, box_width, box_height = [float(value) for value in box]
        objects.append(
            {
                "label": label_id,
                "score": score_value,
                "bbox": [
                    (cx - box_width / 2) * width,
                    (cy - box_height / 2) * height,
                    (cx + box_width / 2) * width,
                    (cy + box_height / 2) * height,
                ],
            }
        )
    return class_names, objects


def _cells_to_html(cells: list[dict[str, Any]]) -> str:
    if not cells:
        return ""

    n_rows = max(max(cell["row_nums"]) for cell in cells) + 1
    n_cols = max(max(cell["column_nums"]) for cell in cells) + 1
    anchors: dict[tuple[int, int], dict[str, Any]] = {}
    covered: set[tuple[int, int]] = set()

    for cell in cells:
        rows = sorted(int(value) for value in cell["row_nums"])
        cols = sorted(int(value) for value in cell["column_nums"])
        if not rows or not cols:
            continue
        anchor = (rows[0], cols[0])
        anchors[anchor] = cell
        for row in rows:
            for col in cols:
                if (row, col) != anchor:
                    covered.add((row, col))

    output = ["<table>"]
    for row in range(n_rows):
        output.append("<tr>")
        for col in range(n_cols):
            if (row, col) in covered:
                continue
            cell = anchors.get((row, col))
            if cell is None:
                output.append("<td></td>")
                continue
            rows = sorted(int(value) for value in cell["row_nums"])
            cols = sorted(int(value) for value in cell["column_nums"])
            tag = "th" if cell.get("header") else "td"
            attributes: list[str] = []
            if len(rows) > 1:
                attributes.append(f'rowspan="{len(rows)}"')
            if len(cols) > 1:
                attributes.append(f'colspan="{len(cols)}"')
            attribute_text = (" " + " ".join(attributes)) if attributes else ""
            text = html.escape(str(cell.get("cell_text", "")), quote=False)
            output.append(f"<{tag}{attribute_text}>{text}</{tag}>")
        output.append("</tr>")
    output.append("</table>")
    return "".join(output)


def _json_safe_cell(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "bbox": [float(value) for value in cell["bbox"]],
        "row_nums": [int(value) for value in cell["row_nums"]],
        "column_nums": [int(value) for value in cell["column_nums"]],
        "header": bool(cell.get("header")),
        "subheader": bool(cell.get("subheader")),
        "cell_text": str(cell.get("cell_text", "")),
    }


def _default_inference_runner(image_path: Path, runtime: _TesseractTATRRuntime) -> dict[str, Any]:
    command, tokens = _run_tesseract(image_path, runtime)
    if not tokens:
        return {
            "html": "",
            "tokens": [],
            "objects": [],
            "structure": {"rows": [], "columns": []},
            "cells": [],
            "token_slot_confidence": 0.0,
            "tesseract_command": command,
        }

    with runtime.image_module.open(image_path) as source_image:
        image = source_image.convert("RGB")

    class_names, objects = _tatr_objects(image, runtime)
    table_label = next((key for key, value in class_names.items() if value == "table"), None)
    if table_label is None:
        raise RuntimeError("TATR model configuration contains no 'table' class.")

    tables = [obj for obj in objects if obj["label"] == table_label]
    if not tables:
        return {
            "html": "",
            "tokens": tokens,
            "objects": objects,
            "structure": {"rows": [], "columns": []},
            "cells": [],
            "token_slot_confidence": 0.0,
            "tesseract_command": command,
        }

    table = max(tables, key=lambda obj: obj["score"]).copy()
    table["page_num"] = 1
    structure_objects = [obj for obj in objects if obj["label"] != table_label]
    thresholds = {
        "table": TATR_THRESHOLD,
        "table column": TATR_THRESHOLD,
        "table row": TATR_THRESHOLD,
        "table column header": TATR_THRESHOLD,
        "table projected row header": TATR_THRESHOLD,
        "table spanning cell": TATR_THRESHOLD,
        "no object": 10.0,
    }
    structure, cells, confidence = runtime.postprocess.objects_to_cells(
        table,
        structure_objects,
        tokens,
        class_names,
        thresholds,
    )
    return {
        "html": _cells_to_html(cells),
        "tokens": tokens,
        "objects": [
            {
                "label": class_names[obj["label"]],
                "score": float(obj["score"]),
                "bbox": [float(value) for value in obj["bbox"]],
            }
            for obj in objects
        ],
        "structure": {
            "rows": [row["bbox"] for row in structure.get("rows", [])],
            "columns": [column["bbox"] for column in structure.get("columns", [])],
            "headers": [header["bbox"] for header in structure.get("headers", [])],
            "supercells": [cell["bbox"] for cell in structure.get("supercells", [])],
        },
        "cells": [_json_safe_cell(cell) for cell in cells],
        "token_slot_confidence": float(confidence),
        "tesseract_command": command,
    }


class TesseractTATRAdapter:
    """Tesseract OCR + Table Transformer structure recognition for MinerU crops."""

    NAME = "tesseract-tatr"
    DISPLAY_NAME = "Tesseract + Table Transformer"
    MODEL_VERSION = MODEL_ID

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
                error=f"Could not initialize Tesseract + TATR: {exc}",
            )

        try:
            reconstruction = self._inference_runner(image_path, runtime)
        except TableOCRDependencyError:
            raise
        except Exception as exc:
            return self._result(
                table,
                status="error",
                error=f"Tesseract + TATR reconstruction failed: {exc}",
            )

        html_table = str(reconstruction.get("html", "")).strip()
        tokens = list(reconstruction.get("tokens", []))
        cells = list(reconstruction.get("cells", []))
        native_json = [
            {
                "tesseract": {
                    "version": getattr(runtime, "tesseract_version", None),
                    "language": TESSERACT_LANGUAGE,
                    "psm": TESSERACT_PSM,
                    "command": reconstruction.get("tesseract_command"),
                    "tokens": tokens,
                },
                "tatr": {
                    "model_id": MODEL_ID,
                    "threshold": TATR_THRESHOLD,
                    "max_size": TATR_MAX_SIZE,
                    "objects": reconstruction.get("objects", []),
                    "structure": reconstruction.get("structure", {}),
                    "cells": cells,
                    "token_slot_confidence": reconstruction.get("token_slot_confidence", 0.0),
                },
            }
        ]

        if not tokens:
            return self._result(
                table,
                status="empty",
                native_json=native_json,
                error="Tesseract returned no OCR word tokens for the canonical crop.",
            )

        if not cells or not html_table:
            return self._result(
                table,
                status="empty",
                native_json=native_json,
                error="TATR returned no usable structured table for the canonical crop.",
            )

        if not any(str(cell.get("cell_text", "")).strip() for cell in cells):
            return self._result(
                table,
                status="empty",
                result_count=1,
                native_json=native_json,
                native_markdown=[html_table],
                error="Tesseract + TATR produced table structure but no populated cells.",
            )

        return self._result(
            table,
            status="ok",
            result_count=1,
            native_json=native_json,
            native_markdown=[html_table],
        )
