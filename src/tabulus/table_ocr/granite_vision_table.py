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
VALIDATED_TRANSFORMERS_VERSION = "4.57.3"
MODEL_REPO = "ibm-granite/granite-vision-4.1-4b"
MODEL_REVISION = "dd48e97503de471803850df70843cf9eb5da8712"
TABLE_PROMPT = "<tables_otsl>"
MODEL_DTYPE = "bfloat16"
ATTENTION_IMPLEMENTATION = "sdpa"

MODEL_VERSION = f"{MODEL_REPO}@{MODEL_REVISION}"


@dataclass(frozen=True)
class _GraniteVisionRuntime:
    torch: Any
    image_module: Any
    processor: Any
    model: Any
    parse_otsl_output: Any
    model_device: str
    docling_version: str | None
    transformers_version: str | None
    torch_version: str | None


RuntimeLoader = Callable[[str], Any]
InferenceRunner = Callable[[Path, Any], dict[str, Any]]


def _installed_package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _granite_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized == "gpu":
        return "cuda"
    if normalized.startswith("gpu:"):
        index = normalized.split(":", maxsplit=1)[1]
        if index.isdigit():
            return f"cuda:{index}"
    raise ValueError(
        "Use device='gpu' or 'gpu:<index>' for granite-vision-table."
    )


def _dependency_error() -> TableOCRDependencyError:
    return TableOCRDependencyError(
        "The granite-vision-table adapter requires docling==2.123.1, "
        "transformers==4.57.3, PyTorch, Pillow, and the Docling VLM "
        "dependencies in the active environment."
    )


def _default_runtime_loader(
    model_device: str,
) -> _GraniteVisionRuntime:
    docling_version = _installed_package_version("docling")
    if docling_version != VALIDATED_DOCLING_VERSION:
        found = docling_version or "not installed"
        raise TableOCRDependencyError(
            "The granite-vision-table adapter is validated against "
            f"docling=={VALIDATED_DOCLING_VERSION}; found {found}."
        )

    transformers_version = _installed_package_version("transformers")
    if transformers_version != VALIDATED_TRANSFORMERS_VERSION:
        found = transformers_version or "not installed"
        raise TableOCRDependencyError(
            "The granite-vision-table adapter is validated against "
            f"transformers=={VALIDATED_TRANSFORMERS_VERSION}; found {found}."
        )

    try:
        torch = importlib.import_module("torch")
        image_module = importlib.import_module("PIL.Image")
        transformers = importlib.import_module("transformers")
        granite_module = importlib.import_module(
            "docling.models.stages.table_structure."
            "table_structure_model_granite_vision"
        )
    except ImportError as exc:
        raise _dependency_error() from exc

    if not bool(torch.cuda.is_available()):
        raise RuntimeError(
            "granite-vision-table requires CUDA in the validated "
            "Tabulus configuration, but CUDA is not available."
        )

    processor = transformers.AutoProcessor.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        use_fast=False,
    )
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        device_map=model_device,
        dtype=torch.bfloat16,
        _attn_implementation=ATTENTION_IMPLEMENTATION,
        trust_remote_code=True,
    )

    if hasattr(model, "merge_lora_adapters"):
        model.merge_lora_adapters()

    model.eval()

    return _GraniteVisionRuntime(
        torch=torch,
        image_module=image_module,
        processor=processor,
        model=model,
        parse_otsl_output=granite_module._parse_otsl_output,
        model_device=model_device,
        docling_version=docling_version,
        transformers_version=transformers_version,
        torch_version=_installed_package_version("torch"),
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
                "Docling returned a Granite table cell outside the "
                "reported table dimensions."
            )

        anchor = (row_start, col_start)
        if anchor in anchors:
            raise ValueError(
                "Docling returned multiple Granite table cells with "
                "the same anchor."
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


def _default_inference_runner(
    image_path: Path,
    runtime: _GraniteVisionRuntime,
) -> dict[str, Any]:
    with runtime.image_module.open(image_path) as source_image:
        image = source_image.convert("RGB")

    width, height = image.size

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": TABLE_PROMPT},
            ],
        }
    ]
    prompt = runtime.processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = runtime.processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
        do_pad=True,
    ).to(runtime.model_device)

    with runtime.torch.inference_mode():
        output_ids = runtime.model.generate(
            **inputs,
            max_new_tokens=runtime.processor.tokenizer.model_max_length,
            use_cache=True,
        )

    prompt_length = int(inputs["input_ids"].shape[1])
    generated = output_ids[0, prompt_length:]
    raw_output = runtime.processor.decode(
        generated,
        skip_special_tokens=True,
    )

    otsl_seq, table_cells, num_rows, num_cols = (
        runtime.parse_otsl_output(raw_output)
    )
    cells = [
        cell.model_dump(mode="json")
        for cell in table_cells
    ]
    num_rows = int(num_rows)
    num_cols = int(num_cols)

    return {
        "raw_output": raw_output,
        "html": _cells_to_html(cells, num_rows, num_cols),
        "image_size": [int(width), int(height)],
        "generated_tokens": int(generated.numel()),
        "otsl_seq": [str(token) for token in otsl_seq],
        "num_rows": num_rows,
        "num_cols": num_cols,
        "cells": cells,
    }


class GraniteVisionTableAdapter:
    """Granite Vision 4.1 4B table reconstruction for MinerU crops."""

    NAME = "granite-vision-table"
    DISPLAY_NAME = "Granite Vision 4.1 4B"
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
        self._model_device = _granite_device(device)
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
                error=f"Could not initialize Granite Vision 4.1 4B: {exc}",
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
                error=f"Granite Vision table reconstruction failed: {exc}",
            )

        native = {
            "granite_vision": {
                "model_repo": MODEL_REPO,
                "model_revision": MODEL_REVISION,
                "prompt": TABLE_PROMPT,
                "dtype": MODEL_DTYPE,
                "attention_implementation": ATTENTION_IMPLEMENTATION,
                "execution_device": getattr(
                    runtime, "model_device", self._model_device
                ),
                "docling_version": adapter_version,
                "transformers_version": getattr(
                    runtime, "transformers_version", None
                ),
                "torch_version": getattr(
                    runtime, "torch_version", None
                ),
                "image_size": reconstruction.get("image_size", []),
                "generated_tokens": reconstruction.get(
                    "generated_tokens", 0
                ),
                "raw_output": reconstruction.get("raw_output", ""),
                "otsl_seq": reconstruction.get("otsl_seq", []),
                "num_rows": reconstruction.get("num_rows", 0),
                "num_cols": reconstruction.get("num_cols", 0),
                "cells": reconstruction.get("cells", []),
            }
        }

        raw_output = str(reconstruction.get("raw_output", ""))
        if not raw_output.strip():
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=[native],
                error="Granite Vision returned no generated table output.",
            )

        num_rows = int(reconstruction.get("num_rows", 0))
        num_cols = int(reconstruction.get("num_cols", 0))
        cells = list(reconstruction.get("cells", []))
        html_table = str(reconstruction.get("html", ""))

        if num_rows <= 0 or num_cols <= 0 or not cells or not html_table:
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=[native],
                error=(
                    "Granite Vision output did not contain a usable "
                    "OTSL table."
                ),
            )

        return self._result(
            table,
            status="ok",
            adapter_version=adapter_version,
            result_count=1,
            native_json=[native],
            native_markdown=[html_table],
        )
