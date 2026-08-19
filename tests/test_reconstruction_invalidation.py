from __future__ import annotations

import json
from pathlib import Path

from tabulus.table_ocr.base import (
    TableOCRCapabilities,
    TableOCRInput,
    TableOCRResult,
)
from tabulus.table_ocr.batch import run_table_ocr_batch


class MinimalAdapter:
    @property
    def capabilities(self) -> TableOCRCapabilities:
        return TableOCRCapabilities(
            name="minimal",
            display_name="Minimal",
            cpu_supported=True,
            gpu_supported=False,
        )

    def extract(self, table: TableOCRInput) -> TableOCRResult:
        return TableOCRResult(
            table_id=table.table_id,
            adapter_name="minimal",
            device="cpu",
            source_image=table.image_path,
            status="ok",
            provenance=table.provenance,
            result_count=1,
            native_markdown=[
                {
                    "markdown_texts": (
                        "<table><tr><th>Material</th></tr>"
                        "<tr><td>Al2O3</td></tr></table>"
                    )
                }
            ],
        )


def test_reconstruction_rerun_invalidates_reference_classification(
    tmp_path: Path,
) -> None:
    crop_root = tmp_path / "crops"
    images_dir = crop_root / "images"
    images_dir.mkdir(parents=True)

    image = images_dir / "page_001_table_001.jpg"
    image.write_bytes(b"image")

    (crop_root / "tables_index.json").write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "table_id": 1,
                        "page_nr": 1,
                        "image": "images/page_001_table_001.jpg",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "reconstructions/minimal"
    output_dir.mkdir(parents=True)

    stale_classification = (
        output_dir / "reference_table_classification.json"
    )
    stale_classification.write_text(
        '{"stale": true}',
        encoding="utf-8",
    )

    run_table_ocr_batch(
        crop_root=crop_root,
        output_dir=output_dir,
        adapter=MinimalAdapter(),
    )

    assert not stale_classification.exists()
