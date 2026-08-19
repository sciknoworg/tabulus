from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tabulus.table_ocr.base import (
    TableOCRCapabilities,
    TableOCRInput,
    TableOCRResult,
)
from tabulus.table_ocr.batch import (
    load_table_ocr_inputs,
    run_table_ocr_batch,
)


def write_crop_index(
    crop_root: Path,
    table_ids: list[int],
) -> None:
    images = crop_root / "images"
    images.mkdir(parents=True, exist_ok=True)

    tables = []

    for table_id in table_ids:
        image_name = f"page_001_table_{table_id:03d}.jpg"
        image_path = images / image_name
        image_path.write_bytes(b"not-a-real-image")
        tables.append(
            {
                "table_id": table_id,
                "page_nr": 1,
                "image": f"images/{image_name}",
                "image_name": image_name,
                "bbox": [1, 2, 3, 4],
                "source": "mineru",
            }
        )

    (crop_root / "tables_index.json").write_text(
        json.dumps(
            {
                "tables_found": len(tables),
                "crops_saved": len(tables),
                "tables": tables,
            }
        ),
        encoding="utf-8",
    )


class FakeBatchAdapter:
    def __init__(self, error_table_id: int | None = None) -> None:
        self.error_table_id = error_table_id
        self.calls: list[int] = []
        self._capabilities = TableOCRCapabilities(
            name="fake",
            display_name="Fake reconstruction adapter",
            cpu_supported=True,
            gpu_supported=False,
        )

    @property
    def capabilities(self) -> TableOCRCapabilities:
        return self._capabilities

    def extract(self, table: TableOCRInput) -> TableOCRResult:
        self.calls.append(table.table_id)

        if table.table_id == self.error_table_id:
            return TableOCRResult(
                table_id=table.table_id,
                adapter_name="fake",
                device="cpu",
                source_image=table.image_path,
                status="error",
                provenance=table.provenance,
                error="synthetic inference failure",
            )

        return TableOCRResult(
            table_id=table.table_id,
            adapter_name="fake",
            device="cpu",
            source_image=table.image_path,
            status="ok",
            provenance=table.provenance,
            result_count=1,
            native_markdown=[
                {
                    "markdown_texts": (
                        "<table>"
                        "<tr><th>Material</th><th>Refs.</th></tr>"
                        f"<tr><td>T{table.table_id}</td>"
                        f"<td>{80 + table.table_id}</td></tr>"
                        "</table>"
                    )
                }
            ],
        )


def test_load_table_ocr_inputs_preserves_index_order_and_identity(
    tmp_path: Path,
) -> None:
    crop_root = tmp_path / "crops"
    write_crop_index(crop_root, [7, 2, 9])

    tables = load_table_ocr_inputs(crop_root)

    assert [table.table_id for table in tables] == [7, 2, 9]
    assert tables[0].provenance["page_nr"] == 1
    assert tables[0].image_path == (
        crop_root / "images/page_001_table_007.jpg"
    )


def test_batch_reuses_one_adapter_and_writes_all_artifact_layers(
    tmp_path: Path,
) -> None:
    crop_root = tmp_path / "crops"
    output_dir = tmp_path / "reconstructions/fake"
    write_crop_index(crop_root, [1, 2, 3])
    adapter = FakeBatchAdapter()

    result = run_table_ocr_batch(
        crop_root=crop_root,
        output_dir=output_dir,
        adapter=adapter,
    )

    assert adapter.calls == [1, 2, 3]
    assert result.tables_requested == 3
    assert result.tables_ok == 3
    assert result.tables_empty == 0
    assert result.tables_error == 0
    assert result.prediction_csvs == 3
    assert result.adapter_name == "fake"

    for table_id in (1, 2, 3):
        stem = f"page_001_table_{table_id:03d}"
        native = output_dir / "native" / f"{stem}.json"
        parsed = output_dir / "parsed" / f"{stem}.json"
        prediction = output_dir / "predictions" / f"{stem}.csv"

        assert native.is_file()
        assert parsed.is_file()
        assert prediction.is_file()

    with (
        output_dir / "predictions/page_001_table_002.csv"
    ).open(newline="", encoding="utf-8") as handle:
        assert list(csv.reader(handle)) == [
            ["Material", "Refs."],
            ["T2", "82"],
        ]

    summary = json.loads(
        (output_dir / "batch_summary.json").read_text(encoding="utf-8")
    )
    assert summary["tables_requested"] == 3
    assert summary["prediction_csvs"] == 3
    assert [item["table_id"] for item in summary["items"]] == [1, 2, 3]
    assert summary["items"][0]["prediction_csv"] == (
        "predictions/page_001_table_001.csv"
    )


def test_batch_continues_after_explicit_table_error(
    tmp_path: Path,
) -> None:
    crop_root = tmp_path / "crops"
    output_dir = tmp_path / "reconstructions/fake"
    write_crop_index(crop_root, [1, 2, 3])
    adapter = FakeBatchAdapter(error_table_id=2)

    result = run_table_ocr_batch(
        crop_root=crop_root,
        output_dir=output_dir,
        adapter=adapter,
    )

    assert adapter.calls == [1, 2, 3]
    assert result.tables_ok == 2
    assert result.tables_error == 1
    assert result.prediction_csvs == 2

    assert (
        output_dir / "native/page_001_table_002.json"
    ).is_file()
    assert (
        output_dir / "parsed/page_001_table_002.json"
    ).is_file()
    assert not (
        output_dir / "predictions/page_001_table_002.csv"
    ).exists()

    error_item = result.items[1]
    assert error_item.table_id == 2
    assert error_item.status == "error"
    assert error_item.error == "synthetic inference failure"


def test_duplicate_table_ids_are_rejected_before_inference(
    tmp_path: Path,
) -> None:
    crop_root = tmp_path / "crops"
    write_crop_index(crop_root, [1, 1])
    adapter = FakeBatchAdapter()

    with pytest.raises(ValueError, match="Duplicate table_id"):
        run_table_ocr_batch(
            crop_root=crop_root,
            output_dir=tmp_path / "out",
            adapter=adapter,
        )

    assert adapter.calls == []


def test_adapter_cannot_change_table_identity(
    tmp_path: Path,
) -> None:
    crop_root = tmp_path / "crops"
    write_crop_index(crop_root, [1])

    class WrongIdentityAdapter(FakeBatchAdapter):
        def extract(self, table: TableOCRInput) -> TableOCRResult:
            result = super().extract(table)
            result.table_id = 999
            return result

    adapter = WrongIdentityAdapter()

    with pytest.raises(
        ValueError,
        match="adapter changed table identity",
    ):
        run_table_ocr_batch(
            crop_root=crop_root,
            output_dir=tmp_path / "out",
            adapter=adapter,
        )
