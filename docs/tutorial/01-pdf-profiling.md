# Step 1: PDF Profiling

## Goal

Read one scientific PDF paper from an existing folder, analyze its page layout, detect tables, and produce the table images and structured metadata that downstream table reconstruction modules consume.

There is no separate upload or ingestion module in the clean workflow. The first module is **PDF Profiling**. In this tutorial, PDF profiling means structural profiling of the paper, not only file metadata.

The module performs:

- page/layout analysis
- table detection
- table bounding-box detection
- table crop export
- caption and footnote capture
- structured JSON export

The plug-and-play part is the library adapter. The first adapter is MinerU.

## Tool Choice

The current web application starts from FastAPI upload, but the reusable processing logic is in the MinerU runner:

```text
src/Tabulus/mineru_service/app/table_extraction_benchmark/runners/mineru_tables_png_runner.py
```

That script handles MinerU outputs. It finds MinerU's `content_list.json`, selects entries with `type == "table"`, reads each table's `img_path`, copies the MinerU-generated table image, and writes table metadata.

Important detail: Tabulus currently does **not** crop the PDF itself from the `bbox`. MinerU has already generated the crop image.

The script also preserves fields such as:

- `bbox`
- `table_caption`
- `table_footnote`
- `mineru_img_path`

## Input

One PDF file from a local folder of papers.

Example:

```text
papers/
  P51.pdf
```

## Output

A run directory with table images and structured metadata.

```text
runs/
  P51/
    input/
      paper.pdf
    images/
      tables/
        page_003_table_001.png
        tables_index.json
    mineru_out/
    mineru_stdout.log
    mineru_stderr.log
    notes.md
```

## Module Contract

The profiling component should produce table images and a structured table index.

```json
{
  "run_id": "P51",
  "adapter": "mineru",
  "adapter_version": "3.4.5",
  "tables_found": 1,
  "crops_saved": 1,
  "refs_start_page": 12,
  "tables": [
    {
      "table_id": 1,
      "page_nr": 3,
      "png_name": "page_003_table_001.png",
      "png": "runs/P51/images/tables/page_003_table_001.png",
      "mineru_img_path": "ocr/images/example.png",
      "bbox": [100, 200, 900, 600],
      "table_caption": "Table 1. Example caption",
      "table_footnote": null
    }
  ],
  "status": "profiled"
}
```

## Default Implementation

The current implementation is the MinerU table PNG runner.

Python snippet:

```python
from pathlib import Path

from app.table_extraction_benchmark.runners.mineru_tables_png_runner import run

pdf_path = Path("/data/runs/P51/input/paper.pdf")
run_dir = Path("/data/runs/P51")

run(pdf_path, run_dir)
```

Expected result:

```text
runs/P51/images/tables/page_003_table_001.png
runs/P51/images/tables/tables_index.json
runs/P51/mineru_out/
```

## Alternative Adapters

- MinerU
- Docling or another layout parser
- A custom PDF layout detector
- A table detector that emits crop images and structured table metadata

All adapters should produce the same table-image and structured-metadata contract.

## Standalone Run Target

The first useful development target is:

```text
Given a folder of papers, run PDF profiling for each PDF and emit table images plus structured JSON for each paper.
```

## Verification

The step succeeds when:

- `images/tables/tables_index.json` exists.
- Each indexed table image exists.
- Each table has a table id and page number.
- Bounding boxes, captions, and footnotes are preserved when the adapter provides them.
- The status is `profiled`.
- A later step can read each table image path.

## Common Failure Modes

| Failure | Likely cause | Fix |
| --- | --- | --- |
| File not found | Wrong path or manifest entry | Validate paths before processing. |
| MinerU output missing | MinerU failed or did not write `content_list.json` | Inspect `mineru_stderr.log` and `notes.md`. |
| No table images | MinerU found no table entries or `img_path` resolution failed | Inspect `content_list.json` and `mineru_img_path` values. |
| Missing bbox or caption | Adapter did not provide optional metadata | Preserve `null` and continue. |

## Next Step

After PDF profiling, send the emitted table images to the table OCR and structure extraction module.
