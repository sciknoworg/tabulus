# Step 1: PDF Profiling

## Goal

Read one scientific PDF paper with MinerU, then use the Tabulus library to inspect the resulting structured MinerU outputs as typed table regions.

There is no separate upload or ingestion module in the clean workflow. The first module is **PDF Profiling**. In this tutorial, PDF profiling means structural profiling of the paper, not only file metadata.

The module performs:

- page/layout analysis
- table detection
- table bounding-box detection
- table crop export
- caption and footnote capture
- structured JSON export

The plug-and-play part is the profiling adapter. The first adapter is MinerU.

Important current boundary: Tabulus can launch MinerU through `tabulus profile`, but the typed table discovery and crop-export steps remain file-contract based. They consume the MinerU output directory after MinerU has produced it.

## Tool Choice

The current validated library entry point is:

```python
from pathlib import Path
from tabulus.mineru import discover_tables

tables, refs_start_page = discover_tables(Path("work/mineru/puurunen_2005"))
```

The library handles existing MinerU outputs. It recursively finds MinerU's `*_content_list.json`, selects entries with `type == "table"`, resolves each table's `img_path`, converts zero-based `page_idx` values into document page numbers, and preserves provenance.

Important detail: Tabulus currently does **not** crop the PDF itself from the `bbox`. MinerU has already generated the crop image.

See `data-contracts/mineru-output-files.md` for the full set of MinerU files and how Tabulus uses them.

The library preserves fields such as:

- `bbox`
- `table_caption`
- `table_footnote`
- `mineru_img_path`
- `table_body`

The current validated CLI entry points are:

```powershell
tabulus profile --pdf paper.pdf --out work/mineru/paper --backend pipeline
tabulus export-table-crops --mineru-root work/mineru/paper --out work/table_crops
```

## Input

An existing MinerU output directory produced from one PDF file.

Example:

```text
work/mineru/puurunen_2005/
  ...
  <document-name>_content_list.json
  images/
```

## Output

A list of typed table-region objects and an optional detected reference-section start page.

```python
tables, refs_start_page = discover_tables(root)
```

MinerU also creates an adapter-owned output directory under `mineru_out/`. A typical MinerU document directory contains:

```text
<document-name>/
  images/
  <document-name>_content_list.json
  <document-name>_content_list_v2.json
  <document-name>_layout.pdf
  <document-name>_middle.json
  <document-name>_model.json
  <document-name>_origin.pdf
  <document-name>.md
```

For the current workflow, the key file is `<document-name>_content_list.json`. It is the source used to identify table entries and locate each MinerU-generated table image through `img_path`.

## Module Contract

The current library contract exposes typed table regions. A table region should carry the following information:

```json
{
  "table_id": 1,
  "page_nr": 19,
  "image_path": "work/mineru/puurunen_2005/images/table_001.jpg",
  "mineru_img_path": "images/table_001.jpg",
  "bbox": [181, 60, 812, 130],
  "caption": [],
  "footnote": [],
  "table_body": "<table>...</table>",
  "in_references": false
}
```

## Default Implementation

The current implementation is the `tabulus.mineru` library module.

Python snippet:

```python
from pathlib import Path

from tabulus.mineru import discover_tables

root = Path.home() / "tabulus/work/mineru/puurunen_2005"

tables, refs_start_page = discover_tables(root)

print("Tables:", len(tables))
print("References start:", refs_start_page)

for table in tables:
    print(table.table_id, table.page_nr, table.image_path.name, table.caption)
```

Validated result for the tested Puurunen 2005 document:

```text
Tables: 23
```

The detected table regions began on page 6 and ended on page 22.

## Alternative Adapters

- MinerU
- Docling or another layout parser
- A custom PDF layout detector
- A table detector that emits crop images and structured table metadata

All adapters should produce the same table-image and structured-metadata contract.

## Standalone Run Target

The first validated development target is:

```text
Given an existing MinerU output directory, discover table regions and expose typed metadata without requiring GPU execution.
```

## Handoff To Table OCR

The next processing step is to create a clean table-crop collection for PaddleOCR-VL.

```text
PDF
  |
  v
MinerU
  |
  v
content_list.json
  |
  v
select entries where type == "table"
  |
  v
resolve each table img_path
  |
  v
copy or convert table images into the Tabulus table-crop collection
  |
  v
write tables_index.json
  |
  v
PaddleOCR-VL
```

The handoff stage should preserve enough provenance to trace every table image back to MinerU: page number, bounding box, `mineru_img_path`, caption, footnote, and `table_body` when available.

This handoff is implemented by:

```powershell
tabulus export-table-crops --mineru-root work/mineru/puurunen_2005 --out work/table_crops
```

The export writes:

```text
work/table_crops/
  tables_index.json
  images/
    page_006_table_001.jpg
```

The exporter preserves the original MinerU image extension instead of converting every crop to PNG. That keeps the library lightweight and avoids adding image-conversion dependencies before the PaddleOCR-VL adapter is implemented.

## Verification

The step succeeds when:

- `discover_tables(root)` returns table-region objects.
- Each table image path resolves to an existing MinerU-generated image.
- Each table has a table id and document page number.
- Bounding boxes, captions, and footnotes are preserved when the adapter provides them.
- The MinerU output directory is retained for debugging and traceability.
- `<document-name>_layout.pdf` can be inspected when layout detection looks suspicious.
- `table_body` is available when MinerU produced its own table reconstruction.
- Unit tests pass without requiring GPU execution.

## Common Failure Modes

| Failure | Likely cause | Fix |
| --- | --- | --- |
| File not found | Wrong path or manifest entry | Validate paths before processing. |
| MinerU output missing | MinerU failed or did not write `content_list.json` | Inspect `mineru_stderr.log` and `notes.md`. |
| No table regions | MinerU found no table entries or `img_path` resolution failed | Inspect `content_list.json` and `mineru_img_path` values. |
| Incorrect table crop | MinerU detected the wrong region or reading order | Inspect `<document-name>_layout.pdf` and compare the copied image with its `content_list.json` entry. |
| Weak structured table | MinerU `table_body` is incomplete or malformed | Compare `table_body` against PaddleOCR-VL reconstruction before deciding which output to trust. |
| Missing bbox or caption | Adapter did not provide optional metadata | Preserve `null` and continue. |

## Next Step

After typed PDF profiling and table-crop export, implement the PaddleOCR-VL adapter that consumes `tables_index.json`.
