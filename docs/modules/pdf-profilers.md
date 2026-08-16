# PDF Profilers

PDF profilers analyze scientific PDFs or existing document-analysis outputs and expose table regions plus provenance.

In the clean Tabulus workflow, PDF profiling is the first major digitization module. It is not a separate upload step and not only a metadata check. At the current library stage, the implemented profiling code consumes MinerU outputs that were produced by a separate MinerU CLI run.

## Target Responsibility

- Validate that the input exists and is a PDF.
- Create the run directory if needed.
- Copy or link the source PDF if the run layout requires it.
- Perform page/layout analysis.
- Detect tables.
- Determine table bounding boxes.
- Save table crop images.
- Capture captions and footnotes.
- Write structured JSON.

## Current Implemented Scope

The new installable library currently implements typed access to existing MinerU outputs through `tabulus.mineru`.

It:

- recursively finds `*_content_list.json`
- loads the structured content representation
- selects entries where `type == "table"`
- resolves table image paths
- converts zero-based MinerU `page_idx` values into document page numbers
- preserves `bbox`, captions, footnotes, and `table_body`
- optionally marks regions after a detected bibliography heading
- returns typed `TableRegion` objects

This code is covered by unit tests and does not require GPU execution.

## Default Tooling

Use Python standard library tools:

- `pathlib`
- `shutil`
- `hashlib`

The current adapter is MinerU.

The current library entry point is:

```python
from pathlib import Path
from tabulus.mineru import discover_tables

tables, refs_start_page = discover_tables(Path("work/mineru/puurunen_2005"))
```

This reads MinerU outputs and returns typed table regions. It does not launch MinerU, copy images, convert JPG to PNG, or write `tables_index.json`.

It does not crop the PDF from bounding boxes itself. MinerU has already generated the crop image.

The legacy repository still contains older service and benchmark runners that copy table images and write table-index files. Treat those as previous implementation areas until the new library grows equivalent commands.

## MinerU Source Outputs

Keep the complete MinerU output directory for traceability and debugging. The files most relevant to Tabulus are:

| File | Role |
| --- | --- |
| `<document-name>_content_list.json` | Primary source for table entries, `img_path`, `bbox`, captions, footnotes, and `table_body`. |
| `<document-name>_layout.pdf` | Visual debugging file for detected regions, bounding boxes, and reading order. |
| `<document-name>_middle.json` | Lower-level parsing detail for investigating layout failures. |
| `<document-name>_model.json` | Rawer model inference output for advanced debugging. |
| `<document-name>.md` | Quick human-readable reconstruction of the parsed document. |
| `images/` | MinerU-generated images, including table images referenced by `img_path`. |

The current stable downstream interface should remain `content_list.json` plus the referenced table images. `content_list_v2.json` is a candidate future interface once the current workflow is stable.

## Not Yet Implemented

The new library does not yet implement:

- MinerU process launching
- table JPG to PNG export
- `tables_index.json` generation
- PaddleOCR-VL execution
- reference processing
- a full end-to-end process command

## Adapter Ideas

- MinerU
- Docling or another document layout parser
- Custom table detector plus crop exporter
- Future web-upload adapter that produces the same file contract
