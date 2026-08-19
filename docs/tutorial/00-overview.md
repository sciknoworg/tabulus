# Core Pipeline Overview

The pipeline begins with one input scientific PDF paper and ends with resolved CSV files whose table reference cells are linked to DOI values where possible.

Each step should be implemented as a standalone processing component with a small, explicit contract. Libraries such as MinerU, PaddleOCR-VL, GROBID, Kreuzberg, Crossref, or future alternatives should be adapters behind these component boundaries.

```{figure} ../_static/pipeline.png
:alt: Tabulus pipeline workflow
:width: 720px
:align: center
```

## First Clean Workflow

This is the first practical workflow to stabilize: run MinerU as the first document-analysis adapter, inspect its outputs with the new Tabulus library, then reconstruct each table through the Table OCR and Structure Extraction component, with PaddleOCR-VL as the first/default adapter.

```text
Scientific PDF
      |
      v

+----------------------+
| Module 1             |
| PDF Profiling        |
|                      |
| - page/layout        |
|   analysis           |
| - detect tables      |
| - determine table    |
|   bbox               |
| - save table crops   |
| - captions/footnotes |
| - structured JSON    |
+----------------------+
| Default adapter:     |
| MinerU 3.4.5         |
+----------------------+
      |
      | table images
      | structured JSON
      v

+----------------------+
| PaddleOCR-VL 1.6     |
| PaddleOCR 3.7.0      |
|                      |
| - read each table    |
| - recognize cells    |
| - understand         |
|   structure          |
| - reconstruct table  |
| - Markdown/structured|
|   output             |
+----------------------+
      |
      v

saved outputs
```

The first module is intentionally named for the processing work, not only the library. In the current clean Tabulus workflow, MinerU is the selected/canonical table-localization and crop-generation stage because prior experimental evaluation found it to be the strongest and most efficient choice for that role. This is a Tabulus design decision for the current pipeline, not a universal claim that MinerU is always superior for every dataset.

The red output between the two modules is the important contract: canonical MinerU table crops plus structured metadata. In the current validated library code, Tabulus does not crop the PDF from the bounding box itself. MinerU already generates the table crop image; Tabulus reads MinerU's `content_list.json`, filters entries with `type == "table"`, resolves each table's `img_path`, and exposes typed table-region objects.

Minimal standalone call using the current library:

```python
from pathlib import Path

from tabulus.mineru import discover_tables

tables, refs_start_page = discover_tables(Path("work/mineru/puurunen_2005"))

print(len(tables), refs_start_page)
```

This library call consumes existing MinerU output. The CLI can also launch MinerU with `tabulus profile` and now exports the canonical table-crop handoff automatically by default. The standalone `tabulus export-table-crops` command remains useful when an expensive MinerU run should be reused without touching the native MinerU output. PaddleOCR-VL is now the first implemented table-reconstruction adapter in the new library.

The component is Table OCR and Structure Extraction. PaddleOCR-VL is the first/default table-reconstruction adapter for that component, not a permanent hard-coded dependency. Other compatible adapters can consume the same normalized table-crop handoff if they preserve the table identifier, MinerU provenance, and structured output contract.

## Comparison To Evaluate

The first clean workflow should compare two table extraction paths scientifically:

```text
MinerU table_body

versus

MinerU crop -> PaddleOCR-VL reconstruction
```

Modern MinerU may be good enough for some tables. The second model should not be assumed necessary until table quality, structure preservation, and runtime are measured.

The broader experimental design treats table reconstruction as a plug-and-play adapter comparison. MinerU provides both the canonical table crop through `img_path` and its own structured `table_body`. The same MinerU-generated crop can be sent independently to PaddleOCR-VL, DeepSeek OCR, Chandra, Kreuzberg, or NuExtract3, while MinerU `table_body` remains a parallel native reconstruction candidate. Those adapters should not independently process the original PDF to locate or crop tables in this comparison; fixing the crop input controls the table-detection variable.

```text
Scientific PDF
      |
      v
    MinerU
      |
      +-- table localization
      +-- table crop generation
      +-- MinerU table_body
              |
              +-------------------------------+
              |                               |
       canonical table crop             table_body
              |                               |
   +----------+----------+----------+---------+---------+
   |          |          |          |         |         |
   v          v          v          v         v         v
PaddleOCR  DeepSeek   Chandra   Kreuzberg  NuExtract3 MinerU
   VL                                             table_body
   |          |          |          |         |         |
   v          v          v          v         v         v
table reconstruction candidates / structured tables
              |
              v
     normalize representations
              |
              v
       compare predictions
              |
              v
       ground-truth CSV
```

All six outputs may use different adapter-native output formats, so Tabulus should normalize them into a common table representation before comparison against the same ground-truth CSV. The evaluation should determine whether an external table-reconstruction adapter improves over MinerU's own `table_body`, and which adapter offers the best quality/runtime tradeoff for different table classes.

## Ordered Steps

1. PDF profiling
2. Table OCR and structure extraction
3. Table normalization
4. Reference-table classification
5. Bibliography extraction
6. Reference matching
7. DOI resolution
8. Resolved CSV export
9. Run report and QA bundle

## Component Rule

Every step should be able to run in two modes:

- **Standalone mode:** process files from disk and write its own outputs.
- **Pipeline mode:** receive the previous step's contract output and return the next contract output.

## Current Code Mapping

| Pipeline step | Current implementation area | Notes |
| --- | --- | --- |
| PDF profiling | `src/tabulus`, `tabulus.mineru` | Current new-library module can launch MinerU, read existing MinerU outputs, discover table regions, resolve image paths, preserve provenance, and return typed `TableRegion` objects. |
| MinerU execution | `tabulus profile` | Tested with MinerU 3.4.5 on Windows CPU using `pipeline` and on a GPU server using `hybrid-engine`; Windows unit tests cover command construction, default output paths, and fallback behavior. |
| Table-crop export | `tabulus profile`, `tabulus export-table-crops` | `tabulus profile` exports canonical MinerU table crops automatically by default. The standalone command regenerates the normalized handoff from existing MinerU output without rerunning MinerU. |
| Table OCR | `src/tabulus/table_ocr` | Adapter contract and lazy registry are implemented. PaddleOCR-VL is the first adapter; Windows CPU inference has been validated on a MinerU crop. |
| Reference processing | Not yet implemented in the new library | Target adapters include GROBID, Kreuzberg, and Crossref. |

## Tutorial Template

Every step page follows this structure:

- Goal
- Input
- Output
- Module contract
- Default implementation
- Alternative adapters
- Standalone run target
- Verification
- Common failure modes
- Next step
