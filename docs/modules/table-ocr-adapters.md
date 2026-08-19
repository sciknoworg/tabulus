# Table OCR Adapters

Table OCR and Structure Extraction adapters convert normalized table crop images into structured tables.

In the current clean Tabulus workflow, MinerU is the canonical table-localization and crop-generation stage. Prior experimental work selected MinerU for this role because it was the strongest and most efficient option for locating scientific tables and producing useful crop images. The table OCR adapter layer is extensible, but its contract starts from the MinerU-generated crop rather than the original PDF.

## Responsibility

- Accept normalized table crop images through the batch layer or one crop directly through the adapter interface.
- Receive and retain the Tabulus table identifier and provenance.
- Reconstruct table structure and content.
- Return a structured result.
- Preserve adapter-native output where useful.
- Explicitly record empty or error results.
- Never silently drop failed tables.

Conceptually:

```text
MinerU table crop
       |
       v
Table reconstruction adapter
       |
       v
structured table
```

not:

```text
original PDF
       |
       v
each OCR adapter independently detects and crops tables
```

The adapter should focus on cell text, rows, columns, table structure, and adapter-native structured output while preserving the table ID and MinerU provenance supplied by the normalized Tabulus handoff.

Adapter-native outputs may differ. A table-reconstruction adapter may produce HTML, Markdown, CSV-like structures, JSON, or model-specific structured output. Downstream Tabulus code should normalize those outputs into the same common table representation before evaluation or export.

## Current Adapter

The clean library now implements the adapter boundary and the first PaddleOCR-VL adapter in `src/tabulus/table_ocr/`.

Key modules:

- `base.py`
- `batch.py`
- `output.py`
- `registry.py`
- `paddleocr_vl.py`
- `parsing.py`

The main abstractions are:

- `TableOCRInput`: one normalized table crop plus table id and provenance.
- `TableOCRResult`: adapter-neutral result for one crop, including status, native outputs, device, versions, and error text when needed.
- `TableOCRCapabilities`: static CPU/GPU support metadata.
- `TableOCRAdapter`: protocol implemented by reconstruction adapters.
- adapter registry: lists adapters and lazy-loads implementation classes.
- `run_table_ocr_batch`: loads `tables_index.json`, reuses one adapter instance, and writes reconstruction artifacts.

PaddleOCR-VL is the first/default adapter implemented in the clean library. It runs on already-isolated MinerU table crops with PaddleOCR layout detection disabled and the table prompt enabled.

PaddleOCR-VL has been validated on CPU and on an NVIDIA L40S GPU for a single canonical MinerU crop. The validated GPU configuration used PaddlePaddle-GPU 3.2.1, PaddleOCR 3.7.0, PaddleOCR-VL 1.6, `device="gpu:0"`, and `engine="paddle"`.

## Batch Reconstruction CLI

The installed `tabulus` entry point exposes table reconstruction as a subcommand:

```bash
tabulus reconstruct-tables \
  --crops "/path/to/tabulus-output/table-crops/<paper>" \
  --adapter paddleocr-vl \
  --device gpu:0
```

The default output for one adapter is:

```text
<crop-root>/
  reconstructions/
    <adapter>/
      native/
      parsed/
      predictions/
      batch_summary.json
```

The command reads `tables_index.json`, preserves table IDs and crop order, processes each physical MinerU crop independently, and reuses one adapter instance for the full batch. Failed tables are written as explicit error results and do not prevent later crops from running. The command rejects duplicate table IDs and adapter results that change table identity.

For the full default output contract, filename semantics, and current rerun behavior, see {doc}`../data-contracts/run-directory`.

Prediction CSV files are pre-reference-resolution artifacts. This command does not classify reference tables, extract bibliographies, match references, resolve DOI values, write final resolved CSV files, or merge continued tables.

OCR and ML dependencies are optional and lazily loaded. Importing core Tabulus or listing registered adapters does not require PaddleOCR or PaddlePaddle. Hardware/model-specific environments can therefore remain separate from the lightweight core Tabulus environment.

## Other Experiment Adapters

Other experiment adapters already represented in the repository include:

- DeepSeek OCR
- Chandra
- Kreuzberg
- NuExtract3

These belong behind the same Table OCR and Structure Extraction contract. They are alternative table-reconstruction adapters, not stages that run after PaddleOCR-VL.

MinerU `table_body` is slightly different: it is produced during PDF profiling rather than by consuming the normalized table-crop handoff. It should still satisfy the same downstream comparison and normalization concept as another candidate table reconstruction.
