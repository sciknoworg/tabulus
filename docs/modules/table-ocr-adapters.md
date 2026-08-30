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

## Current Adapters

The clean library implements the adapter boundary and the current registered reconstruction adapters in `src/tabulus/table_ocr/`.

Key modules:

- `base.py`
- `batch.py`
- `output.py`
- `registry.py`
- `paddleocr_vl.py`
- `chandra.py`
- `nuextract3.py`
- `parsing.py`

The main abstractions are:

- `TableOCRInput`: one normalized table crop plus table id and provenance.
- `TableOCRResult`: adapter-neutral result for one crop, including status, native outputs, device, versions, and error text when needed.
- `TableOCRCapabilities`: static CPU/GPU support metadata.
- `TableOCRAdapter`: protocol implemented by reconstruction adapters.
- adapter registry: lists adapters and lazy-loads implementation classes.
- `run_table_ocr_batch`: loads `tables_index.json`, reuses one adapter instance, and writes reconstruction artifacts.

The current registered crop-consuming reconstruction adapters are:

- `paddleocr-vl`: PaddleOCR-VL reconstruction on already-isolated MinerU table crops with PaddleOCR layout detection disabled and the table prompt enabled.
- `chandra`: Chandra OCR 2 reconstruction through the Hugging Face/in-process API with `prompt_type="ocr"`.
- `nuextract3`: NuExtract3 reconstruction through Hugging Face Transformers in document-to-Markdown mode.

PaddleOCR-VL and Chandra report CPU and GPU support in the registry. NuExtract3 is registered as GPU-only in the validated Tabulus configuration.

MinerU `table_body` is also a reconstruction candidate, but it is produced during PDF profiling rather than by a crop-consuming `tabulus.table_ocr` adapter.

For the external tools as used by Tabulus, see:

- {doc}`../external-tools/paddleocr-vl`
- {doc}`../external-tools/chandra`
- {doc}`../external-tools/nuextract3`

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

OCR and ML dependencies are optional and lazily loaded. Importing core Tabulus or listing registered adapters does not require PaddleOCR, PaddlePaddle, Chandra, PyTorch, Transformers, or other heavyweight adapter runtimes. Hardware/model-specific environments can therefore remain separate from the lightweight core Tabulus environment.

## Other Candidate Adapters

Other reconstruction candidates can be added behind the same Table OCR and Structure Extraction contract. DeepSeek OCR remains future work in the rebuilt installable library.

These candidates are alternatives, not sequential stages that run after another reconstruction adapter.
