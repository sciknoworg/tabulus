# Table Reconstruction Adapters

Table reconstruction adapters convert normalized table crop images into structured tables.

The package path is still `tabulus.table_ocr` because that is the current code API, but the adapter layer is broader than conventional OCR. An adapter may use OCR, a document vision-language model, or another reconstruction architecture as long as it accepts the same canonical MinerU crop input and returns the common result contract.

In the rebuilt Tabulus workflow, MinerU is the canonical table-localization and crop-generation stage. The reconstruction adapter contract starts from the MinerU-generated crop rather than the original PDF.

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
each reconstruction adapter independently detects and crops tables
```

The adapter should focus on cell text, rows, columns, table structure, and adapter-native structured output while preserving the table ID and MinerU provenance supplied by the normalized Tabulus handoff.

Adapter-native outputs may differ. A table-reconstruction adapter may produce HTML, Markdown, CSV-like structures, JSON, or model-specific structured output. Downstream Tabulus code normalizes those outputs into the same common table representation before evaluation or later processing.

## Current Adapters

The rebuilt library implements the adapter boundary and the current registered reconstruction adapters in `src/tabulus/table_ocr/`.

Key modules:

- `base.py`
- `batch.py`
- `output.py`
- `registry.py`
- `paddleocr_vl.py`
- `chandra.py`
- `nuextract3.py`
- `tesseract_tatr.py`
- `tatr_postprocess.py`
- `rapidocr_tableformer.py`
- `granite_vision_table.py`
- `trivia.py`
- `glm_ocr.py`
- `dolphin_v2.py`
- `deepseek_ocr_2.py`
- `nanonets_ocr_s.py`
- `monkeyocrv2_b_parsing.py`
- `nemotron_parse_v1_2.py`
- `hunyuanocr_1_5.py`
- `dots_mocr.py`
- `parsing.py`

The main abstractions are:

- `TableOCRInput`: one normalized table crop plus table id and provenance.
- `TableOCRResult`: adapter-neutral result for one crop, including status, native outputs, device, versions, and error text when needed.
- `TableOCRCapabilities`: static CPU/GPU support metadata.
- `TableOCRAdapter`: protocol implemented by reconstruction adapters.
- adapter registry: lists adapters and lazy-loads implementation classes.
- `run_table_ocr_batch`: loads `tables_index.json`, reuses one adapter instance, and writes reconstruction artifacts.

The current registered crop-consuming reconstruction adapters are exactly:

- `paddleocr-vl`: PaddleOCR-VL reconstruction on already-isolated MinerU table crops with PaddleOCR layout detection disabled and the table prompt enabled.
- `chandra`: Chandra OCR 2 reconstruction through the Hugging Face/in-process API with `prompt_type="ocr"`.
- `nuextract3`: NuExtract3 reconstruction through Hugging Face Transformers in document-to-Markdown mode.
- `tesseract-tatr`: Tesseract OCR word recognition plus Microsoft Table Transformer structure recognition, fused deterministically into an HTML table.
- `rapidocr-tableformer`: RapidOCR with ONNX Runtime for OCR and word boxes, combined with Docling TableFormer V1 structure recognition on the complete canonical crop.
- `granite-vision-table`: Granite Vision 4.1 4B receives the complete canonical crop directly, generates `<tables_otsl>`, and uses Docling's Granite OTSL parser for structure and cell reconstruction.
- `trivia`: TRivia-3B receives the complete canonical crop directly, generates native OTSL, and uses Tabulus-owned deterministic OTSL-to-HTML normalization before the shared parser.
- `glm-ocr`: GLM-OCR receives the complete canonical crop directly, generates native HTML, and passes clean generated HTML through the shared span-aware parser without GLM-specific structural normalization.
- `dolphin-v2`: Dolphin-v2 receives the complete canonical crop directly, applies deterministic model-input resizing, generates native HTML with the `ByteDance/Dolphin-v2` checkpoint, and passes clean generated HTML through the shared span-aware parser without Dolphin-specific semantic repair.
- `deepseek-ocr-2`: DeepSeek-OCR-2 receives the complete canonical crop directly, uses model-internal dynamic-resolution tiling, and passes the model-specific `infer(...)` output unchanged through the shared HTML/Markdown parser.
- `nanonets-ocr-s`: Nanonets-OCR-s receives the complete canonical crop directly, generates native structured HTML with the `nanonets/Nanonets-OCR-s` checkpoint, and passes clean generated HTML through the shared span-aware parser without Nanonets-specific semantic repair.
- `monkeyocrv2-b-parsing`: MonkeyOCRv2-B-Parsing receives the complete canonical crop directly, uses direct single-task table recognition to generate native OTSL, and passes deterministic OTSL-to-HTML output through the shared parser without semantic repair.
- `nemotron-parse-v1-2`: NVIDIA Nemotron Parse v1.2 receives the complete canonical crop directly, generates grounded semantic objects with Table-class LaTeX/tabular content, and passes NVIDIA-postprocessed HTML through the shared parser without using generated bounding boxes for recropping.
- `hunyuanocr-1-5`: HunyuanOCR-1.5 receives the complete canonical crop directly, uses the official table task to generate native HTML, preserves HunyuanOCR repetition safeguards as inference controls, and passes clean HTML through the shared parser without HunyuanOCR-specific semantic repair.
- `dots-mocr`: dots.mocr receives the complete canonical crop directly, uses its active layout prompt to generate JSON layout output with model-emitted Table objects, preserves their bounding boxes as provenance only, and passes their HTML through the shared parser without JSON repair or semantic repair.

PaddleOCR-VL, Chandra, Tesseract + Table Transformer, and RapidOCR + Docling TableFormer report CPU and GPU support in the registry. RapidOCR itself runs on CPU while its TableFormer component uses the requested device. NuExtract3, Granite Vision 4.1 4B, TRivia-3B, GLM-OCR, Dolphin-v2, DeepSeek-OCR-2, Nanonets-OCR-s, MonkeyOCRv2-B-Parsing, NVIDIA Nemotron Parse v1.2, HunyuanOCR-1.5, and dots.mocr are registered as GPU-only in the validated Tabulus configuration.

MinerU `table_body` is also a reconstruction candidate, but it is produced during PDF profiling rather than by a crop-consuming `tabulus.table_ocr` adapter.

For the external tools as used by Tabulus, see:

- {doc}`../external-tools/paddleocr-vl`
- {doc}`../external-tools/chandra`
- {doc}`../external-tools/nuextract3`
- {doc}`../external-tools/tesseract-tatr`
- {doc}`../external-tools/docling`
- {doc}`../external-tools/granite-vision`
- {doc}`../external-tools/trivia`
- {doc}`../external-tools/glm-ocr`
- {doc}`../external-tools/dolphin-v2`
- {doc}`../external-tools/deepseek-ocr-2`
- {doc}`../external-tools/nanonets-ocr-s`
- {doc}`../external-tools/monkeyocrv2-b-parsing`
- {doc}`../external-tools/nemotron-parse-v1-2`
- {doc}`../external-tools/hunyuanocr-1-5`
- {doc}`../external-tools/dots-mocr`

## Batch Orchestration

The installed `tabulus` entry point exposes the batch layer through:

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

The batch layer reads `tables_index.json`, preserves table IDs and crop order, processes each physical MinerU crop independently, and reuses one adapter instance for the full batch. Failed tables are written as explicit error results and do not prevent later crops from running. Duplicate table IDs and adapter results that change table identity are rejected.

For multiple crop roots, the CLI processes papers sequentially and keeps output isolated by paper. Adapter instances are still created once per command invocation and reused across the selected batch.

## Artifact Separation

Every adapter writes the same Tabulus-owned reconstruction layers:

```text
native/
parsed/
predictions/
batch_summary.json
```

`native/` preserves adapter-native evidence. `parsed/` preserves the common structured representation produced by the shared parser. `predictions/` contains raw reconstruction CSVs before reference resolution. `batch_summary.json` is the reconstruction batch manifest.

A prediction CSV is written only when the adapter result status is `ok` and exactly one structured table was parsed from the canonical crop. Empty results and multiple-table ambiguity preserve native and parsed evidence without writing a CSV.

For the full default output contract, filename semantics, and current rerun behavior, see {doc}`../data-contracts/run-directory`.

Prediction CSV files are pre-reference-resolution artifacts. Reconstruction does not classify reference tables, extract bibliographies, match references, resolve DOI values, write final resolved CSV files, or merge continued tables.

ML dependencies are optional and lazily loaded. Importing core Tabulus or listing registered adapters does not require PaddleOCR, PaddlePaddle, Chandra, Tesseract, TRivia, GLM-OCR, Dolphin-v2, DeepSeek-OCR-2, Nanonets-OCR-s, MonkeyOCRv2-B-Parsing, NVIDIA Nemotron Parse v1.2, HunyuanOCR-1.5, dots.mocr, PyTorch, Transformers, or other heavyweight adapter runtimes. Hardware/model-specific environments can therefore remain separate from the lightweight core Tabulus environment.

## Other Candidate Adapters

Other reconstruction candidates can be added behind the same reconstruction contract.

These candidates are alternatives, not sequential stages that run after another reconstruction adapter.
