# Step 4: Table OCR And Structure Extraction

## Goal

Turn table crop images into structured rows and columns.

## Input

`tables_index.json` and table crop images from the table-crop handoff directory.

The input is intentionally image-based. Table-reconstruction adapters consume canonical MinerU-generated table crops from the normalized Tabulus handoff; they do not independently process the original PDF to detect and crop tables for this comparison.

## Output

Adapter-neutral table OCR results. The implemented batch command writes per-adapter `native/`, `parsed/`, `predictions/`, and `batch_summary.json` artifacts under the crop handoff directory unless `--out` is provided.

This output is an intermediate reconstruction checkpoint, not the final CSV:

```text
adapter-native result
      |
      v
parsed rows / structured OCR
      |
      v
normalized reconstruction
      |
      v
prediction CSV
```

## Module Contract

See `data-contracts/ocr-tables-json.md`.

## Default Implementation

The new Tabulus library now includes the adapter contract, lazy registry, PaddleOCR-VL adapter, Chandra OCR 2 adapter, legacy-compatible parser layer, batch reconstruction layer, and output writer under `src/tabulus/table_ocr/`. It provides a batch CLI for table reconstruction, but it does not yet provide a full end-to-end pipeline command.

The component is model-independent: a Table OCR and Structure Extraction adapter consumes the normalized Tabulus table-crop handoff and returns a structured table result. The currently registered reconstruction adapters are `paddleocr-vl` and `chandra`; another table-reconstruction adapter can be substituted later if it accepts the same handoff and preserves the same MinerU provenance.

PaddleOCR-VL is more than ordinary OCR. Its current architecture performs layout analysis followed by vision-language-model recognition. The layout stage detects elements such as tables, crops them, determines reading order, and the VLM converts the elements into structured recognition results.

In the intended Tabulus workflow, MinerU has already isolated the table image during PDF profiling. PaddleOCR-VL therefore receives a cleaner input than it would receive from a full page and is evaluated as a table-reconstruction adapter, not as a competing full-document table detector.

The expected handoff from PDF profiling is a table-crop collection with image paths plus provenance: page number, bounding box, caption, footnote, original MinerU `img_path`, and MinerU `table_body` when available.

Conceptually, the adapter contract is:

```text
MinerU table crop
       |
       v
Table reconstruction adapter
       |
       v
structured table
```

It is not:

```text
original PDF
       |
       v
each OCR adapter independently detects and crops tables
```

The adapter stage should focus on extracting or reconstructing cell text, rows, columns, table structure, and adapter-native structured output while preserving the table ID and MinerU provenance supplied by the normalized handoff.

In the first clean workflow, the adapter stack is:

```text
table crop images
      |
      v
PaddleOCR-VL or Chandra OCR 2
      |
      v
HTML, Markdown, or structured table output
      |
      v
tables/ocr_tables.json
```

The diagram above shows the common adapter contract, not a restriction to one model family.

## Batch CLI

Run batch reconstruction from a canonical MinerU table-crop handoff:

```bash
tabulus reconstruct-tables \
  --crops "/path/to/tabulus-output/table-crops/<paper>" \
  --adapter paddleocr-vl \
  --device gpu:0
```

Chandra uses the same CLI contract:

```bash
tabulus reconstruct-tables \
  --crops "/path/to/tabulus-output/table-crops/<paper>" \
  --adapter chandra \
  --device gpu:0
```

For multiple papers, pass the parent table-crops directory:

```bash
tabulus reconstruct-tables \
  --crops-folder "/path/to/tabulus-output/table-crops" \
  --adapter chandra \
  --device gpu:0
```

If `--out` is omitted, the default output is:

```text
<crop-root>/
  reconstructions/
    <adapter>/
      native/
      parsed/
      predictions/
      batch_summary.json
```

For Chandra, that adapter directory is:

```text
<crop-root>/
  reconstructions/
    chandra/
      native/
      parsed/
      predictions/
      batch_summary.json
```

`native/` preserves the full adapter-neutral result and adapter provenance. `parsed/` preserves the rectangular parsed table representation and metadata. `predictions/` contains pre-reference-resolution CSV files for downstream processing and table-reconstruction evaluation. `batch_summary.json` records batch-level counts plus per-table status, runtime, artifact paths, and errors.

The batch layer reads `tables_index.json`, preserves the existing `table_id` values and crop order, and processes every physical MinerU crop independently. It reuses one adapter instance for the complete batch so heavyweight models can be initialized once. A table-level OCR error is preserved as an error result and later crops continue. Duplicate table IDs and adapters that change table identity are rejected.

For the full default output contract, filename semantics, and current rerun behavior, see {doc}`../data-contracts/run-directory`.

This command performs no reference-table classification, bibliography extraction, reference matching, DOI resolution, final resolved CSV generation, or continued-table merging.

## Practical Runtime Guidance

Approximate document-level processing times for the currently implemented
adapters are:

| Adapter | Hardware used in validation | Approximate processing time |
| --- | --- | --- |
| PaddleOCR-VL | NVIDIA L40S GPU | ~10-20 minutes per document |
| Chandra OCR 2 | NVIDIA L40S GPU | ~30-45 minutes per document |

These are rough engineering estimates, not benchmark results. Runtime varies
significantly with the number, size, and complexity of table crops in a
document. Model initialization, caching, hardware, and serving configuration
also affect runtime, so these values should be read only as approximate
order-of-magnitude guidance.

For PaddleOCR-VL, the estimate is derived from observed GPU crop-processing
times with the canonical MinerU table crop used directly and layout detection
disabled. It is not a formal document-level benchmark.

For Chandra OCR 2, the estimate applies to the in-process Hugging Face backend
used by Tabulus. Alternative serving configurations such as vLLM have not yet
been evaluated and may have different throughput.

Runtime observations do not imply anything about reconstruction accuracy or
which adapter is preferable.

## PaddleOCR-VL Configuration

Use precise Paddle naming:

- PaddleOCR 3.x is the overall PaddleOCR toolkit generation.
- PaddleOCR 3.7.0 is the package version used in the validated Windows CPU run.
- PaddleOCR-VL 1.6 / `PaddleOCR-VL-1.6-0.9B` is the document VLM used by Tabulus.

Tabulus initializes the first adapter as:

```python
PaddleOCRVL(
    pipeline_version="v1.6",
    device="cpu",
    engine="paddle",
    use_layout_detection=False,
)
```

`device` is configurable. The validated CPU configuration used `device="cpu"`; the validated Linux GPU configuration used `device="gpu:0"`.

For each canonical MinerU table crop, prediction uses:

```python
pipeline.predict(
    str(image_path),
    use_layout_detection=False,
    prompt_label="table",
)
```

MinerU has already localized and cropped the table, so `use_layout_detection=False` avoids rerunning a layout detector. `prompt_label="table"` tells PaddleOCR-VL that the crop is already a table. The adapter preserves PaddleOCR's native public JSON and Markdown result views as two serializations of the same inference result, not as independent predictions.

OCR and ML dependencies are optional and lazily loaded. Importing core Tabulus does not require PaddleOCR or PaddlePaddle; install those dependencies only in the hardware/model-specific environment that will run the adapter.

## Legacy-Compatible Parsing

The legacy Paddle `/ocr/images` table parsing behavior has been restored as a reusable parser layer.

The current preference order is:

1. Read PaddleOCR's native Markdown/HTML textual representation.
2. If one or more HTML `<table>...</table>` elements are present, parse them as HTML tables, preserve cell strings, expand `rowspan` and `colspan` into a rectangular matrix, pad shorter rows, and mark the parsed source as `"html"`.
3. Only if no HTML table is found, fall back to GitHub-style pipe-table Markdown parsing and mark the parsed source as `"markdown"`.

The generic parsed representation contains:

- `rows`
- `n_rows`
- `n_cols`
- `source`

This is not final scientific normalization. It restores the legacy rectangular row representation; semantic fill-down, section-row interpretation, formula rewriting, prediction CSV export, and reference resolution remain later stages.

## Validated Windows CPU Inference

A real PaddleOCR-VL CPU inference was validated on Windows 11 with:

- Python 3.12.10
- PaddlePaddle 3.2.1
- PaddleOCR 3.7.0
- PaddleOCR-VL 1.6 / `PaddleOCR-VL-1.6-0.9B`
- device: CPU
- engine: Paddle
- layout detection disabled
- table prompt enabled

The source table was the MinerU-generated crop `page_006_table_001.jpg` from `Puurunen - February 2005`. Paddle reported image dimensions of 1431 x 1923, inference status `ok`, one result object, one parsed table, parsed source `"html"`, and reconstructed dimensions of 58 rows x 6 columns.

The reconstructed header was:

```text
$ Z^{a} $ | Material | Reactant A | Reactant B | Substrate $ ^{b} $ | Refs.
```

Preserved scientific strings included `$ B_{2}O_{3} $`, `$ B_{x}P_{y}O_{z} $`, and `$ Al_{2}O_{3} $`.

The first CPU inference was very slow. Treat that as a performance observation only; CPU is validated for correctness but should not be assumed to be the recommended production mode.

## Validated Linux GPU Inference

A real PaddleOCR-VL GPU inference was validated on an NVIDIA L40S in a separate Conda environment from MinerU. The validated stack was:

- Python 3.12
- PaddlePaddle-GPU 3.2.1
- PaddleOCR 3.7.0
- PaddleOCR-VL 1.6
- device: `gpu:0`
- engine: Paddle
- layout detection disabled
- table prompt enabled

The validated input was again the canonical MinerU crop `page_006_table_001.jpg` from `Puurunen - February 2005`. PaddleOCR-VL was applied only to that crop, not to the original PDF. The run succeeded and produced one parsed HTML table with 58 rows x 6 columns.

A repeatability check using the same loaded adapter and the same crop produced:

```text
first cached-model pass: 44.58 s
warm second pass:       25.24 s
parsed table shape:     58 x 6 both times
parsed cell differences: 0
```

The first-ever GPU run took 91.97 s because it also included model download and setup. Treat these timings as observations from this validation, not as formal benchmarks.

The Windows CPU crop and Linux GPU crop were not byte-identical: the observed image dimensions were 1431 x 1923 on Windows CPU and 1432 x 1923 on Linux GPU. Do not make strong CPU-vs-GPU accuracy claims from differences between those outputs.

## Chandra OCR 2 Configuration

Chandra OCR 2 is implemented as a Tabulus table-reconstruction adapter under
`src/tabulus/table_ocr/chandra.py`. It uses the Hugging Face/in-process backend
for `datalab-to/chandra-ocr-2`, not the Chandra CLI or a vLLM server.

The implemented Chandra path is:

```text
canonical MinerU crop
      |
      v
Chandra OCR 2
      |
      v
raw structured HTML
      |
      v
Tabulus common span-aware HTML parser
      |
      v
parsed rectangular representation
      |
      v
prediction CSV
```

Chandra consumes canonical MinerU table crops directly. It does not redetect
tables or recrop the original PDF. Tabulus maps `--device gpu:0` to PyTorch
`cuda:0`, passes `prompt_type="ocr"`, loads one Chandra model instance, and
reuses that instance through the batch layer.

The adapter preserves generated raw HTML as native adapter evidence. It also
preserves Chandra metadata such as `token_count` and Chandra's generation error
status in the native result. The raw HTML is passed into the same
adapter-neutral parser used for other HTML-emitting reconstruction models, and
prediction CSV files follow the same output contract as PaddleOCR-VL.

The validated Chandra stack used:

- Chandra package: `chandra-ocr` 0.2.0
- model: `datalab-to/chandra-ocr-2`
- Python 3.12.13
- PyTorch 2.13.0+cu130
- Transformers 5.15.1
- GPU: NVIDIA L40S
- prompt: `prompt_type="ocr"`

The first direct feasibility validation was performed before adapter
integration. It used the same canonical MinerU table crop through Chandra's
in-process API. The model loaded successfully on `cuda:0`; first model
load/download took 111.40 s, inference took 134.79 s, the run generated 3838
tokens, peak allocated inference GPU memory was 9.35 GiB, and `result.error`
was `False`.

After Chandra was integrated into Tabulus, a real CLI smoke test using one
canonical MinerU crop with `--adapter chandra --device gpu:0` completed with:

```text
status: ok
result_count: 1
structured HTML tables: 1
parsed shape: 65 x 6
prediction CSVs: 1
token_count: 3838
Chandra generation error: false
```

Chandra emitted meaningful HTML `rowspan` attributes, which exposed a
limitation in the common parser; that parser limitation has been fixed
separately.

Do not treat this single-table validation as evidence that PaddleOCR-VL or
Chandra is more accurate.

## Xberg Direct Validation

Xberg is the successor/rebrand of Kreuzberg. It has been directly validated as
a table-reconstruction candidate, but it is not yet registered as a Tabulus
`table_ocr` adapter. Do not select Xberg through `tabulus reconstruct-tables`
until an adapter has been implemented.

The direct validation used:

- Python 3.12.13
- `xberg` 1.0.14
- Tesseract 5.5.2
- canonical MinerU table crop input
- Xberg OCR backend: Tesseract
- Xberg layout detection enabled
- layout model: RT-DETR v2
- table structure model: TATR
- CPU execution
- cache disabled for the measured TATR run

The tested path was:

```text
canonical MinerU crop
      |
      v
Xberg
      |
      +-- Tesseract OCR
      |
      +-- RT-DETR layout detection
      |
      v
TATR table structure recognition
      |
      v
Xberg Table.cells / Table.markdown
```

Xberg consumed the existing canonical MinerU crop and did not redetect or crop
tables from the original PDF for the Tabulus comparison.

For `Puurunen - February 2005 / page_006_table_001.jpg`, the Tesseract-only
Xberg run produced:

```text
results: 1
errors: 0
structured tables: 0
OCR content length: 1380
wall time: approximately 2.8 s
```

With layout detection plus TATR enabled, the same crop produced:

```text
results: 1
errors: 0
layout regions: 1
detected region class: table
table-region confidence: 0.9471566081047058
structured tables: 1
reconstructed shape: 39 rows x 6 columns
wall time: approximately 11.1 s
```

The structured result demonstrates feasibility, not accuracy superiority. The
OCR output contains noticeable chemistry transcription errors, so quality must
be measured against the reconstruction gold standard before drawing conclusions.

## Alternative Adapters

- PaddleOCR-VL -- implemented
- Chandra -- implemented
- DeepSeek OCR -- future
- Xberg -- future
- NuExtract3 -- future

These are alternative table-reconstruction adapters, not sequential pipeline stages.

## Evaluation Question

The initial validation should preserve the first clean comparison against MinerU's own structured table output:

```text
MinerU table_body

versus

MinerU crop -> PaddleOCR-VL reconstruction
```

Modern MinerU may be sufficient for some table classes. The second model should remain a measured choice, not an assumption.

For each table, keep both outputs during evaluation:

- MinerU `table_body`
- PaddleOCR-VL reconstruction from the MinerU-generated crop image

The extended adapter benchmark is the more general version of the same question:

```text
MinerU table_body
versus
MinerU crop -> PaddleOCR-VL
versus
MinerU crop -> DeepSeek OCR
versus
MinerU crop -> Chandra
versus
MinerU crop -> Xberg
versus
MinerU crop -> NuExtract3
```

The pipeline can later decide whether to use the lighter MinerU output directly for some table classes or route crops through one of the implemented or future table-reconstruction adapters.

## Verification

The step succeeds when each crop has an OCR result object, even if the OCR result is empty or contains an error.
