# Current State

Tabulus is being reorganized into an installable Python library using a standard `src/tabulus` package structure.

The first implemented library module provides MinerU-backed PDF profiling utilities. It can launch MinerU through the `tabulus profile` command, and it can also inspect an existing MinerU document-specific output directory.

## Validated Library Behavior

The current `tabulus.mineru` module:

- selects a CPU-compatible or GPU-backed MinerU backend
- distinguishes the profiler (`mineru`) from MinerU backends (`pipeline` and `hybrid-engine`)
- constructs and runs non-interactive MinerU commands
- writes default profiling output under `<PDF directory>/tabulus-output/<profiler>/<backend>/` when `--out` is omitted
- writes MinerU stdout, stderr, and run metadata logs
- recursively locates MinerU `*_content_list.json` files
- loads MinerU's structured content representation
- identifies entries where `type == "table"`
- resolves each table entry's associated image file
- converts MinerU's zero-based `page_idx` values into document page numbers
- preserves bounding boxes, captions, footnotes, and `table_body`
- optionally marks table regions that occur after a detected bibliography heading
- exposes typed `TableRegion` objects
- exports canonical MinerU table crops automatically by default after a successful `tabulus profile` run

The current `tabulus export-table-crops` command:

- consumes an existing MinerU output directory
- copies only discovered table images into an `images/` handoff directory
- preserves the source image extension
- writes a normalized `tables_index.json`
- keeps MinerU provenance, including original `img_path`, source image path, page number, bounding box, captions, footnotes, `table_body`, and reference-section status

The current `tabulus.table_ocr` package:

- defines `TableOCRInput`, `TableOCRResult`, `TableOCRCapabilities`, and the `TableOCRAdapter` protocol
- provides an adapter registry with lazy loading
- implements PaddleOCR-VL, Chandra OCR 2, NuExtract3, Tesseract + Table Transformer, and RapidOCR + Docling TableFormer adapters for MinerU-generated table crops
- provides `tabulus.table_ocr.batch` for adapter-neutral batch reconstruction
- provides `tabulus.table_ocr.output` for native, parsed, and prediction artifact writing
- initializes PaddleOCR-VL with layout detection disabled
- predicts with `prompt_label="table"`
- preserves PaddleOCR native JSON and Markdown result views
- invokes Chandra OCR 2 through the Hugging Face/in-process path with `prompt_type="ocr"`
- invokes NuExtract3 through Hugging Face Transformers in Markdown mode with deterministic generation
- invokes Tesseract for OCR word tokens and Microsoft Table Transformer for structure recognition, then fuses tokens and structure deterministically
- invokes RapidOCR with ONNX Runtime for CPU OCR and word boxes, then uses Docling TableFormer V1 in accurate mode with cell matching on the requested device
- preserves RapidOCR/Docling native OTSL and table-structure evidence before shared parsing
- restores the legacy HTML-first, Markdown-fallback row parser
- records explicit `ok`, `empty`, or `error` statuses instead of silently dropping tables

The current `tabulus reconstruct-tables` command:

- reads a canonical `tables_index.json` handoff
- preserves existing `table_id` values and crop order
- reuses one adapter instance across the complete crop batch
- processes every physical MinerU crop independently
- writes `native/`, `parsed/`, `predictions/`, and `batch_summary.json`
- preserves table-level errors and continues with later crops
- rejects duplicate table IDs and adapters that change table identity
- does not perform reference classification, bibliography extraction, reference matching, DOI resolution, final resolved CSV export, or continued-table merging

The current `tabulus.reference_tables` package and `tabulus classify-reference-tables` command:

- classify reconstruction outputs for reference-like scientific table content
- consume `batch_summary.json` and the common parsed reconstruction artifacts
- write `reference_table_classification.json` beside the reconstruction artifacts by default
- preserve `native/`, `parsed/`, `predictions/`, and `batch_summary.json`
- retain independent classification decisions separately from continuation-inherited decisions
- record matched header evidence, matched citation evidence, classification source, continuation provenance, and reason text
- do not perform bibliography extraction, reference matching, DOI resolution, final resolved CSV export, or continued-table merging

The behavior is covered by unit tests that do not require GPU execution.

The current tests verify:

- content-list discovery
- table extraction
- page and provenance handling
- reference-section detection
- missing-output error handling
- backend selection and MinerU command construction
- default profiling output path generation
- mocked MinerU execution logging
- table-crop export and missing-source-image errors
- profile-driven automatic table-crop export
- table reconstruction registry lazy loading
- PaddleOCR-VL adapter configuration and result preservation
- Chandra OCR 2 adapter configuration and result preservation
- NuExtract3 adapter configuration, GPU-only device handling, and runtime reuse
- Tesseract + Table Transformer registry metadata, device handling, runtime reuse, native evidence preservation, TSV parsing, HTML generation, and empty-result handling
- legacy-compatible HTML/Markdown table parsing
- batch table-reconstruction input loading and error handling
- native, parsed, prediction CSV, and batch-summary output writing
- reference-table classification heuristics and manifest writing
- continued-table classification inheritance without file merging

## Validated Execution

MinerU 3.4.5 has been tested through two profiling paths:

- Windows CPU-only setup with Python 3.12.10, CPU-only PyTorch 2.10.0+cpu, CUDA unavailable, and MinerU 3.4.5 `pipeline`
- Linux GPU-server setup with Python 3.12, a dedicated Conda environment, and MinerU `hybrid-engine`

The tested GPU workflow is:

```text
PDF
  |
  v
MinerU 3.4.5
  |
  v
MinerU structured output
  |
  v
tabulus.mineru
  |
  v
typed TableRegion objects
```

For the tested 53-page Puurunen 2005 GPU run, the library found 23 tables. Detected table regions began on page 6 and ended on page 22.

On Windows, MinerU 3.4.5 `pipeline` completed a real 53-page PDF profiling run with Python 3.12.10, PyTorch 2.10.0+cpu, and `torch.cuda.is_available()` returning `False`.

MinerU 3.4.5 required the additional `six` package in that Windows pipeline installation because bundled MinerU OCR code imports `six`, but `six` was not installed automatically by the `mineru[pipeline]` dependency set. Treat this as a MinerU 3.4.5 compatibility workaround, not a Tabulus dependency or a claim about later MinerU releases.

PaddleOCR-VL CPU inference has been validated on a real MinerU-generated crop from `Puurunen - February 2005`:

```text
PaddlePaddle 3.2.1
PaddleOCR 3.7.0
PaddleOCR-VL 1.6 / PaddleOCR-VL-1.6-0.9B
device: CPU
engine: Paddle
layout detection: disabled
prompt label: table
```

The validated crop was `page_006_table_001.jpg`; Paddle reported image dimensions 1431 x 1923, status `ok`, one result object, one parsed HTML table, and a 58 x 6 rectangular row representation. The first CPU inference was very slow, so treat this as a correctness validation and performance observation, not a production recommendation.

PaddleOCR-VL GPU inference has also been validated on a real MinerU-generated crop in a separate Conda environment from MinerU:

```text
Python 3.12
PaddlePaddle-GPU 3.2.1
PaddleOCR 3.7.0
PaddleOCR-VL 1.6
NVIDIA L40S
device: gpu:0
engine: Paddle
layout detection: disabled
prompt label: table
```

The Linux GPU MinerU run regenerated the Puurunen PDF outputs with `hybrid-engine` and automatically exported 23 canonical table crops. PaddleOCR-VL was applied only to the MinerU crop `page_006_table_001.jpg`, not to the original PDF. The GPU run succeeded with one parsed HTML table and a 58 x 6 rectangular row representation.

Repeatability observations using the same loaded adapter and crop:

```text
first cached-model pass: 44.58 s
warm second pass:       25.24 s
parsed table shape:     58 x 6 both times
parsed cell differences: 0
```

The first-ever GPU run took 91.97 s because it included model download and setup. These timings are validation observations, not formal benchmarks. The Windows CPU and Linux GPU crops were not byte-identical, with observed dimensions of 1431 x 1923 and 1432 x 1923 respectively, so the documentation should not make strong CPU-vs-GPU accuracy claims from their output differences.

Chandra OCR 2 has been integrated as a registered reconstruction adapter using the Hugging Face/in-process API for `datalab-to/chandra-ocr-2`. A real CLI smoke test on an NVIDIA L40S with `--adapter chandra --device gpu:0` completed with one `ok` result, one structured HTML table, one prediction CSV, and no Chandra generation error.

NuExtract3 has been integrated as a registered GPU-only reconstruction adapter using Hugging Face Transformers for `numind/NuExtract3`. A real NVIDIA L40S CLI smoke test with `--adapter nuextract3 --device gpu:0` completed with one table requested, one `ok` result, zero errors, and one prediction CSV.

NuExtract3 has also completed one selected three-document engineering-validation slice containing 83 canonical table crops. The count is test-specific: it is not a fixed benchmark size, a standard Tabulus evaluation dataset, or a model capability claim. All 83 adapter runs returned status `ok`, and 82 prediction CSVs were written. The one missing prediction CSV came from a case where NuExtract3 itself emitted two sibling parseable HTML tables from one canonical crop, so Tabulus correctly preserved the native and parsed evidence without arbitrarily choosing or merging one table.

Tesseract + Table Transformer has been integrated as a registered reconstruction adapter using Tesseract OCR word tokens plus `microsoft/table-transformer-structure-recognition-v1.1-all`. It consumes canonical MinerU crops directly, applies the official-style max-dimension-1000 TATR preprocessing with ImageNet normalization, and uses deterministic token/structure fusion to produce HTML for the shared Tabulus parser.

A real Tabulus CLI validation on an NVIDIA L40S setup completed on one scientific crop and on the same selected three-document engineering-validation slice used during development. That slice contained 83 canonical crops; all 83 adapter runs returned status `ok`, all 83 wrote prediction CSVs, and the observed total runtime was approximately 165.78 s. This count and timing describe one selected engineering-validation slice only, not a benchmark size, accuracy result, or model-superiority claim.

Heavyweight ML integration validations are separate from the mocked unit test suite and should not be treated as scientific accuracy benchmarks.

## Not Yet Implemented In The New Library

- GROBID, Kreuzberg, or Crossref integration
- DeepSeek OCR adapter
- continued-table merging
- standalone scientific table normalization command
- bibliography extraction, reference matching, DOI resolution, and resolved CSV export
- full `tabulus run` process command

## Documentation Boundary

The documentation should describe the installable `tabulus` library and validated module behavior. Development notes, temporary refactor paths, and removable source folders should not be presented as user-facing setup guidance.

Docker instructions remain out of scope for the current Windows CPU workflow.
