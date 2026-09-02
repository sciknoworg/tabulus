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
- implements PaddleOCR-VL, Chandra OCR 2, NuExtract3, Tesseract + Table Transformer, RapidOCR + Docling TableFormer, Granite Vision 4.1 4B, TRivia-3B, GLM-OCR, Dolphin-v2, DeepSeek-OCR-2, Nanonets-OCR-s, MonkeyOCRv2-B-Parsing, NVIDIA Nemotron Parse v1.2, HunyuanOCR-1.5, and dots.mocr adapters for MinerU-generated table crops
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
- invokes Granite Vision 4.1 4B directly on canonical crops with the `<tables_otsl>` prompt and Docling OTSL parsing
- preserves Granite model/revision metadata, raw generated output, OTSL sequence, structured cells, table dimensions, image dimensions, and generation provenance
- invokes TRivia-3B directly on canonical crops through Hugging Face Transformers with deterministic generation
- preserves TRivia model/revision metadata, generation settings, token counts, raw OTSL, image dimensions, and Tabulus OTSL-normalization provenance
- invokes GLM-OCR directly on canonical crops through Hugging Face Transformers using `AutoProcessor` and `AutoModelForImageTextToText`
- preserves GLM-OCR model/revision metadata, raw generated HTML, clean parser-facing HTML, resolved dtype and device metadata, and generation provenance
- invokes Dolphin-v2 directly on canonical crops through Hugging Face Transformers using the `ByteDance/Dolphin-v2` checkpoint and Qwen2.5-VL model class
- preserves Dolphin-v2 model/revision metadata, backbone/model-class metadata, raw generated HTML, clean parser-facing HTML, deterministic generation settings, image-preprocessing provenance, and token counts
- invokes DeepSeek-OCR-2 directly on canonical crops through its model-specific `infer(...)` method with custom Transformers code from the pinned Hugging Face model revision
- preserves DeepSeek-OCR-2 model/revision metadata, grounding/model output, dynamic-resolution settings, parser-input policy, structured-table counts, dependency/runtime versions, and recropping flags
- invokes Nanonets-OCR-s directly on canonical crops through Hugging Face Transformers using `AutoProcessor` and `AutoModelForImageTextToText`
- preserves Nanonets-OCR-s model/revision metadata, Qwen2.5-VL backbone and model class, processor settings, raw generated HTML, clean parser-facing HTML, dependency/runtime versions, and canonical-crop provenance
- invokes MonkeyOCRv2-B-Parsing directly on canonical crops through Hugging Face Transformers using direct single-task table recognition
- preserves MonkeyOCRv2-B-Parsing model/revision metadata, direct table-recognition settings, raw generated OTSL, special-token cleanup provenance, deterministic OTSL-to-HTML normalization provenance, and canonical-crop provenance
- invokes NVIDIA Nemotron Parse v1.2 directly on canonical crops through Hugging Face Transformers with GPU-only validated registry support
- preserves Nemotron model/revision metadata, C-RADIO dependency revision metadata, grounded semantic objects, generated bounding boxes as provenance, Table-class LaTeX/tabular content, NVIDIA-postprocessed HTML, generation settings, helper provenance, runtime package versions, and canonical-crop provenance
- invokes HunyuanOCR-1.5 directly on canonical crops through Hugging Face Transformers with GPU-only validated registry support
- preserves HunyuanOCR-1.5 model/revision metadata, model class/type, official table-task prompt, raw/decoded/clean HTML outputs, official repetition-safeguard metadata, dependency/runtime versions, and canonical-crop provenance
- invokes dots.mocr directly on canonical crops through Hugging Face Transformers with GPU-only validated registry support
- preserves dots.mocr model/revision metadata, resolved remote-code classes, active layout prompt, raw and clean JSON layout output, Table-category objects, table bounding boxes as provenance only, model-emitted HTML, dependency/runtime versions, and canonical-crop provenance
- normalizes supported OTSL structural tokens into HTML before shared parsing without semantic cell correction or heuristic reconstruction repair
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
- Granite Vision registry metadata, GPU-only device handling, OTSL generation/parsing, native evidence preservation, and empty-result handling
- TRivia registry metadata, GPU-only device handling, runtime reuse, native OTSL preservation, OTSL normalization, and empty-result handling
- GLM-OCR registry metadata, GPU-only device handling, runtime reuse, raw/clean HTML preservation, shared-parser dispatch, and empty-result handling
- Dolphin-v2 registry metadata, GPU-only device handling, runtime reuse, Dolphin resize preprocessing, deterministic generation metadata, raw/clean HTML preservation, shared-parser dispatch, and empty-result handling
- DeepSeek-OCR-2 registry metadata, GPU-only device handling, exact dependency-version checks, runtime reuse, unchanged model-output parser dispatch, dynamic-resolution metadata, Markdown fallback, and empty-result handling
- Nanonets-OCR-s registry metadata, GPU-only device handling, deterministic generation settings, raw/clean HTML preservation, shared-parser dispatch, processor configuration, and empty-result handling
- MonkeyOCRv2-B-Parsing registry metadata, GPU-only device handling, deterministic generation settings, direct table-recognition configuration, raw OTSL preservation, and OTSL-to-HTML parser dispatch
- NVIDIA Nemotron Parse v1.2 registry metadata, GPU-only device handling, local pinned helper loading, C-RADIO revision verification, NVIDIA generation processors, grounded object preservation, Table-class HTML postprocessing, shared-parser dispatch, multiple-table preservation, and empty-result handling
- HunyuanOCR-1.5 registry metadata, GPU-only device handling, exact dependency-version checks, runtime reuse, official table-task prompt, raw/decoded/clean HTML preservation, official repetition safeguards, shared-parser dispatch, multiple-table preservation, and empty-result handling
- dots.mocr registry metadata, GPU-only device handling, exact dependency-version checks, runtime reuse, remote-code class verification, active layout prompt, JSON layout traversal, Table-category filtering, bounding-box provenance policy, shared-parser dispatch, multiple-table preservation, invalid-JSON handling, and empty-result handling
- legacy-compatible HTML/Markdown table parsing
- shared OTSL-to-HTML normalization for `fcel`, `ecel`, `lcel`, `ucel`, `xcel`, and `nl`
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

Granite Vision 4.1 4B has completed focused adapter tests, full-suite
validation, and real GPU CLI integration validation. The validated
configuration uses `ibm-granite/granite-vision-4.1-4b` at revision
`dd48e97503de471803850df70843cf9eb5da8712`, Docling 2.123.1, Transformers
4.57.3, bfloat16, and SDPA. This confirms integration behavior only and does
not establish reconstruction accuracy or model superiority.

TRivia-3B has been integrated as a registered GPU-only reconstruction adapter
using Hugging Face Transformers for `opendatalab/TRivia-3B` at revision
`fcf890f3869afaa9fc768a14e72ab1ff46bfc813`. The validated configuration uses
Transformers 5.16.1, `AutoProcessor`, `AutoModelForMultimodalLM`, bfloat16,
`do_sample=False`, `max_new_tokens=8192`, and `repetition_penalty=1.05`.
TRivia receives canonical MinerU crops directly and produces native OTSL,
which Tabulus preserves before deterministic OTSL-to-HTML normalization and
shared HTML parsing. This confirms integration behavior only and does not
establish reconstruction accuracy or model superiority.

GLM-OCR has been integrated as a registered GPU-only reconstruction adapter
using Hugging Face Transformers for `zai-org/GLM-OCR` at revision
`ca5d8b3e287e52589e37c28385d9655ee4372f9d`. The validated configuration uses
Transformers 5.16.1, `AutoProcessor`, `AutoModelForImageTextToText`,
`torch_dtype="auto"` resolving to BF16 in the validated run, the prompt
`Table Recognition:`, and `max_new_tokens=8192`. GLM-OCR receives canonical
MinerU crops directly and produces native HTML; Tabulus preserves the raw
generated output, removes model special tokens only for the clean
parser-facing representation, and passes that HTML through the existing shared
HTML parser. This confirms integration behavior only and does not establish
reconstruction accuracy or model superiority.

Dolphin-v2 has been integrated as a registered GPU-only reconstruction adapter
using the `ByteDance/Dolphin-v2` checkpoint at revision
`c37c62768c644bb594da4283149c627765aa80f3`. The checkpoint uses a Qwen2.5-VL
backbone and the Transformers class `Qwen2_5_VLForConditionalGeneration`; the
adapter does not substitute a generic Qwen checkpoint. The validated
configuration used Python 3.12, PyTorch 2.6.0, torchvision 0.21.0,
Transformers 4.51.0, Accelerate 1.4.0, and `qwen-vl-utils` 0.0.14.
Dolphin-v2 receives canonical MinerU crops directly, applies RGB conversion
and deterministic Dolphin `resize_img`-style preprocessing, and generates
native HTML with `do_sample=False`, `temperature=None`, and
`max_new_tokens=4096`. Tabulus preserves raw and clean HTML plus model,
generation, image-preprocessing, token-count, and device provenance before
shared HTML parsing.

A deterministic reproducibility check reconstructed the same canonical crop
twice independently. Both runs produced identical clean model output with
SHA-256 `74e76fe66108ec3e1f20b2b0d9d27e47fe3e2699ff656b9aad9b1e3ac9bb8711`,
2317 generated tokens, `do_sample=False`, `temperature=None`, and one detected
HTML table. The implementation passed 11 focused Dolphin-v2 adapter tests and
the complete Tabulus test suite with 195 tests.

On one selected ALD-paper engineering reconstruction slice containing 83
canonical table crops, Dolphin-v2 produced 82 successful reconstructions, one
empty reconstruction, zero runtime errors, and 82 prediction CSVs in
approximately 45 minutes 10 seconds. The empty result was a large table where
generation reached the 4096-token ceiling before completing a closing HTML
table. Downstream reference-table classification over that run reported 65
reference tables and 18 non-reference tables. These are operational
engineering observations for one selected slice, not a benchmark size,
accuracy metric, or model-superiority claim.

DeepSeek-OCR-2 has been integrated as a registered GPU-only reconstruction
adapter using `deepseek-ai/DeepSeek-OCR-2` at revision
`aaa02f3811945a91062062994c5c4a3f4c0af2b0`. The resolved model class in the
validated configuration is `DeepseekOCR2ForCausalLM`. The adapter uses custom
Transformers model code from the pinned Hugging Face model revision with
`trust_remote_code=True`, `use_safetensors=True`, FlashAttention 2,
`bfloat16`, and the model-specific `infer(...)` path. It explicitly validates
`transformers==4.46.3`, `tokenizers==0.20.3`, and `flash-attn==2.7.3`.

DeepSeek-OCR-2 receives canonical MinerU crops directly. The adapter records
`input_policy=canonical_mineru_crop`, `layout_redetection=False`,
`recropping=False`, and `external_recropping=False`. Its `crop_mode=True`
setting is model-internal dynamic-resolution tiling/resizing of the already
supplied canonical crop, not external table redetection or recropping.

The exact prompt is:

```text
<image>
<|grounding|>Convert the document to markdown.
```

Validated inference settings include `base_size=1024`, `image_size=768`,
`crop_mode=True`, `save_results=False`, and `eval_mode=True`. The underlying
validated eval-mode generation path uses `max_new_tokens=8192`,
`do_sample=False`, `temperature=0.0`, `no_repeat_ngram_size=35`, and
`use_cache=True`. Sampling is disabled, so `temperature=0.0` should not be
interpreted as stochastic temperature-based sampling.

DeepSeek-OCR-2 can emit grounding metadata followed by structured table
markup. Tabulus preserves the returned model output unchanged and records
`normalization=none` and `parser_input=model_infer_output_unchanged`. The
validated smoke-test output contained surrounding DeepSeek grounding metadata
plus one HTML table, and the shared parser extracted a 60 x 5 table without
DeepSeek-specific preprocessing.

A reproducibility check reconstructed the same canonical Miikkulainen table
crop twice independently with the validated direct inference configuration.
Both runs produced identical model output with SHA-256
`a429813fd6c6d0d839697ce10bca0ff7e63547de55ce48f9e210b15ee6ee803a`, 4205
characters, 1404 decoded output tokens, and one 60 x 5 parsed table. The real
CLI smoke test reproduced one requested table, one `ok` result, zero empty
results, zero errors, and one prediction CSV.

The implementation validation completed with 11 focused DeepSeek-OCR-2 tests,
206 complete Tabulus tests, and a clean `git diff --check`.

On the existing three-paper ALD engineering reconstruction slice, DeepSeek-OCR-2
processed 83 canonical table crops in 30m 28.890s:

```text
Cremers - 2019:               14 requested, 14 ok, 0 empty, 0 error, 14 prediction CSVs
Miikkulainen 2013:            46 requested, 44 ok, 2 empty, 0 error, 44 prediction CSVs
Puurunen - February 2005:     23 requested, 21 ok, 2 empty, 0 error, 21 prediction CSVs
Total:                        83 requested, 79 ok, 4 empty, 0 error, 79 prediction CSVs
```

The four empty cases were inspected:

```text
Miikkulainen 2013:            page_024_table_018, page_024_table_019
Puurunen - February 2005:     page_019_table_014, page_019_table_015
```

They were not runtime errors or shared-parser errors. All had
`parser_error=None` and `structured_tables_detected=0`. The model-native
outputs contained image grounding, a subtitle plus image grounding, or
individually grounded text/equation-like elements, but no structured table
representation. Tabulus therefore correctly retained them as `status="empty"`
and did not write prediction CSVs.

Downstream reference-table classification over this engineering run reported:

```text
Cremers - 2019:               14 considered, 8 reference tables
Miikkulainen 2013:            46 considered, 41 reference tables
Puurunen - February 2005:     23 considered, 13 reference tables
Total:                        83 considered, 62 reference tables, 21 non-reference tables
```

These are downstream operational classification results, not reconstruction
accuracy. They do not imply that all 83 crops produced prediction CSVs:
reconstruction produced 79 prediction CSVs and four model-native empty
results. The reconstruction and classification counts are engineering
observations for one selected slice, not gold-standard precision, recall, F1,
or evidence that DeepSeek-OCR-2 is better or worse than another adapter.

Nanonets-OCR-s has been integrated as a registered GPU-only reconstruction
adapter using `nanonets/Nanonets-OCR-s` at revision
`3baad182cc87c65a1861f0c30357d3467e978172`. The checkpoint uses a Qwen2.5-VL
backbone and the runtime Transformers class
`Qwen2_5_VLForConditionalGeneration`; the adapter does not substitute a
generic Qwen checkpoint for Nanonets-OCR-s.

The validated configuration used the `tabulus-nanonets-ocr-s` environment,
Transformers 4.52.4, tokenizers 0.21.4, FlashAttention 2.7.3, PyTorch
2.6.0+cu124, bfloat16, `flash_attention_2`, `AutoProcessor` with
`use_fast=False`, and `AutoModelForImageTextToText` on an NVIDIA L40S.
Nanonets-OCR-s receives canonical MinerU crops directly, converts the supplied
crop to RGB, and relies on model-internal image preprocessing rather than
external redetection or recropping.

Nanonets-OCR-s produced native structured HTML in the validated table test.
Tabulus preserves the raw decoded generation, removes model special tokens
only for the clean parser-facing representation, and passes the clean HTML to
the shared `parse_table_text` parser without Nanonets-specific semantic or
structural normalization. The validated output included rich HTML table
constructs such as `thead`, `tbody`, `th`, `td`, `rowspan`, `sup`, `sub`, and
`br`.

Direct model validation confirmed that the frozen configuration can reproduce
identical clean output across independent runs. A real end-to-end Tabulus CLI
reconstruction was also validated. The implementation validation reported 10
focused Nanonets adapter tests, 216 complete Tabulus tests, and a clean
`git diff --check`.

On the existing three-paper ALD engineering reconstruction slice, Nanonets-OCR-s
processed 83 canonical table crops in 97m48.535s:

```text
Total: 83 requested, 76 ok, 7 empty, 0 error, 75 prediction CSVs
```

Downstream reference-table classification over this engineering run reported:

```text
Total: 83 considered, 59 reference tables, 24 non-reference tables
```

These are aggregate engineering observations only. They are not reconstruction
accuracy, classification accuracy, precision, recall, F1, runtime guarantees,
or evidence that Nanonets-OCR-s is better or worse than another adapter.
Accuracy requires comparison against the appropriate gold-standard
annotations in the evaluation stage.

MonkeyOCRv2-B-Parsing has been integrated as a registered GPU-only
reconstruction adapter using `zenosai/MonkeyOCRv2-B-Parsing` at revision
`2419139b7bcd3fda2689b2a83167172afba91c8b`. The validated configuration uses
Python 3.11, Transformers 4.57.1, Accelerate 1.11.0, timm 1.0.27, einops
0.8.1, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, bfloat16, direct
Transformers inference, `AutoProcessor` with `use_fast=False`, explicit SDPA
attention, and GPU execution.

MonkeyOCRv2-B-Parsing receives canonical MinerU crops directly and uses
MonkeyOCRv2's direct single-task table-recognition prompt rather than the full
document-layout pipeline. FlashAttention, vLLM, and DFlash are not required by
the documented adapter path. The adapter preserves raw generated OTSL, removes
special tokens only for the parser-facing representation, converts OTSL through
the existing deterministic `otsl_table_to_html` function, and then uses the
shared table parser. It does not perform external redetection, external
recropping, semantic repair, reference resolution, or continued-table merging.

The full multi-paper MonkeyOCRv2-B-Parsing engineering run had not yet been
finalized when this note was written, so no aggregate batch counts or runtime
are recorded here.

NVIDIA Nemotron Parse v1.2 has been integrated as a registered GPU-only
reconstruction adapter using `nvidia/NVIDIA-Nemotron-Parse-v1.2` at revision
`2bd0189bffd6cdded6280d9f22a4077b25a504e3`. The adapter uses direct Hugging
Face Transformers inference, bfloat16, SDPA attention, model image canvas
`[2048, 1664]`, and the prompt
`</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>`.

The adapter verifies the transitive `nvidia/C-RADIOv2-H` implementation
against revision `0d8f4c18c877166eda07ddae1386bcad256b7a6a`. Nemotron helper
files are loaded from the pinned Hugging Face model revision with local-file
behavior rather than silently fetched during inference.

NVIDIA Nemotron Parse v1.2 receives canonical MinerU crops directly. It
generates grounded semantic objects whose Table-class content is represented
natively as LaTeX/tabular. Tabulus preserves the generated objects and bounding
boxes as provenance, converts Table-class output to HTML through NVIDIA's
deterministic table postprocessing, and passes the resulting HTML through the
shared parser. Generated bounding boxes are not used for recropping. The
adapter does not perform external layout redetection, table redetection,
semantic repair, reference resolution, or continued-table merging.

HunyuanOCR-1.5 has been integrated as a registered GPU-only reconstruction
adapter using `tencent/HunyuanOCR` at revision
`47644ecc4fc854efa4f505155158831f36773ee4`. The validated configuration uses
Python 3.12, Transformers 5.13.0, Accelerate 1.14.0, PyTorch 2.11.0+cu130,
torchvision 0.26.0+cu130, bfloat16, eager attention,
`HunYuanVLForConditionalGeneration`, and model type `hunyuan_vl`.

HunyuanOCR-1.5 receives canonical MinerU crops directly and uses the official
table prompt `把图中的表格解析为HTML。`. Tabulus preserves raw output with special
tokens, decoded output with special tokens removed, and clean parser-facing
HTML after the official repeated-suffix cleanup. Tail-repetition stopping and
final repeated-suffix cleanup are recorded as inference safeguards, not
semantic table repair. The clean HTML is passed through the shared parser, and
multiple HTML tables are preserved without arbitrary selection, collapse,
concatenation, or merging.

dots.mocr has been integrated as a registered GPU-only reconstruction adapter
using `dots-studio/dots.mocr` at revision
`e539fbb52280393adc081b289ec597430a0f9031`. The validated configuration uses
Python 3.12, Transformers 4.57.6, Accelerate 1.14.0, PyTorch 2.7.0+cu128,
torchvision 0.22.0+cu128, qwen-vl-utils 0.0.14, FlashAttention 2.8.0.post2,
bfloat16, and `flash_attention_2`.

dots.mocr receives canonical MinerU crops directly and uses the active
`prompt_layout_all_en` prompt to generate model-native JSON layout output.
Tabulus selects model-emitted objects whose category is `Table`, preserves
their HTML and bounding boxes as native evidence, and passes the HTML through
the shared parser. The bounding boxes are provenance only and are not used for
recropping. The adapter performs no JSON repair, semantic repair, external
layout redetection, external table redetection, external recropping, reference
resolution, or continued-table merging.

## Not Yet Implemented In The New Library

- GROBID, Kreuzberg, or Crossref integration
- continued-table merging
- standalone scientific table normalization command
- bibliography extraction, reference matching, DOI resolution, and resolved CSV export
- full `tabulus run` process command

## Documentation Boundary

The documentation should describe the installable `tabulus` library and validated module behavior. Development notes, temporary refactor paths, and removable source folders should not be presented as user-facing setup guidance.

Docker instructions remain out of scope for the current Windows CPU workflow.
