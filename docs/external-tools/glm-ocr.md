# GLM-OCR

## Official Resources

- [GLM-OCR model](https://huggingface.co/zai-org/GLM-OCR)

## Role In Tabulus

GLM-OCR is a vision-language model used by Tabulus for table reconstruction
from canonical MinerU table crops. The Tabulus adapter sends the crop directly
to the model and expects native HTML table output.

GLM-OCR is a multimodal OCR model for complex document understanding from
Z.ai. The official materials describe an architecture with a CogViT visual
encoder, a lightweight cross-modal connector, and a GLM-0.5B language decoder,
with support for text, formula, table, and information-extraction tasks. The
upstream SDK can run a full layout-and-recognition pipeline, but Tabulus uses a
direct table-recognition prompt against a single canonical crop.

The registered Tabulus adapter is:

```text
glm-ocr
```

The adapter consumes canonical MinerU table crops through the shared Step 2 adapter contract and emits native HTML table output for shared parsing. Tabulus does not invoke the GLM-OCR SDK document pipeline.

## Invocation

Run GLM-OCR through the shared reconstruction CLI:

```bash
export CUDA_VISIBLE_DEVICES=0

tabulus reconstruct-tables \
  --crops /path/to/canonical/table-crops \
  --adapter glm-ocr \
  --device gpu:0
```

The same adapter can be used with `--crops-folder` for a sequential
multi-paper batch.

## Settings Used By Tabulus

The validated integration uses:

- model: `zai-org/GLM-OCR`
- model revision: `ca5d8b3e287e52589e37c28385d9655ee4372f9d`
- runtime: Hugging Face Transformers in-process
- processor/model loading: `AutoProcessor` and `AutoModelForImageTextToText`
- Transformers: `5.16.1`
- model loading dtype: `torch_dtype="auto"`
- validated resolved dtype: BF16
- prompt: `Table Recognition:`
- `max_new_tokens=8192`
- registry capability: GPU only

Tabulus maps `--device gpu:0` to the PyTorch device `cuda:0`. The processor
and model are loaded lazily and reused across crops by the batch layer.

The implemented adapter path does not require the GLM-OCR SDK document
pipeline, vLLM, SGLang, Docker, or a hosted API.

## Native Output

GLM-OCR produces native HTML table output. Tabulus preserves the raw generated
output in provenance. Model special tokens are removed only for the clean
representation used for parsing.

There is no GLM-specific structural normalization. Clean HTML is passed
directly to Tabulus's shared HTML table parser, which preserves existing
`rowspan` and `colspan` semantics in the common rectangular representation.
Tabulus does not heuristically repair inconsistent model-generated HTML
structure or semantically correct cell contents.

The adapter writes the standard reconstruction artifact layers:

```text
<crop-root>/
  reconstructions/
    glm-ocr/
      native/
      parsed/
      predictions/
      batch_summary.json
```

Prediction CSV creation follows the shared Tabulus rule: the adapter result
must be `ok` and exactly one structured table must parse from the crop. Native
and parsed evidence is retained for empty or ambiguous results.

## Limitations

This adapter reconstructs one canonical MinerU crop at a time. It does
not independently locate or crop tables from the source PDF, run PP-DocLayout,
semantically correct cell contents, merge continued tables, classify reference
tables, extract bibliographies, match references, resolve DOI values, or write
final resolved CSV files.

The integration validation confirms adapter behavior and artifact generation.
It does not establish that GLM-OCR is more accurate, better, or worse than any
other reconstruction candidate.