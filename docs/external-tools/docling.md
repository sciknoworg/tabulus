# Docling TableFormer

RapidOCR + Docling TableFormer is an external table-reconstruction integration
used by Tabulus. Tabulus exposes only the bare-crop TableFormer path required
for the current reconstruction workflow; readers who need Docling's broader
PDF conversion and layout capabilities should consult the official project
documentation.

Docling functionality is also used by the separate Granite Vision adapter to
parse Granite-generated OTSL. See {doc}`granite-vision` for that integration;
it is distinct from `rapidocr-tableformer`.

## Official Resources

- [Docling project repository](https://github.com/docling-project/docling)
- [Docling model catalog](https://docling-project.github.io/docling/usage/model_catalog.html)
- [RapidOCR project repository](https://github.com/RapidAI/RapidOCR)

## Role In Tabulus

The registered adapter name is:

```text
rapidocr-tableformer
```

It consumes one canonical MinerU table crop at a time. RapidOCR with ONNX
Runtime performs OCR and extracts word bounding boxes on the supplied crop.
Docling TableFormer V1 performs table-structure recognition on that same
complete crop. The adapter does not redetect or recrop tables from the source
PDF.

This keeps the comparison boundary shared with the other reconstruction
adapters:

```text
canonical MinerU crop
        |
        +--> RapidOCR OCR and word boxes (CPU)
        |
        +--> Docling TableFormer V1 (requested CPU/GPU device)
                    |
                    v
        shared Tabulus parsing and output contract
```

For the generic adapter interface and artifact contract, see
{doc}`../modules/table-ocr-adapters`.

## Invocation

Reconstruct one paper's canonical crops:

```bash
tabulus reconstruct-tables \
  --crops "/path/to/tabulus-output/table-crops/<paper>" \
  --adapter rapidocr-tableformer \
  --device gpu:0
```

The same adapter name can be used with `--crops-folder` for a sequential
multi-paper batch.

## Settings Used By Tabulus

The current integration uses:

- RapidOCR with ONNX Runtime for OCR and word-bounding-box extraction.
- Docling TableFormer V1 in `accurate` mode.
- Docling cell matching enabled.
- RapidOCR execution on CPU.
- TableFormer execution on the requested Tabulus device, such as `cpu` or
  `gpu:0`.

Tabulus uses Docling's bare-crop TableFormer path over the entire supplied
crop. It does not invoke Docling PDF or page-layout detection, semantically
correct cell contents, merge continued tables, or apply reference-resolution
heuristics.

## Native Output

The adapter preserves raw OTSL and the final Docling table structure as native
adapter evidence. That evidence is then passed through the existing common
HTML/Markdown structural parser and the standard Tabulus artifact layers:

```text
<crop-root>/
  reconstructions/
    rapidocr-tableformer/
      native/
      parsed/
      predictions/
      batch_summary.json
```

Prediction CSV creation follows the shared conservative rule: Tabulus writes a
prediction only when the result is `ok` and exactly one structured table is
parsed from the crop. Native and parsed evidence is retained for empty or
ambiguous results.

## Limitations

This adapter operates on canonical MinerU crops and does not independently
locate tables in the original PDF. Continued physical table crops remain
separate, and the adapter does not perform semantic cell correction,
bibliography extraction, reference matching, DOI resolution, or final resolved
CSV generation.
