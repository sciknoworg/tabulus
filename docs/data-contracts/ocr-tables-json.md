# ocr_tables.json

`tables/ocr_tables.json` describes the adapter-neutral structured table reconstruction result shape. The current library exposes `TableOCRResult` objects and the implemented batch command persists the same boundary as per-table JSON files under `reconstructions/<adapter>/native/` and `reconstructions/<adapter>/parsed/`.

This is an intermediate reconstruction checkpoint:

```text
adapter-native output
        |
        v
structured table evidence
        |
        v
normalized reconstruction
        |
        v
prediction CSV
```

```json
{
  "success": true,
  "tables_found": 1,
  "tables": [
    {
      "table_id": 1,
      "adapter_name": "paddleocr-vl",
      "device": "cpu",
      "source_image": "table-crops/images/page_003_table_001.png",
      "status": "ok",
      "result_count": 1,
      "adapter_version": "3.7.0",
      "model_version": "PaddleOCR-VL v1.6",
      "native_json": [],
      "native_markdown": [],
      "parsed_tables": [
        {
          "n_rows": 2,
          "n_cols": 2,
          "rows": [["Reference", "Value"], ["[1]", "A"]],
          "source": "html"
        }
      ],
      "provenance": {
        "table_id": 1,
        "page_nr": 3,
        "source": "mineru"
      },
      "error": null
    }
  ]
}
```

## Concepts

`native_json` and `native_markdown` preserve the adapter's public result representations for provenance and debugging. For PaddleOCR-VL these are two serializations or views of the same Paddle inference result, not two independent predictions. Chandra OCR 2 preserves generated HTML and generation metadata; NuExtract3 preserves generated Markdown/HTML content and model/generation metadata. Tesseract + Table Transformer preserves OCR word tokens, word bounding boxes, TATR structure objects, fused cells, and generated HTML. Granite Vision preserves model metadata, raw generated OTSL, the OTSL sequence, and Docling-parsed cells and dimensions. TRivia-3B preserves model metadata, generation settings, token counts, raw OTSL, and the Tabulus OTSL-to-HTML normalization provenance. GLM-OCR preserves model/revision metadata, raw generated HTML, clean parser-facing HTML with special tokens removed, resolved dtype/device metadata, and generation provenance. Dolphin-v2 preserves model/revision metadata, Qwen2.5-VL backbone and model class, deterministic generation settings, source and resized image dimensions, token counts, raw generated HTML, clean parser-facing HTML, and image-preprocessing provenance. DeepSeek-OCR-2 preserves model/revision metadata, custom model class, generation settings, dynamic-resolution settings, grounding/model output, parser-input policy, detected structured table counts, and dependency/runtime provenance. Nanonets-OCR-s preserves model/revision metadata, Qwen2.5-VL backbone and model class, processor settings, raw decoded HTML, clean parser-facing HTML, dependency/runtime versions, and canonical-crop input policy. MonkeyOCRv2-B-Parsing preserves model/revision metadata, direct table-recognition settings, raw generated OTSL, special-token cleanup provenance, deterministic OTSL-to-HTML normalization provenance, and canonical-crop input policy. NVIDIA Nemotron Parse v1.2 preserves model and C-RADIO revision metadata, grounded semantic objects, generated bounding boxes for provenance, Table-class LaTeX/tabular content, NVIDIA-postprocessed HTML, generation settings, helper provenance, and canonical-crop input policy. HunyuanOCR-1.5 preserves model/revision metadata, model class/type, raw/decoded/clean HTML outputs, official repetition-safeguard metadata, dependency/runtime versions, and canonical-crop input policy. dots.mocr preserves model/revision metadata, resolved remote-code classes, active layout prompt, JSON layout output, Table-category objects, table bounding boxes as provenance only, model-emitted HTML, dependency/runtime versions, and canonical-crop input policy.

`parsed_tables` contains the legacy-compatible rectangular row representation recovered from the native Markdown/HTML text. The parser prefers HTML tables and falls back to GitHub-style pipe tables only when no HTML table is present.

`status` should explicitly record `"ok"`, `"empty"`, or `"error"`. A table reconstruction adapter should never silently drop an input crop.

The implemented batch output is:

```text
reconstructions/<adapter>/
  native/
  parsed/
  predictions/
  batch_summary.json
```

The batch layer preserves `table_id` values and crop order from `tables_index.json`. A table-level adapter error is persisted as an error result and does not abort later crops.

`parsed/` is Tabulus's common structured table representation. The per-table parsed JSON records table identity, adapter name/version, model version, device, source crop, status, parsed table count, parsed rows, row/column dimensions, parse source such as HTML or Markdown, warnings, and the prediction CSV path when one is written. This layer is the bridge between adapter-specific native output and downstream Tabulus processing.

For the full default reconstruction directory contract and rerun behavior, see {doc}`run-directory`.

This contract records table reconstruction output. It is not final scientific normalization, continued-table merging, formula rewriting, reference resolution, or the final user-facing CSV. Prediction CSV export and resolved CSV export are separate downstream contracts.
