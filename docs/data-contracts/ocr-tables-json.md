# ocr_tables.json

`tables/ocr_tables.json` records structured table OCR results. The current library exposes adapter-neutral `TableOCRResult` objects; a later batch stage can materialize them in this file shape.

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

`native_json` and `native_markdown` preserve the adapter's public result representations for provenance and debugging. For PaddleOCR-VL these are two serializations or views of the same Paddle inference result, not two independent predictions.

`parsed_tables` contains the legacy-compatible rectangular row representation recovered from the native Markdown/HTML text. The parser prefers HTML tables and falls back to GitHub-style pipe tables only when no HTML table is present.

`status` should explicitly record `"ok"`, `"empty"`, or `"error"`. A table OCR adapter should never silently drop an input crop.

This contract records table reconstruction output. It is not final scientific normalization, continued-table merging, formula rewriting, or reference resolution.
