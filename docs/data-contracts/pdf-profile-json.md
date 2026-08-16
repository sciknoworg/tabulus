# pdf_profile.json

`metadata/pdf_profile.json` describes the target first PDF profiling module output.

The first adapter is MinerU, which emits table images and structured metadata from a scientific PDF.

The source MinerU files are documented in `data-contracts/mineru-output-files.md`. The Tabulus profile contract below is the normalized module output that downstream steps should consume, regardless of which profiling adapter produced it.

Current implementation status: the new `tabulus.mineru` library discovers typed table regions from existing MinerU output, but it does not yet write this JSON file.

```json
{
  "run_id": "P51",
  "adapter": "mineru",
  "adapter_version": "3.4.5",
  "tables_found": 1,
  "crops_saved": 1,
  "refs_start_page": 12,
  "tables": [
    {
      "table_id": 1,
      "page_nr": 3,
      "png_name": "page_003_table_001.png",
      "png": "runs/P51/images/tables/page_003_table_001.png",
      "mineru_img_path": "ocr/images/example.png",
      "bbox": [100, 200, 900, 600],
      "table_caption": "Table 1. Example caption",
      "table_footnote": null
    }
  ],
  "status": "profiled"
}
```

## MinerU Field Mapping

| Tabulus field | MinerU source | Notes |
| --- | --- | --- |
| `png` | `content_list.json` table entry `img_path` | Future export step should copy or convert the referenced MinerU-generated image into the run output. |
| `mineru_img_path` | `content_list.json` table entry `img_path` | Preserve the original adapter path for traceability. |
| `page_nr` | `content_list.json` table entry `page_idx` | MinerU uses a zero-based page index. The current library converts it into document page numbering. |
| `bbox` | `content_list.json` table entry `bbox` | Used for QA and traceability; Tabulus does not currently crop from this value. |
| `table_caption` | `content_list.json` table entry `table_caption` | Preserve as text or normalized list according to the final contract. |
| `table_footnote` | `content_list.json` table entry `table_footnote` | Preserve as text or normalized list according to the final contract. |
| `table_body` | `content_list.json` table entry `table_body` | Keep when available so MinerU reconstruction can be compared against PaddleOCR-VL output. |
