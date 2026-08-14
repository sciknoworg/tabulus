# pdf_profile.json

`metadata/pdf_profile.json` describes the first PDF profiling module output.

The first adapter is MinerU, which emits table images and structured metadata from a scientific PDF.

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
