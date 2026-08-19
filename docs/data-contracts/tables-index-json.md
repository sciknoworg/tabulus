# tables_index.json

`tables_index.json` records detected and copied table images in a table-crop handoff directory. It is the canonical manifest of physical tables passed from MinerU table localization/cropping to table reconstruction.

For the full filesystem contract around native MinerU output, canonical crop
handoffs, reconstruction artifacts, and rerun behavior, see {doc}`run-directory`.

This file is generated automatically by `tabulus profile` after a successful MinerU run unless `--no-export-table-crops` is passed. It can also be regenerated from an existing MinerU output directory with `tabulus export-table-crops`.

The stable file name is `tables_index.json`. Earlier scratch runs sometimes used `table_index.json`; new code and docs should use the plural name.

```json
{
  "tables_found": 2,
  "crops_saved": 2,
  "refs_start_page": 12,
  "image_format_policy": "preserve-source-extension",
  "images_dir": "work/table_crops/images",
  "tables": [
    {
      "table_id": 1,
      "page_nr": 3,
      "in_references": false,
      "image": "images/page_003_table_001.png",
      "image_name": "page_003_table_001.png",
      "bbox": null,
      "table_caption": null,
      "table_footnote": null,
      "mineru_img_path": "images/example_table.png",
      "mineru_source_image": "work/mineru/paper/images/example_table.png",
      "mineru_table_body": "<table>...</table>",
      "source": "mineru"
    }
  ]
}
```

## Required Information

Each table record should provide:

- stable `table_id`
- the image path to pass to the table OCR module
- page provenance
- bounding-box provenance when available
- caption and footnote context when available
- adapter source information
- MinerU's own `table_body` when available, so it can be compared with table-reconstruction adapter output

Downstream reconstruction commands should treat `tables_index.json` as the authoritative crop order and identity source. They should preserve the existing `table_id` values rather than renumbering physical crops. These IDs identify physical detected tables in the document; they are not necessarily the printed table numbers in the paper.

## MinerU Handoff

The current clean MinerU handoff uses MinerU as the canonical table-localization and crop-generation stage:

```text
<document-name>_content_list.json
  |
  v
select entries where type == "table"
  |
  v
resolve each table img_path under MinerU images/
  |
  v
copy only those table images into the Tabulus table-crop collection
  |
  v
write tables_index.json
```

Tabulus does not need to crop the PDF again from `bbox`. The `bbox` should be preserved for traceability and visual QA, but the image passed to table OCR is the MinerU-generated image referenced by `img_path`.

`mineru_table_body` is MinerU's own candidate reconstruction for the detected
table. It is separate from the canonical crop image consumed by PaddleOCR-VL or
another reconstruction adapter, and it should not be described as adapter OCR
output.

Each record represents one physical MinerU table crop. Current Tabulus does not merge multi-page or continued table segments; logical table continuity remains a future concern.

The default handoff directory for `tabulus profile --pdf paper.pdf` is:

```text
<PDF directory>/tabulus-output/table-crops/paper/
  tables_index.json
  images/
    page_003_table_001.png
```

This layout is intentionally separate from MinerU's native output directory. It is useful for profiling, debugging, and reproducible comparison of MinerU `table_body` against reconstruction adapters that consume identical MinerU-generated crops.

## Current Library Status

The current library writes this file with:

```bash
tabulus profile --pdf paper.pdf --backend pipeline
```

To regenerate the handoff from an already completed MinerU run without rerunning MinerU:

```bash
tabulus export-table-crops --mineru-root work/mineru/puurunen_2005 --out work/table_crops
```

The exporter preserves the source image extension rather than converting every image to PNG.
