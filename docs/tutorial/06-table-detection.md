:orphan:

# PDF Profiling Internal: Table Detection

## Goal

Locate tables in the paper and assign stable table ids.

## Input

`layout/layout_items.json`.

## Output

`tables/tables_detected.json`.

## Module Contract

```json
{
  "tables": [
    {
      "table_id": 1,
      "page_nr": 3,
      "bbox": [100, 200, 900, 600],
      "caption": null,
      "footnote": null,
      "source": "mineru"
    }
  ],
  "tables_found": 1,
  "status": "tables_detected"
}
```

## Default Implementation

The current pipeline reads MinerU layout items where `type == "table"`.

## Verification

The step succeeds when each table has a table id, page number, and crop-ready location or image path.
