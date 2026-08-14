# reference_matches.json

`references/reference_matches.json` records row-level matches between table references and bibliography entries.

```json
{
  "reference_tables_checked": 1,
  "matched_tables": [
    {
      "table_id": 1,
      "source_file": "page_003_table_001.png",
      "reference_column_index": 0,
      "matches_found": 1,
      "matches_total": 1,
      "matches": [
        {
          "row_index": 1,
          "value": "[1]",
          "found": true,
          "matched_reference_indices": [1],
          "matched_references": ["Smith J. Example paper. 2020."],
          "doi": ["10.1234/example"],
          "is_header": false
        }
      ]
    }
  ]
}
```
