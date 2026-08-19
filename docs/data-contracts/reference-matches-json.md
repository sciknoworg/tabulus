# reference_matches.json

`references/reference_matches.json` records row-level matches between table references and bibliography entries.

This artifact belongs to the reference-resolution branch of the pipeline. It consumes reference-like normalized table rows or prediction CSV content plus `references/bibliography.json`; it does not mutate the prediction CSV used for table-reconstruction evaluation.

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

The retained legacy implementation records whether each reference-like cell was matched, which bibliography entries were selected, and which DOI values were available. The rebuilt `src/tabulus` library has not yet implemented this stage.
