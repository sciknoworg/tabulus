# reference_matches.json

`references/reference_matches.json` records deterministic row-level linkages
between Stage 3-selected reference-like table cells and entries in
`references/bibliography.json`.

Stage 5 consumes `selected_reference_tables.json` and
`references/bibliography.json`. It reads parsed table artifacts through
`source_parsed`; prediction CSV paths may be retained as provenance, but raw
prediction CSV files are not modified.

## Example

```json
{
  "schema_version": 1,
  "numeric_reference_semantics": "1-based bibliography position in normalized GROBID TEI order",
  "reference_tables_selected": 2,
  "reference_tables_checked": 1,
  "reference_tables_skipped": 1,
  "matched_tables": [
    {
      "table_id": 1,
      "source_file": "table-001.csv",
      "source_parsed": "/path/to/reconstructions/adapter/parsed/table-001.json",
      "source_prediction": "predictions/table-001.csv",
      "reference_column_index": 2,
      "matches_found": 1,
      "matches_total": 1,
      "matches": [
        {
          "row_index": 1,
          "value": "[12, 14]",
          "found": true,
          "matched_reference_indices": [12, 14],
          "matched_references": [
            "Example reference A.",
            "Example reference B."
          ],
          "doi": ["10.1234/example-a", ""],
          "match_provenance": [
            {
              "reference_index": 12,
              "method": "numeric_position",
              "token": "12"
            },
            {
              "reference_index": 14,
              "method": "numeric_position",
              "token": "14"
            }
          ],
          "tokens_total": 2,
          "tokens_matched": 2,
          "unmatched_tokens": [],
          "is_header": false
        }
      ]
    }
  ],
  "skipped_tables": [
    {
      "table_id": 2,
      "source_status": "ok",
      "source_parsed": "parsed/table-002.json",
      "source_prediction": "predictions/table-002.csv",
      "reason": "multiple_parsed_tables",
      "parsed_table_count": 2
    }
  ]
}
```

## Top-Level Fields

`schema_version`
: Contract version for the reference-match artifact.

`numeric_reference_semantics`
: Always `1-based bibliography position in normalized GROBID TEI order` for
  the current implementation.

`reference_tables_selected`
: Number of Stage 3 selected reference-like tables.

`reference_tables_checked`
: Number of selected tables that were safe to match.

`reference_tables_skipped`
: Number of selected tables skipped because the parsed-table artifact could
  not be used safely.

`matched_tables`
: Per-table matching results.

`skipped_tables`
: Selected tables skipped with diagnostics.

## Matched Table Fields

`table_id`
: Physical table identifier from `selected_reference_tables.json`.

`source_file`
: Display/source filename derived from the prediction CSV path when available,
  otherwise from the parsed artifact path.

`source_parsed`
: Resolved parsed-table artifact path used for matching.

`source_prediction`
: Prediction CSV path retained from Stage 3 provenance.

`reference_column_index`
: Zero-based index of the detected reference column, or `null` when no
  reference column is found.

`matches_found`
: Number of non-header reference-cell values with at least one candidate
  bibliography match.

`matches_total`
: Number of non-header reference-cell values attempted.

`matches`
: Row-level match records for non-empty values in the detected reference
  column, including the header row when one is detected.

## Row-Level Fields

`row_index`
: Row index in the parsed table representation.

`value`
: Normalized original cell value considered for matching.

`found`
: Whether at least one candidate bibliography entry was linked.

`matched_reference_indices`
: Bibliography entry indices linked by Stage 5.

`matched_references`
: Raw bibliography strings from `references/bibliography.json`.

`doi`
: DOI values already present in matched bibliography entries. Stage 5 performs
  no external DOI lookup.

`match_provenance`
: Per-candidate evidence. Method values are:

- `numeric_position`
- `doi_exact`
- `author_year`
- `author_only`
- `text_containment`

For example:

```json
{
  "reference_index": 12,
  "method": "numeric_position",
  "token": "12"
}
```

`tokens_total`
: Number of citation tokens attempted for the row-level value.

`tokens_matched`
: Number of attempted tokens with at least one candidate match.

`unmatched_tokens`
: Citation tokens that did not produce a candidate.

`is_header`
: Whether the row-level value was treated as a reference-column header.

## Skipped Table Fields

`table_id`
: Physical table identifier from the selected manifest.

`source_status`
: Reconstruction status retained from Stage 3 provenance.

`source_parsed`
: Parsed artifact path from the selected manifest.

`source_prediction`
: Prediction CSV path from the selected manifest.

`reason`
: Skip reason. Current safe-skip reasons are `no_parsed_table` and
  `multiple_parsed_tables`.

`parsed_table_count`
: Number of structured tables found in the parsed artifact.

## Numeric Semantics

Numeric references are interpreted as bibliography positions:

```makefile
numeric_reference_semantics = "1-based bibliography position in normalized GROBID TEI order"
```

This is deterministic linkage semantics, not guaranteed bibliographic ground
truth. If bibliography extraction skips, merges, or reorders entries, numeric
positions can shift.

## Coverage Boundary

`matches_found / matches_total` is a coverage statistic, not an accuracy
metric. Token counts are diagnostic coverage information. Scientific matching
accuracy requires comparison with human gold-standard data.

Raw reconstruction prediction CSVs remain pre-reference-resolution artifacts
and must not be overwritten by reference matching.
