# Resolved CSV

Resolved CSV files are planned Stage 7 user-facing exports after reference
matching and paper-level scholarly resolution. Stage 7 is not implemented yet.

A prediction CSV is the reconstructed table before enrichment and is the artifact compared with ground truth during table-reconstruction evaluation. A resolved CSV is created later only for a relevant/reference-containing table after bibliography extraction, reference matching, and paper-level DOI/scholarly resolution.

The planned Stage 7 exporter should join the paper-level Stage 6
`references/reference_resolution.json` registry back onto Stage 5 table-cell
links. The same bibliography index should therefore receive the same validated
identity wherever it appears.

The deterministic join should propagate validated identities to every relevant
table cell/reference occurrence, preserving original reference values and
provenance. `validated_without_doi` remains a legitimate identity; a missing DOI
alone must not discard it. Exact columns and rejected-link handling are not yet
implemented. Prediction CSVs remain unchanged.

Illustrative DOI-bearing export, not a finalized schema:

```text
DOI,Value
10.1234/example,A
```
