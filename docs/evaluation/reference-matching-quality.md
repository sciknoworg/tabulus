# Reference Matching Quality

Reference matching evaluation inspects table-cell-to-bibliography-position
links. Stage 5 is deterministic and offline; DOI values carried from Stage 4
are extraction evidence, not independently resolved scholarly identities.

This should be measured after bibliography extraction and table reconstruction contracts are stable.

The scored artifact is `references/reference_matches.json`, optionally compared with curated row-level match labels. Metrics should distinguish:

- extracting the reference-like cell from the table
- matching that cell to the correct bibliography entry
- carrying through DOI values when available
- leaving unresolved rows traceable

Without human gold-standard labels, report coverage, consistency, agreement,
and unmatched-token/row counts. Precision, recall, F1, and accuracy require
suitable curated labels; successful links alone do not establish correctness.

Reference matching evaluation should not mutate prediction CSV files. Resolved CSV files are downstream outputs produced after matching and DOI enrichment.

## Frozen Stage 5 JVSTA Demonstration

These are frozen operational and matching-coverage counts, not accuracy results:

| Measure | Count |
| --- | ---: |
| Successful runs | 153 |
| Selected reference-containing reconstructed-table instances | 1,301 |
| Checked instances | 1,295 |
| Skipped instances | 6 |
| Cells matched | 35,846 / 37,892 |
| Citation tokens matched | 114,691 / 126,282 |
| Ambiguous tokens | 0 |
| Rows containing unmatched tokens | 2,132 |
| Skipped: `multiple_parsed_tables` | 2 |
| Skipped: `no_parsed_table` | 4 |

All 114,691 successful links use `numeric_position`, the 1-based GROBID TEI
bibliography order. Selected instances are reconstruction-specific outputs,
not a count of distinct physical tables. One deterministic regex/rule classifier
is applied independently to the reconstruction methods' outputs in Stage 3.

## Stage 6 Evaluation Boundary

Stage 6 consumes the union of referenced bibliography indices across all
reconstruction methods and resolves each unique `(paper, bibliography_index)`
once. Its coverage denominator is the unique target set, not Stage 5 cells or
tokens. `validated_without_doi` is a legitimate identity outcome; `rejected`
means insufficient evidence, while operational failures abort/checkpoint a run.

Earlier Stage 6 corpus runs are provisional diagnostics. The clean final rerun
with enriched Stage 4 artifacts is underway, so final Stage 6 coverage/result
percentages are not yet reported. See {doc}`../project-notes/current-state`.
Resolution coverage, consistency, and agreement should not be presented as
accuracy without a human gold standard.
