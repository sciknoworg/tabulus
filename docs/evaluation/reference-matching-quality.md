# Reference Matching Quality

Reference matching evaluation inspects table-cell-to-bibliography-position
links. Stage 5 is deterministic and offline; DOI values carried from Stage 4
are extraction evidence, not independently resolved scholarly identities.

This should be measured after bibliography extraction and table reconstruction
contracts are stable.

The scored artifact is `references/reference_matches.json`, optionally
compared with curated row-level match labels. Metrics should distinguish:

- extracting the reference-like cell from the table
- matching that cell to the intended bibliography entry
- carrying through DOI values when available in Stage 4 extraction evidence
- leaving unresolved rows traceable

Without human gold-standard labels, report coverage, consistency, agreement,
and unmatched-token/row counts. Precision, recall, F1, and accuracy require
suitable curated labels; successful links alone do not establish correctness.

Reference matching evaluation should not mutate prediction CSV files. Resolved
CSV files are planned downstream outputs produced after paper-level scholarly
reference resolution exists.

## Stage 6 Evaluation Boundary

Stage 6 is not implemented in the current `src/tabulus` package. When it is
implemented, its evaluation should use a paper-level denominator: the union of
referenced bibliography indices across reconstruction methods, deduplicated by
bibliography index. That denominator is distinct from Stage 5 cells, rows, or
citation tokens.

Resolution coverage, consistency, and agreement should not be presented as
accuracy without a human gold standard.
