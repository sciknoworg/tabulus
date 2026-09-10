# Reference Matchers

Reference matchers link table cells to bibliography entries.

## Responsibility

- Detect the reference column.
- Split multi-reference cells.
- Match against bibliography entries.
- Preserve match evidence and failures.
- Leave reconstruction prediction CSVs unchanged.

## Current Strategies

- `numeric_position`
- `doi_exact`
- `author_year`
- `author_only`
- `text_containment`

Reference matching is deterministic Tabulus logic. It consumes selected
reference-like tables and `references/bibliography.json`; it does not call
GROBID, Crossref, CORE, LLMs, embedding models, search engines, or external
metadata services. DOI values can only be matched when they are already present
in the Step 4 bibliography artifact.

Numeric references use one-based bibliography positions in normalized GROBID
TEI order. A positional match is a linkage result, not an accuracy metric by
itself.

The matcher writes table-cell links. Step 6 aggregates
`matched_reference_indices` across supplied Step 5 artifacts at paper scope,
deduplicates them by bibliography index, and resolves each unique
`(paper, bibliography_index)` once rather than resolving every cell occurrence
independently.

See {doc}`../tutorial/13-doi-resolution` for Step 6 and
{doc}`../data-contracts/reference-matches-json` for the Step 5 artifact.
