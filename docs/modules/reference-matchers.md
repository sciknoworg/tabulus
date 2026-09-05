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
GROBID, Crossref, LLMs, embedding models, search engines, or external metadata
services. DOI values can only be matched when they are already present in the
Stage 4 bibliography artifact. Missing DOI enrichment belongs to the later
DOI-resolution stage.

Numeric references use one-based bibliography positions in normalized GROBID
TEI order. A positional match is a linkage result, not an accuracy metric by
itself.
