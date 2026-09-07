# DOI Resolvers

DOI and scholarly-identity resolution are planned Stage 6 responsibilities.
They are not implemented in the current `src/tabulus` package.

## Planned Responsibility

The intended Stage 6 boundary is paper-level:

- consume the union of unique bibliography indices referenced by selected
  tables for one paper
- resolve each `(paper, bibliography_index)` once
- keep Stage 4 bibliography extraction separate from scholarly metadata
  retrieval
- apply conservative bibliographic evidence rules before writing any validated
  identity
- distinguish scientific rejection from operational failure
- preserve enough provenance for reproducibility

The current repository does not expose a resolver CLI command, resolver
package, Crossref or CORE provider client, LLM adjudication module, status
model, checkpoint writer, or `references/reference_resolution.json` writer.

## Current Boundary

Crossref, CORE, LLM providers, embedding services, and external scholarly
search are not used by the implemented rebuilt pipeline. Stage 4 extracts
bibliography text from GROBID, and Stage 5 deterministically links table-cell
references to bibliography positions offline.

See {doc}`../tutorial/13-doi-resolution` for the planned Stage 6 boundary and
{doc}`reference-matchers` for the implemented Stage 5 matcher.
