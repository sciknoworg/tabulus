# DOI Resolvers

Stage 6 validates scholarly identities for matched bibliography entries and
records DOI values when established. It creates one paper-level registry.

## Responsibility

- Consume the union of unique bibliography indices referenced by selected
  tables for one paper.
- Resolve each `(paper, bibliography_index)` once.
- Use DOI values parsed from bibliography text when available.
- Validate existing DOIs through Crossref, then use Crossref bibliographic
  candidates, CORE fallback, and bounded LLM adjudication/query reformulation
  when needed.
- Apply a final deterministic admissibility gate; title similarity alone is
  insufficient without independent supporting evidence.
- Write `validated_with_doi`, `validated_without_doi`, or `rejected` outcomes
  to `references/reference_resolution.json`.
- Checkpoint completed references incrementally and abort on operational
  failures rather than silently recording rejections.

## Current Status

Stage 6 is implemented on the GPU cluster; see
{doc}`../project-notes/current-state` for the local checkout boundary.
Crossref, CORE, and LLM calls occur only here, not in Stage 4 extraction or
offline Stage 5 matching. The LLM may only select supplied candidates, reject
them all, or propose one better query; there is at most one scholarly-search
retry and no relaxation of the deterministic minimum-evidence requirement.

See {doc}`../tutorial/13-doi-resolution` for the conservative evidence policy,
chapter/container, translated-citation, and edition/year safeguards, bounded
operational retries, and checkpoint fingerprint rules. Stage 7 export remains
unimplemented.
