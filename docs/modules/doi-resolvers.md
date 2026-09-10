# DOI Resolvers

DOI and scholarly-identity resolution are implemented as Step 6 paper-level
reference resolution.

## Responsibility

Step 6 consumes the union of bibliography indices linked by Step 5 for one
paper, deduplicates them by bibliography index, and resolves each unique
`(paper, bibliography_index)` once. It keeps Step 4 bibliography extraction
separate from scholarly metadata retrieval and does not modify Step 4 or Step
5 artifacts.

The implementation lives in `src/tabulus/reference_resolution/` and is exposed
through:

```bash
tabulus resolve-references --help
```

The resolver:

- loads Step 4 `references/bibliography.json` as immutable extraction evidence
- loads one or more Step 5 `references/reference_matches.json` artifacts as
  table-cell linkage evidence
- retrieves Crossref and CORE scholarly-work candidates
- applies deterministic bibliographic scoring before accepting candidates
- uses bounded LLM adjudication only for unresolved candidate sets
- applies a final deterministic admissibility gate to LLM-selected candidates
- writes one paper-level `references/reference_resolution.json` artifact only
  after all targets complete
- checkpoints completed references in
  `references/reference_resolution.checkpoint.json` for safe resumption

## Provider Boundary

Crossref and CORE are used only in Step 6. They are not part of Step 4
bibliography extraction or Step 5 reference matching. CORE is a fallback
discovery source, not an automatically trusted authority; its candidates must
pass the same deterministic evidence policy before acceptance.

The LLM boundary is provider-neutral at the request interface. The standard
Tabulus client uses OpenAI-compatible chat-completions endpoints, disables
thinking, uses deterministic sampling settings, records provider/model
provenance in serialized responses, and reads API keys from environment
variables.

If fallback LLM configuration is present, each adjudication starts with the
primary provider and falls back only after operational failure. Failover does
not bypass deterministic validation.

## Status Boundary

Final artifact statuses are:

- `validated_with_doi`
- `validated_without_doi`
- `rejected`

`rejected` is a scientific decision that means the available evidence was
insufficient for safe assignment under the current resolver. Operational
failures abort the run and leave a resumable checkpoint when possible; they are
not converted into rejected references.

See {doc}`../tutorial/13-doi-resolution` for runnable Step 6 commands and
{doc}`../data-contracts/reference-resolution-json` for the artifact contract.
