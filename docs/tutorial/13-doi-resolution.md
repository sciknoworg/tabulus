# Planned Stage 6: Scholarly Reference Resolution

## Current Status

The rebuilt `src/tabulus` package in this checkout does not currently implement
Stage 6 scholarly reference resolution. There is no Stage 6 CLI command,
resolver package, provider client, status model, checkpoint writer, or
`references/reference_resolution.json` writer in the current implementation.

The implemented reference-processing pipeline currently ends at Stage 5
`references/reference_matches.json`.

## Planned Goal

Stage 6 is the planned paper-level stage after deterministic reference
matching. Its purpose is to resolve matched bibliography entries to scholarly
identities without changing the Stage 4 bibliography artifact or Stage 5 match
artifact.

The planned resolution key is paper-level:

```text
(paper, bibliography_index)
```

That boundary matters because multiple selected table cells, tables, and
reconstruction adapters may point to the same bibliography entry. Resolution
should operate once per unique bibliography entry for a paper, rather than once
per table occurrence.

## Planned Input

The planned inputs are:

- Stage 4 `references/bibliography.json`
- one or more Stage 5 `references/reference_matches.json` artifacts for the
  paper

Stage 4 bibliography entries are extraction evidence. Stage 5 match artifacts
are linkage evidence. A future resolver should keep those roles separate from
scholarly metadata retrieved during resolution.

## Planned Output

The planned canonical output is:

```text
references/reference_resolution.json
```

Because Stage 6 is not implemented in this checkout, this documentation does
not define final status names, serialized provider provenance, checkpoint
schema, or retry semantics as current behavior. Those details should be
documented from the implementation when the resolver is added to `src/tabulus`.

## Boundary

Stage 6 is separate from:

- Stage 4 bibliography extraction, which sends the original PDF to GROBID and
  writes immutable extraction evidence
- Stage 5 reference matching, which deterministically links table-cell
  references to bibliography positions offline
- Stage 7 export, which is planned to join resolved identities back to table
  content

The current rebuilt pipeline does not call Crossref, CORE, LLM providers,
embedding services, or scholarly search APIs in Stage 4 or Stage 5.

## Planned Implementation Requirements

When Stage 6 is implemented, documentation should be updated from the source
code to describe:

- paper-level union and deduplication of linked bibliography indices
- deterministic evidence used before any model adjudication
- any external scholarly metadata providers used by the resolver
- whether and how bounded LLM adjudication is used
- the final deterministic admissibility gate
- scientific rejection semantics versus operational failure semantics
- provider failover and retry behavior
- provider/model provenance recorded in serialized outputs
- checkpointing, resumability, and all-or-nothing final artifact creation
- configuration and credential handling without exposing secret values
- reproducibility-relevant configuration fingerprinting

Stage 7 remains unimplemented. Its planned responsibility is to join resolved
paper-level identities back to every relevant table cell or reference
occurrence and produce downstream export artifacts.
