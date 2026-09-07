# Stage 6: Paper-Level Scholarly Reference Resolution

## Goal

Validate and enrich the unique bibliography entries that selected tables
actually referenced.

## Input

Stage 5 `references/reference_matches.json` artifacts and the paper-level
`references/bibliography.json` artifact.

Stage 4 extracts the complete bibliography. Stage 5 determines which
bibliography entries are actually referenced by selected tables. Stage 6 takes
the union of matched bibliography indices across all reconstruction methods
for a paper and deduplicates by bibliography index. It resolves each target
once per paper, not once per table fragment, method, or cell occurrence.

The bibliography remains immutable extraction evidence. Optional document
context can also be supplied to resolution.

The resolution key is:

```text
(paper, bibliography_index)
```

## Output

One paper-level scholarly-resolution registry:

```text
references/
  reference_resolution.json
```

Each targeted bibliography index has one final scholarly-resolution outcome
for that paper:

- `validated_with_doi`: a validated scholarly identity with an established DOI
- `validated_without_doi`: a legitimate validated scholarly identity for which
  no DOI is established
- `rejected`: insufficient evidence to assign a safe scholarly identity

Operational failures are not converted into rejected references. They abort
the run while preserving completed work in a checkpoint.

## Default Implementation

Stage 6 is implemented in the GPU-cluster version described in
{doc}`../project-notes/current-state`. The local documentation checkout does
not yet contain that implementation; no unverified Stage 6 CLI command is
specified here.

```text
existing DOI -> Crossref DOI validation
    | if unresolved
    v
Crossref bibliographic candidates
    | strong unique match -> accept
    | otherwise
    v
CORE fallback
    | strong unique match -> accept
    | otherwise
    v
bounded LLM adjudication / query reformulation
    |
    v
final deterministic admissibility gate
    |
    v
validated candidate or rejection
```

All accepted candidates must satisfy the deterministic evidence policy.
Crossref, CORE, and LLM calls belong to Stage 6. Stage 4 only extracts
bibliography evidence; Stage 5 only matches table-cell citations to
bibliography positions offline. Reconstruction prediction CSVs remain intact.

## Evidence Policy

Resolution is deliberately conservative: false negatives and unresolved
references are preferable to false-positive DOI contamination. Missing
bibliographic fields do not count as disagreement; explicit contradictions
do. An exact DOI, when successfully validated, is decisive. Strong title
similarity alone is insufficient without independent bibliographic support.

The LLM may select only supplied candidates, reject all candidates, or propose
one better search query. There is at most one LLM-generated scholarly-search
retry. It may not invent a DOI, title, publication, or scholarly entity, and
cannot lower Tabulus's deterministic minimum-evidence requirement.

## Scientific Safeguards

### Chapter And Containing-Book Titles

GROBID can extract a container/book title as the title of a chapter citation.
For raw references with the structure `author, in <container>, edited by ...,
pp. ...`, that container title is excluded from title-comparison evidence.
Independent agreement such as author, year, and page span can still support a
chapter identity.

For example, `N. J. Mason, in Atomic Layer Epitaxy ... pp. 63-109.` resolves
through Crossref to the chapter "Comparison of ALE with other techniques",
DOI `10.1007/978-94-009-0389-0_3`, using author/year/pages. The whole-book DOI
is not accepted merely because its title matches the extracted container.

### Original-Language And Translated-Journal Pairs

An original-language journal citation followed by a bracketed English
translation can describe one scholarly work. The structural exception requires
exactly two complete citation tails, a bracketed second tail, agreeing volumes,
and equal or adjacent years. Four-digit page values are not treated as
publication years.

Genuine multiple-work concatenations remain non-atomic and are rejected under
the current single-work resolution model.

### Edition And Year Conflicts

A numbered-edition citation identifies a specific bibliographic manifestation.
If the source explicitly specifies a numbered edition and both source and
candidate supply different years, the candidate cannot satisfy minimum
evidence. For example, `H. S. Fogler, Elements of Chemical Reaction Engineering,
2nd ed. ..., 1992.` must not accept a same-title/same-author candidate from
2020 as the cited manifestation. The LLM cannot explain away this contradiction.

## Operational Robustness

Transient raw `TimeoutError` failures are retried with bounded exponential
backoff. Syntactically malformed or truncated JSON in `message.content` is
retried within the same bounded policy. Semantic contract violations remain
hard failures; they are not retried until a desired answer appears. These
operational retries are distinct from the single allowed scholarly-search retry.

## Resumability

Completed references are checkpointed incrementally in:

```text
references/reference_resolution.checkpoint.json
```

On restart, completed references are skipped. The final
`references/reference_resolution.json` is written only after all targets
complete successfully, including targets with a final `rejected` status. The
checkpoint is then removed.

The checkpoint fingerprint covers bibliography contents, Stage 5 reference-match
artifact contents, optional document-context input, the Stage 6 source
implementation, and resolver configuration such as model and base URL. API keys
and email addresses are excluded. Source changes intentionally invalidate old
checkpoints.

## Verification

The step completes when every target has a single recorded outcome and
rejections remain traceable. A validated identity meets the implemented
evidence policy; successful resolution is not proof of correctness. Report
resolution coverage, consistency, and agreement. Accuracy requires a human
gold standard.

Earlier JVSTA Stage 6 runs are provisional diagnostics. The clean final rerun
with enriched Stage 4 artifacts is underway; no final Stage 6 coverage or result
percentages are available. See {doc}`../project-notes/current-state` for the
canary and checkpoint-resume status.

Stage 7 is not implemented. Its next task is to deterministically join this
paper-level registry back to every relevant table cell/reference occurrence
and produce downstream export artifacts.
