# reference_resolution.json

`references/reference_resolution.json` is the canonical Step 6 paper-level
scholarly reference-resolution artifact. It contains one final decision for
each unique bibliography index linked by Step 5 for a paper.

Step 6 writes this artifact only after every target bibliography entry reaches
a final scientific status. Partial work is stored in
`references/reference_resolution.checkpoint.json` and promoted only when the
paper-level run completes.

## Top-Level Fields

`schema_version`
: Contract version for the Step 6 artifact. The current value is `1`.

`resolution_count`
: Number of paper-level bibliography entries resolved in this artifact.

`status_counts`
: Counts for the final statuses `validated_with_doi`,
  `validated_without_doi`, and `rejected`.

`llm_adjudicated_count`
: Number of entries for which a first or second LLM response is serialized.

`retry_count`
: Number of entries for which the single allowed LLM-generated scholarly-search
  retry was used.

`entries`
: Final trace entries sorted by `resolution.reference_index`.

## Entry Fields

Each `entries[]` item records the final decision and the evidence path that led
to it.

`resolution`
: The final Step 6 decision for one bibliography index.

`initial_scholarly_resolution`
: The deterministic Crossref and CORE assessment state before any LLM retry
  search. Crossref evidence is always present; CORE evidence is present when
  Crossref did not validate the reference.

`first_llm_response`
: The first validated LLM response when LLM adjudication was required, or
  `null` when deterministic resolution completed without the model.

`retry_used`
: Whether the LLM used its single allowed `retry_search` action.

`retry_query`
: The LLM-generated retry query when `retry_used` is true; otherwise an empty
  string.

`retry_crossref_assessment`
: Crossref assessment for the retry query, or `null` when no retry was used or
  retry Crossref validation ended the workflow.

`retry_core_assessment`
: CORE assessment for the retry query when retry Crossref evidence was still
  insufficient, or `null` otherwise.

`second_llm_response`
: The final LLM response after one retry when deterministic retry evidence
  remained insufficient, or `null` otherwise. In the finalized contract, this
  response may contain only `select_candidate` or `reject_all`; `retry_search`
  is invalid after the retry has already been consumed.

## Final Resolution Object

`resolution.reference_index`
: One-based Step 4 bibliography index. This is the paper-level key used by
  Step 7 to join Step 6 output back to Step 5 table-cell links.

`resolution.raw_reference`
: Raw reference text copied from Step 4 bibliography evidence.

`resolution.status`
: One of the final statuses below.

`resolution.canonical_doi`
: Normalized DOI for a validated identity, or an empty string when no DOI is
  established.

`resolution.canonical_title`
: Candidate title for a validated identity, or an empty string for rejection.

`resolution.canonical_authors`
: Candidate author names for a validated identity.

`resolution.canonical_year`
: Candidate publication year for a validated identity, or `null` when missing.

`resolution.canonical_venue`
: Candidate venue for a validated identity, or an empty string when missing.

`resolution.source`
: Source of the validated candidate, such as `crossref` or `core`, or an empty
  string for rejection.

`resolution.confidence`
: Deterministic score or LLM confidence associated with an accepted candidate
  when available. It is `null` for rejected entries in the current
  implementation.

`resolution.reason`
: Human-readable reason for the final Step 6 decision.

## Final Statuses

`validated_with_doi`
: Step 6 assigned a validated scholarly identity and established a canonical
  DOI.

`validated_without_doi`
: Step 6 assigned a validated scholarly identity for which no DOI was
  established.

`rejected`
: Step 6 had insufficient evidence to assign a safe scholarly identity.

The final artifact cannot contain the internal intermediate status
`unresolved`. Operational failures do not appear as `rejected`; they abort the
run and preserve checkpointed completed work when possible.

## Candidate And Assessment Evidence

Crossref and CORE candidate records share these fields:

- `source`
- `source_id`
- `doi`
- `title`
- `authors`
- `year`
- `venue`
- `volume`
- `issue`
- `pages`
- `url`

Deterministic assessments include the selected candidate, selected score,
ranked candidates, and a reason. Candidate scores record the normalized score,
comparable field weight, per-field scores, comparable fields,
`sufficient_evidence`, and `strong_match`.

Missing fields are not evidence of disagreement. Explicit contradictions are
recorded through the score and can block acceptance. The final admissibility
gate accepts only candidates with deterministic support such as DOI consistency
where available, title similarity plus independent structured bibliographic
evidence, or sufficiently strong title-less evidence from authors, year, and
publication metadata.

## LLM Response Provenance

Serialized LLM responses include:

- `decision`
- `model`
- `response_id`
- `finish_reason`
- `usage`
- `provider`, when the client supplies one

`decision` contains the accepted bounded action:

- `select_candidate`
- `retry_search`
- `reject_all`

The first LLM-assisted adjudication may use any of the three actions. If
`retry_search` is used, Step 6 performs one bounded scholarly-search retry.
After that retry is consumed, the final LLM-assisted adjudication may return
only `select_candidate` or `reject_all`. A final `retry_search` violates the
adjudication contract and is treated as an invariant/error condition, not as a
scientific rejection.

The LLM response does not add scholarly identity fields directly. A selected
candidate must be one of the candidates supplied by Tabulus and must pass the
final deterministic admissibility gate before the final `resolution` can be
validated. Explicit DOI contradiction prevents acceptance, and sufficient
bibliographic evidence remains a deterministic requirement even for
LLM-selected candidates.

API keys and authorization headers are not serialized.


## Non-Atomic Reference Guard

Step 6 resolves one scholarly work per bibliography entry. If a raw
bibliography entry appears to contain multiple complete citation-like
publication records or document-layout contamination, the final resolution is a
conservative `rejected` decision because single-work scholarly resolution would
be unsafe. Known bracketed original-language/translated-journal pairs are
permitted when they satisfy the implemented compatibility conditions. These are
scientific safety decisions, not operational failures.

## Controlled Evaluation Boundary

`reference_resolution.json` reports final Step 6 statuses and provenance. A
`validated_with_doi` status is a resolver output status, not an accuracy label.
Validation yield must not be reported as precision, recall, DOI accuracy, or
scholarly-identity accuracy without an independent identity gold standard. See
{doc}`../tutorial/13-doi-resolution` for the controlled P251/P252 component
setup and run protocol.

## Checkpoint Contract

During incomplete runs, Step 6 writes:

```text
<artifact-root>/references/reference_resolution.checkpoint.json
```

The checkpoint stores:

- `schema_version`
- `run_fingerprint`
- `target_count`
- `target_indices`
- `completed_count`
- serialized completed `entries`

On restart, Tabulus loads the checkpoint only when its fingerprint and target
set match the current run. Compatible completed entries are skipped. The
checkpoint is removed after the final `reference_resolution.json` artifact is
written successfully.

The run fingerprint covers Step 6 source files, bibliography contents, the set
of Step 5 match artifact contents, optional document-context contents, and
public resolver configuration relevant to reproducibility. Credentials and
Crossref contact email are excluded.

## Relationship To Other Artifacts

`bibliography.json`
: Immutable Step 4 extraction evidence from GROBID.

`reference_matches.json`
: Step 5 table-cell links to bibliography positions.

`reference_resolution.json`
: Step 6 paper-level validated identities or conservative rejections for the
  union of linked bibliography indices.

Step 7 joins these paper-level identities back to Step 5 physical-row matches
without re-resolving the same bibliography index. See {doc}`resolved-csv` for
the resolved CSV export contract.
