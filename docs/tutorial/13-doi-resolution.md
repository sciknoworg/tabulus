# Stage 6: Scholarly Reference Resolution

## Goal

Resolve bibliography entries linked by Stage 5 to conservative scholarly
identities at paper scope. Stage 6 consumes Stage 4 extraction evidence and
Stage 5 table-cell links, retrieves scholarly metadata from external services,
and writes one paper-level registry at:

```text
<artifact-root>/references/reference_resolution.json
```

Stage 6 does not modify `references/bibliography.json` or
`references/reference_matches.json`. It resolves each unique bibliography
index once for a paper, even when several cells, selected tables, or
reconstruction adapters point to the same index.

```text
PAPER
  |
  +--> table branch
  |      |
  |      v
  |    Stage 3 selected_reference_tables.json
  |      |
  |      v
  |    Stage 5 reference_matches.json
  |
  +--> bibliography branch
         |
         v
       Stage 4 bibliography.json

union of Stage 5 matched bibliography indices
  |
  v
Stage 6 reference_resolution.json
  |
  v
Stage 7 join / resolved export (planned)
```

Stage 4 is extraction. Stage 5 is deterministic table-cell to bibliography
position matching. Stage 6 is scholarly-identity resolution.

## Inputs

Required inputs:

- Stage 4 {doc}`../data-contracts/bibliography-json`
- at least one Stage 5 {doc}`../data-contracts/reference-matches-json` artifact
  for the same paper
- an artifact root passed through `--out`

Optional input:

- `references/reference_context.json`, a document-context artifact containing
  body-text citation contexts extracted from MinerU content. When omitted,
  document context is disabled.

Stage 6 collects `matched_reference_indices` from all supplied Stage 5
artifacts, validates that every index exists in `bibliography.json`, unions and
deduplicates the set, and resolves targets in bibliography-index order.

## Resolution Workflow

For each unique linked bibliography entry, Stage 6 runs this conservative
workflow:

```text
Stage 4 bibliography evidence
  |
  v
existing DOI Crossref lookup, if a DOI was extracted
  |
  v
Crossref bibliographic search, if DOI lookup did not validate a work
  |
  v
deterministic Crossref assessment
  |
  +--> validated candidate, when evidence is strong and unambiguous
  |
  v
CORE search and deterministic assessment
  |
  +--> validated candidate, when evidence is strong and unambiguous
  |
  v
bounded LLM adjudication over supplied candidates
  |
  +--> select a supplied candidate
  |      |
  |      v
  |    final deterministic admissibility gate
  |
  +--> reject all candidates
  |
  `--> propose one improved scholarly-search query
         |
         v
       one Crossref/CORE retry, then deterministic assessment or one final
       LLM adjudication over the combined candidate evidence
```

An exact DOI returned by Crossref for an extracted DOI is decisive. Otherwise,
Crossref and CORE candidates must pass deterministic scoring rules that compare
only fields present on both sides. Missing bibliographic fields do not count as
disagreement. Explicit contradictions, such as conflicting DOIs or incompatible
numbered-edition years, block automatic acceptance.

Strong title similarity alone is not enough for final acceptance. A candidate
must also satisfy Tabulus's minimum bibliographic-evidence requirement. The LLM
cannot lower that requirement.

## LLM Boundary

The LLM step is evidence-bounded. The model receives structured source
evidence, ranked Crossref and CORE candidates, and optional document contexts.
It may only return one of these decisions:

- `select_candidate`
- `retry_search`
- `reject_all`

For `select_candidate`, the model must choose a `candidate_id` supplied by
Tabulus. It may not invent a DOI, title, author, venue, publication, or
scholarly entity. For `retry_search`, it may propose one improved search query.
If that retry does not lead to deterministic validation, Stage 6 allows one
final LLM adjudication over the combined initial and retry candidates. A second
retry request is rejected because the retry budget is exhausted.

LLM response content is strictly parsed. Unknown identity-bearing fields are
rejected. Syntactically invalid or truncated JSON message content is retried
within the bounded retry policy; semantic contract violations remain hard
operational failures.

## Status Semantics

The final artifact contains only final scientific statuses:

`validated_with_doi`
: Stage 6 assigned a validated scholarly identity and established a canonical
  DOI.

`validated_without_doi`
: Stage 6 assigned a validated scholarly identity for which no DOI was
  established.

`rejected`
: Stage 6 had insufficient evidence to assign a safe scholarly identity under
  the current single-work resolution model.

Operational failures are different from `rejected`. Provider outages, invalid
provider responses, exhausted operational retries, invalid inputs, interrupted
runs, or checkpoint incompatibilities abort the command after checkpointing
completed references where possible. They are not silently converted into
scientific rejection decisions.

## Configuration

Stage 6 uses Crossref, CORE, and an OpenAI-compatible LLM endpoint. The CLI
reads secrets from environment variables and prints only environment-variable
names for API keys. It does not print API-key values.

Required configuration:

- Crossref contact email through `--crossref-mailto` or
  `TABULUS_CROSSREF_MAILTO`
- CORE API key through the environment variable named by `--core-api-key-env`
  (default: `CORE_API_KEY`)
- primary LLM base URL through `--llm-base-url` or `TABULUS_LLM_BASE_URL`
- primary LLM model through `--llm-model` or `TABULUS_LLM_MODEL`
- primary LLM API key through the environment variable named by
  `--llm-api-key-env` (default: `TABULUS_LLM_API_KEY`)

Optional fallback LLM configuration:

- `--fallback-llm-base-url` or `TABULUS_FALLBACK_LLM_BASE_URL`
- `--fallback-llm-model` or `TABULUS_FALLBACK_LLM_MODEL`
- fallback API key through the environment variable named by
  `--fallback-llm-api-key-env` (default: `TABULUS_FALLBACK_LLM_API_KEY`)

Fallback configuration is all-or-nothing. If any fallback base URL, model, or
API-key value is supplied, all three must be present. When fallback
configuration is absent, Stage 6 uses only the primary LLM provider. When it is
present, every adjudication starts with the primary provider and falls back for
that adjudication only after the primary provider fails its bounded attempts.
The next adjudication starts with the primary provider again.

The standard client records provider names in serialized LLM responses so a
run can be audited. Current standard-client provider labels are `kisski` for
the primary client and `openrouter` for the fallback client.

## Single-Paper Execution

Run Stage 6 with explicit Stage 5 artifacts when you already know which
reconstruction outputs belong to the paper:

```bash
tabulus resolve-references \
  --bibliography /path/to/artifact-root/references/bibliography.json \
  --reference-matches /path/to/reconstruction-a/references/reference_matches.json \
  --reference-matches /path/to/reconstruction-b/references/reference_matches.json \
  --out /path/to/artifact-root
```

Add optional document context when a compatible artifact exists:

```bash
tabulus resolve-references \
  --bibliography /path/to/artifact-root/references/bibliography.json \
  --reference-matches /path/to/reconstruction/references/reference_matches.json \
  --reference-context /path/to/artifact-root/references/reference_context.json \
  --out /path/to/artifact-root
```

Use paper-level discovery when the Stage 5 artifacts follow the supported
experiment layout:

```text
<reference-matches-root>/<adapter>/<run>/reconstruction/<paper>/<adapter>/references/reference_matches.json
```

```bash
tabulus resolve-references \
  --bibliography /path/to/artifact-root/references/bibliography.json \
  --reference-matches-root /path/to/reference-matches-root \
  --paper "<paper-directory-name>" \
  --out /path/to/artifact-root
```

Discovery selects one `reference_matches.json` per adapter for the exact paper
name. If more than one artifact is found for the same adapter, Tabulus refuses
to choose among historical runs implicitly; pass explicit `--reference-matches`
paths instead.

A complete run writes `references/reference_resolution.json` beneath the
artifact root and removes the matching checkpoint. The command prints the
number of unique bibliography entries processed, status counts, LLM-adjudicated
entries, retry-search count, and final output path.

## Multi-Paper Execution

The CLI does not provide a native multi-paper `resolve-references` batch mode.
Process multiple papers by orchestrating independent single-paper invocations.
Each paper keeps its own Stage 4 bibliography, Stage 5 match artifacts, Stage 6
checkpoint, and final paper-level registry.

A generic shell pattern is:

```bash
while IFS=, read -r paper_name artifact_root reference_matches_root; do
  [ "$paper_name" = "paper_name" ] && continue

  tabulus resolve-references \
    --bibliography "$artifact_root/references/bibliography.json" \
    --reference-matches-root "$reference_matches_root" \
    --paper "$paper_name" \
    --out "$artifact_root"
done < /path/to/stage6-papers.csv
```

Use neutral manifests that contain only local paths and paper names. Keep API
keys in the environment instead of embedding them in scripts, manifests, logs,
or artifacts. A completed paper has
`<artifact-root>/references/reference_resolution.json`. If a paper fails for an
operational reason, its checkpoint remains under the same artifact root and can
be resumed by rerunning the same command with compatible inputs and
configuration. Other completed papers do not need to be rerun.

## Checkpointing And Reproducibility

During a run, Stage 6 writes:

```text
<artifact-root>/references/reference_resolution.checkpoint.json
```

Completed references are checkpointed incrementally. On restart, a compatible
checkpoint is loaded and completed bibliography indices are skipped. The final
`reference_resolution.json` is written only after every target reaches a final
scientific status; until then, the canonical final artifact is not promoted.
After successful final writing, the checkpoint is removed.

The checkpoint fingerprint covers:

- Stage 6 source files in `src/tabulus/reference_resolution/`
- the Stage 4 bibliography artifact contents
- the set of Stage 5 reference-match artifact contents, independent of the
  order in which they were supplied
- the optional reference-context artifact contents, or the disabled-context
  marker
- reproducibility-relevant resolver configuration such as LLM provider labels,
  base URLs, model identifiers, and thinking/fallback policy

Credentials and Crossref contact email are not included in the fingerprint.
Source changes intentionally invalidate old checkpoints.

## Output

See {doc}`../data-contracts/reference-resolution-json` for the full artifact
contract. The final artifact stores one entry per resolved bibliography index,
including:

- the final `resolution` object
- initial Crossref/CORE scholarly-resolution evidence
- first and optional second LLM response provenance
- retry-search state and retry Crossref/CORE assessments when used

Stage 7 remains unimplemented. Its planned responsibility is to join the
validated paper-level identities back to every relevant table cell or reference
occurrence and produce downstream export artifacts.
