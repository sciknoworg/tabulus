# Step 6: Scholarly Reference Resolution

## Goal

Resolve bibliography entries linked by Step 5 to conservative scholarly
identities at paper scope. Step 6 consumes Step 4 extraction evidence and
Step 5 table-cell links, retrieves scholarly metadata from external services,
and writes one paper-level registry at:

```text
<artifact-root>/references/reference_resolution.json
```

Step 6 does not modify `references/bibliography.json` or
`references/reference_matches.json`. It resolves each unique bibliography
index once for a paper, even when several cells, selected tables, or
reconstruction adapters point to the same index.

```text
PAPER
  |
  +--> table branch
  |      |
  |      v
  |    Step 3 selected_reference_tables.json
  |      |
  |      v
  |    Step 5 reference_matches.json
  |
  +--> bibliography branch
         |
         v
       Step 4 bibliography.json

union of Step 5 matched bibliography indices
  |
  v
Step 6 reference_resolution.json
  |
  v
Step 7 resolved CSV export
```

Step 4 is extraction. Step 5 is deterministic table-cell to bibliography
position matching. Step 6 is scholarly-identity resolution.

## Inputs

Required inputs:

- Step 4 {doc}`../data-contracts/bibliography-json`
- at least one Step 5 {doc}`../data-contracts/reference-matches-json` artifact
  for the same paper
- an artifact root passed through `--out`

Optional input:

- `references/reference_context.json`, a document-context artifact containing
  body-text citation contexts extracted from MinerU content. When omitted,
  document context is disabled.

Step 6 collects `matched_reference_indices` from all supplied Step 5
artifacts, validates that every index exists in `bibliography.json`, unions and
deduplicates the set, and resolves targets in bibliography-index order.

## Resolution Workflow

For each unique linked bibliography entry, Step 6 runs this conservative
workflow:

```text
Step 4 bibliography evidence + Step 5 linked bibliography indices
  |
  v
collect unique paper-level Step 6 targets
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
bounded LLM-assisted adjudication over supplied candidates
  |
  +--> select_candidate
  |      |
  |      v
  |    final deterministic admissibility gate
  |
  +--> reject_all
  |
  `--> retry_search
         |
         v
       one Crossref/CORE scholarly-search retry
         |
         v
       deterministic assessment or final LLM-assisted adjudication
       over combined initial and retry evidence
       (select_candidate or reject_all only)
```

The first LLM-assisted adjudication may return exactly one of
`select_candidate`, `retry_search`, or `reject_all`. Only one bounded scholarly
search retry is permitted. If the retry is consumed and deterministic retry
evidence remains insufficient, the final LLM-assisted adjudication may return
only `select_candidate` or `reject_all`; `retry_search` is no longer a valid
final decision. A custom or adversarial client that returns `retry_search` after
the retry has already been consumed violates the adjudication contract and is
treated as an invariant/error condition rather than as a scientific `rejected`
outcome.

An exact DOI returned by Crossref for an extracted DOI is decisive. Otherwise,
Crossref and CORE candidates must pass deterministic scoring rules that compare
only fields present on both sides. Missing bibliographic fields do not count as
disagreement. Explicit contradictions, such as conflicting DOIs or incompatible
numbered-edition years, block acceptance.

The final acceptance boundary is deterministic. The LLM proposes or adjudicates
among supplied candidates, but a selected candidate is accepted only when it
passes the final admissibility gate. Deterministic evidence rules include DOI
consistency where available, title similarity together with independent
structured bibliographic evidence, or sufficiently strong title-less evidence
from authors, year, and publication metadata. Strong title similarity alone is
not enough, and the LLM cannot lower Tabulus's minimum bibliographic-evidence
requirement.

Step 6 also guards against bibliography entries that appear to contain multiple
complete citation-like publication records or document-layout contamination.
Such non-atomic entries are rejected when single-work scholarly resolution would
be unsafe. The implementation deliberately permits known bracketed
translation-pair patterns when they satisfy the compatibility conditions in the
resolver. These safety rejections are scientific safeguards, not resolver
crashes or operational failures.

## LLM Processing Boundary

LLM processing is evidence-bounded. The model receives structured source
evidence, ranked Crossref and CORE candidates, and optional document contexts.
For `select_candidate`, it must choose a `candidate_id` supplied by Tabulus. It
may not invent a DOI, title, author, venue, publication, or scholarly entity.
For `retry_search`, it may propose one improved search query, subject to the
single bounded retry described above.

LLM response content is strictly parsed. Unknown identity-bearing fields are
rejected. Syntactically invalid or truncated JSON message content is retried
within the bounded retry policy; semantic contract violations remain hard
operational failures.

## Status Semantics

The final artifact contains only final scientific statuses:

`validated_with_doi`
: Step 6 assigned a validated scholarly identity and established a canonical
  DOI.

`validated_without_doi`
: Step 6 assigned a validated scholarly identity for which no DOI was
  established.

`rejected`
: Step 6 had insufficient evidence to assign a safe scholarly identity under
  the current single-work resolution model.

Operational failures are different from `rejected`. Provider outages, invalid
provider responses, exhausted operational retries, invalid inputs, interrupted
runs, or checkpoint incompatibilities abort the command after checkpointing
completed references where possible. They are not silently converted into
scientific rejection decisions.

## Configuration

Step 6 uses Crossref, CORE, and an OpenAI-compatible LLM endpoint. The CLI
reads secrets from environment variables and prints only environment-variable
names for API keys. It does not print API-key values.

Required configuration:

- Crossref contact email through `--crossref-mailto` or
  `TABULUS_CROSSREF_MAILTO`
- CORE API key through the environment variable named by `--core-api-key-env`
  (default: `CORE_API_KEY`)
- primary LLM provider label through `--llm-provider` or
  `TABULUS_LLM_PROVIDER`; if unset, `openai-compatible` is recorded
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
configuration is absent, Step 6 uses only the primary LLM provider. When it is
present, every adjudication starts with the primary provider and falls back for
that adjudication only after the primary provider fails its bounded attempts.
The next adjudication starts with the primary provider again.

The provider-neutral implementation uses an OpenAI-compatible chat-completions
interface and records provider/model provenance in serialized LLM responses.
The current controlled evaluation uses provider `saia`, model `qwen3.8-27b`,
and base URL `https://chat-ai.academiccloud.de/v1`. Do not place API keys,
tokens, or secret values in commands, manifests, logs, or documentation.

Model-specific reasoning mode is an implementation option and is disabled in
the current controlled evaluation. The standard client disables that capability
for the primary configuration used by Step 6 and records `reasoning_mode=disabled`
in the resolver fingerprint.

## Single-Paper Execution

Run Step 6 with explicit Step 5 artifacts when you already know which
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

Use paper-level discovery when the Step 5 artifacts follow the supported
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
Each paper keeps its own Step 4 bibliography, Step 5 match artifacts, Step 6
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

During a run, Step 6 writes:

```text
<artifact-root>/references/reference_resolution.checkpoint.json
```

Completed references are checkpointed incrementally. On restart, a compatible
checkpoint is loaded and completed bibliography indices are skipped. The final
`reference_resolution.json` is written only after every target reaches a final
scientific status; until then, the canonical final artifact is not promoted.
After successful final writing, the checkpoint is removed.

The checkpoint fingerprint covers:

- Step 6 source files in `src/tabulus/reference_resolution/`
- the Step 4 bibliography artifact contents
- the set of Step 5 reference-match artifact contents, independent of the
  order in which they were supplied
- the optional reference-context artifact contents, or the disabled-context
  marker
- reproducibility-relevant resolver configuration such as LLM provider labels,
  base URLs, model identifiers, LLM processing policy, retry policy, and
  fallback policy

Credentials and Crossref contact email are not included in the fingerprint.
Source changes intentionally invalidate old checkpoints.


## Controlled Step 6 Evaluation

The controlled Step 6 evaluation is component-focused. It is distinct from a
true full-production run over all 252 papers. In production, Step 6 runs after
Step 1 production detections, Step 2 production reconstructions, Step 3
classification, one Step 4 bibliography per paper, Step 5 per adapter, and the
paper-level union/deduplication of linked bibliography entries.

For the controlled P251/P252 evaluation, the original controlled Step 5 gold
bibliography artifacts contain bibliography indices and raw references but do
not provide enough structured bibliographic metadata for a fair component-level
resolver evaluation. The final Step 4 component-aware alignment artifact is
therefore used as a bridge between the controlled Step 5 bibliography indices
and the corresponding Step 4/GROBID bibliography entries. Only clean one-to-one
Step 4 to controlled-gold bibliography alignments are used as resolver inputs.

Canonical gold artifacts remain immutable. Derived controlled Step 6 inputs are
stored under:

```text
runs/stage6/controlled-p251-p252-final/<paper>/input_one_to_one/
```

Each controlled input directory contains:

- `bibliography.json`
- `reference_matches.json`
- `manifest.json`

The manifest records provenance and excluded alignment cases.

Current controlled populations are:

| Paper | Original linked bibliography targets | One-to-one controlled Step 6 targets | Excluded upstream alignment cases |
| --- | ---: | ---: | --- |
| P251 | 2,288 | 2,280 | 8 total: 4 split, 4 merge |
| P252 | 1,053 | 996 | 57 total: 53 merge, 3 split, 1 unmatched |

The excluded split, merge, and unmatched cases are not silently discarded from
the broader methodological accounting. They are upstream
bibliography-segmentation/alignment cases and are kept separate from the clean
Step 6 component evaluation. The one-to-one subset must not be described as the
full end-to-end pipeline population.

### Independent identity gold

Step 6 scholarly-identity accuracy requires an independent scholarly identity
gold standard. Step 5 gold links establish which bibliography entry was cited;
they do not by themselves establish the canonical scholarly-work identity.
Therefore resolver validation status is not equivalent to correctness,
`validated_with_doi` is an output status rather than an accuracy label, and
validation yield must not be reported as precision, recall, or DOI accuracy.
Independent identity gold is required before those accuracy metrics can be
computed.

### Controlled run protocol

Step 6 is a hybrid workflow rather than a fully deterministic computation.
Deterministic components include target collection, deduplication,
bibliographic scoring, final admissibility, retry bounds, and final status
logic. Complete executions can vary because they depend on live Crossref
results, live CORE results, external provider/network behavior, and
LLM-assisted adjudication.

For controlled Step 6 evaluation, perform two fresh complete runs per paper:

```text
P251/run_01
P251/run_02
P252/run_01
P252/run_02
```

Each run starts independently from the same controlled inputs. A second run
must not reuse the first run's checkpoint. Run-to-run consistency should be
assessed at the individual bibliography-reference level, including final
resolution-status agreement, exact DOI agreement, agreement on validated
scholarly identity, retry-use agreement, LLM-decision agreement where
applicable, and the number/type of references whose final outcome changes. For
references validated in both runs, exact DOI agreement is particularly
important.

Observed Step 6 runtime is part of the computer-science evaluation. Report it
as observed end-to-end wall-clock runtime because Step 6 includes remote
service latency from Crossref, CORE, and the LLM provider. Where instrumented,
record wall-clock runtime, processed references, seconds/reference, number of
LLM-processed entries, number of bounded retries, and optionally CPU time and
maximum resident memory. Do not describe observed Step 6 wall-clock runtime as
pure computation time.

### Current controlled P252 run

The current canonical first valid P252 controlled run is:

```text
runs/stage6/controlled-p251-p252-final/P252/run_01/
```

This is the first scientifically valid controlled execution after repairing the
controlled Step 6 inputs and fixing the final-adjudication retry contract. Its
final output contains:

| Quantity | Count |
| --- | ---: |
| unique targets | 996 |
| validated with DOI | 761 |
| validated without DOI | 0 |
| rejected | 235 |
| LLM-adjudicated entries | 794 |
| scholarly-search retries used | 139 |
| old retry-budget-exhausted scientific rejection | 0 |

The 235 rejected entries comprise:

- 117 where no candidate could be validated after Crossref, CORE, one bounded
  retry, and final LLM adjudication
- 100 where the first LLM-selected candidate failed the deterministic
  admissibility gate
- 11 where the final LLM-selected candidate failed the deterministic
  admissibility gate
- 7 non-atomic bibliography entries

First LLM decisions:

- 655 `select_candidate`
- 139 `retry_search`

Final LLM decisions after retry:

- 117 `reject_all`
- 21 `select_candidate`
- 0 `retry_search`

This confirms that the corrected final-adjudication contract is being enforced.
The 761/996 result is a resolution/validation yield, or the proportion of
targets emitted as `validated_with_doi`; it is not accuracy.

Do not add P251 outcome numbers yet. P251 `run_01` is currently in progress and
its final outcome has not been established. In the canonical controlled naming,
`run_01` is the first valid controlled execution and `run_02` is the second
fresh controlled execution for consistency evaluation. Earlier debugging
attempts are not part of the canonical controlled-results tree and should not
be documented as experimental replicates.

## Output

See {doc}`../data-contracts/reference-resolution-json` for the full artifact
contract. The final artifact stores one entry per resolved bibliography index,
including:

- the final `resolution` object
- initial Crossref/CORE scholarly-resolution evidence
- first and optional second LLM response provenance
- retry-search state and retry Crossref/CORE assessments when used

Step 7 consumes this paper-level registry together with Step 5 matches and
exports resolved CSV files without rerunning scholarly lookup. See
{doc}`14-csv-export` and {doc}`../data-contracts/resolved-csv`.
