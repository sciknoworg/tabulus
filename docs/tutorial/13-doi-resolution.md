# Step 6: Scholarly Reference Resolution

## Goal

Resolve bibliography entries linked by Step 5 to conservative scholarly
identities at paper scope. Step 6 consumes the Step 4 bibliography and one or
more Step 5 reference-match artifacts, collects the union of linked
bibliography indices for a paper, deduplicates them, and resolves each unique
cited bibliography entry once.

Step 6 writes one paper-level registry at:

```text
<artifact-root>/references/reference_resolution.json
```

Step 6 does not modify `references/bibliography.json`,
`references/reference_matches.json`, or reconstruction prediction CSVs. Step 7
later consumes the resolved scholarly identities deterministically; it does not
re-resolve references.

```text
Step 4 references/bibliography.json
            \
             -> Step 6 scholarly identity resolution
            /
Step 5 references/reference_matches.json
             |
             v
references/reference_resolution.json
             |
             v
Step 7 resolved CSV export
```

## Inputs

Required inputs:

- Step 4 {doc}`../data-contracts/bibliography-json`
- one or more Step 5 {doc}`../data-contracts/reference-matches-json` artifacts
  for the same paper
- an artifact root passed through `--out`

Optional input:

- `references/reference_context.json`, a document-context artifact containing
  body-text citation contexts extracted from MinerU content. When omitted,
  document context is disabled.

Step 6 collects `matched_reference_indices` from all supplied Step 5 artifacts,
validates that every index exists in `bibliography.json`, unions and
deduplicates the set, and resolves targets in bibliography-index order. This is
paper-level processing, not one resolver call per table occurrence.

## Resolution Workflow

For each unique linked bibliography entry, Step 6 runs this conservative
workflow:

1. Load the Step 4 bibliography.
2. Load one or more Step 5 `reference_matches.json` artifacts.
3. Collect and deduplicate linked bibliography indices for the paper.
4. Attempt scholarly resolution with Crossref.
5. Fall back to CORE where required.
6. Use bounded LLM-assisted adjudication when deterministic evidence is
   insufficient.
7. Permit at most one scholarly-search retry.
8. Apply the deterministic final admissibility gate before accepting an
   LLM-selected candidate.
9. Emit unresolved or unsafe entries as `rejected`.

The first LLM-assisted adjudication may return exactly one of
`select_candidate`, `retry_search`, or `reject_all`. If `retry_search` is used,
Step 6 performs one bounded Crossref/CORE scholarly-search retry. After that
retry has been consumed, the final LLM-assisted adjudication may return only
`select_candidate` or `reject_all`; `retry_search` is no longer valid.

An LLM-selected candidate is not accepted automatically. The final acceptance
boundary is deterministic. Deterministic evidence rules include DOI consistency
where available, title similarity together with independent structured
bibliographic evidence, or sufficiently strong title-less evidence from
authors, year, and publication metadata. Explicit DOI contradiction prevents
acceptance, and the LLM cannot lower Tabulus's minimum bibliographic-evidence
requirement.

Step 6 also guards against bibliography entries that appear to contain multiple
complete citation-like publication records or document-layout contamination.
Such non-atomic entries are rejected when single-work scholarly resolution
would be unsafe. Known bracketed original-language/translated-journal pairs are
permitted when they satisfy the implemented compatibility conditions. These are
scientific safety decisions, not resolver crashes or operational failures.

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

Common environment variables are:

- `TABULUS_CROSSREF_MAILTO`
- `CORE_API_KEY`
- `TABULUS_LLM_PROVIDER`
- `TABULUS_LLM_BASE_URL`
- `TABULUS_LLM_MODEL`
- `TABULUS_LLM_API_KEY`

The corresponding CLI options include `--crossref-mailto`, `--llm-base-url`, `--llm-model`, `--llm-provider`, `--core-api-key-env`, and `--llm-api-key-env`. Inspect the exact current interface with:

```bash
tabulus resolve-references --help
```

For example, a provider-neutral OpenAI-compatible configuration can be supplied
as:

```bash
export TABULUS_CROSSREF_MAILTO="name@example.org"
export CORE_API_KEY="<set in environment>"
export TABULUS_LLM_PROVIDER="saia"
export TABULUS_LLM_BASE_URL="https://chat-ai.academiccloud.de/v1"
export TABULUS_LLM_MODEL="qwen3.8-27b"
export TABULUS_LLM_API_KEY="<set in environment>"
```

The provider, model, and base URL above are configuration examples, not
hard-coded requirements. Do not place API keys, tokens, or secret values in
commands, manifests, logs, or documentation. The current controlled
configuration disables model-specific reasoning mode and records that setting
in the resolver fingerprint.

Optional fallback LLM configuration is supported with
`TABULUS_FALLBACK_LLM_BASE_URL`, `TABULUS_FALLBACK_LLM_MODEL`, and the API-key
environment variable named by `--fallback-llm-api-key-env`. Fallback
configuration is all-or-nothing. If it is absent, Step 6 uses only the primary
provider. If it is present, each adjudication starts with the primary provider
and falls back only after operational failure. Failover does not bypass the
final deterministic admissibility gate.

## Examples

### TabulusBench

TabulusBench examples in this page use controlled Step 6 inputs. The controlled
layout is different from a full production run: it uses derived one-to-one
Step 4 to controlled-gold bibliography alignments so that Step 6 can be tested
as a resolver component without folding upstream bibliography split/merge cases
into the resolver input.

The canonical gold artifacts remain immutable. The derived controlled inputs
live under the literal run tree:

```text
$HOME/tabulusbench/runs/stage6/controlled-p251-p252-final/<paper>/input_one_to_one/
```

Each controlled input directory contains:

- `bibliography.json`
- `reference_matches.json`
- `manifest.json`

`manifest.json` records provenance and excluded upstream alignment cases. The
one-to-one controlled subset must not be described as the full end-to-end
pipeline population.

### 1. Resolve one controlled TabulusBench paper

This example resolves P252 from the controlled one-to-one Step 6 input layout
and writes a fresh first controlled run under `run_01`:

```bash
PAPER_ROOT="$HOME/tabulusbench/runs/stage6/controlled-p251-p252-final/P252/input_one_to_one"
OUT="$HOME/tabulusbench/runs/stage6/controlled-p251-p252-final/P252/run_01"

PYTHONPATH="$HOME/tabulus/src" \
PYTHONUNBUFFERED=1 \
python -m tabulus.cli resolve-references \
  --bibliography "$PAPER_ROOT/bibliography.json" \
  --reference-matches "$PAPER_ROOT/reference_matches.json" \
  --out "$OUT"
```

A complete run writes:

```text
$HOME/tabulusbench/runs/stage6/controlled-p251-p252-final/P252/run_01/
  references/
    reference_resolution.json
```

During an incomplete run, Step 6 writes:

```text
$OUT/references/reference_resolution.checkpoint.json
```

Rerunning the same command with compatible inputs and configuration resumes
from that checkpoint and skips completed bibliography indices.

### 2. Resolve one production paper

For production-style paper processing, pass the Step 4 bibliography and every
Step 5 match artifact that should contribute bibliography indices for that
paper:

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

A completed paper has:

```text
/path/to/artifact-root/references/reference_resolution.json
```

### 3. Use artifact discovery for repeated paper runs

When Step 5 artifacts follow the supported experiment layout, Step 6 can
discover one `reference_matches.json` per adapter for a named paper:

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

If more than one artifact is found for the same adapter, Tabulus refuses to
choose among historical runs implicitly. In that case, pass explicit
`--reference-matches` paths instead.

The CLI does not provide a native multi-paper `resolve-references` batch mode.
Corpus-scale processing consists of independent paper-level invocations, each
with its own bibliography, match artifacts, checkpoint, and final registry.
Completed papers do not need to be rerun when another paper fails for an
operational reason.

### 4. Controlled consistency runs

Step 6 is a hybrid workflow. Target collection, deduplication, bibliographic
scoring, retry bounds, final admissibility, and final status logic are
deterministic. Complete executions can still vary because they depend on live
Crossref results, live CORE results, external provider/network behavior, and
LLM-assisted adjudication.

For controlled Step 6 consistency checks, run two fresh complete executions per
paper from the same controlled input directory:

```text
P251/run_01
P251/run_02
P252/run_01
P252/run_02
```

A second run must not reuse the first run's checkpoint. Compare individual
bibliography-reference outcomes, including final status, exact DOI, validated
identity, retry use, LLM decision where applicable, and the number and type of
references whose final outcome changes. For references validated in both runs,
exact DOI agreement is especially important.

Observed Step 6 runtime should be reported as observed end-to-end wall-clock
runtime because the command includes remote service latency from Crossref,
CORE, and the LLM provider. Where instrumented, record wall-clock runtime,
processed references, seconds/reference, number of LLM-processed entries,
number of bounded retries, and optionally CPU time and maximum resident memory.
Do not describe observed Step 6 wall-clock runtime as pure computation time.

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

## Output And Evaluation Boundary

See {doc}`../data-contracts/reference-resolution-json` for the full artifact
contract. The final artifact stores one entry per resolved bibliography index,
including final status, canonical identity fields where accepted, Crossref/CORE
evidence, LLM response provenance where used, retry state, and rejection
reasons.

Step 6 scholarly-identity accuracy requires an independent scholarly identity
gold standard. Step 5 gold links establish which bibliography entry was cited;
they do not by themselves establish the canonical scholarly-work identity.
Therefore resolver validation status is not equivalent to correctness,
`validated_with_doi` is an output status rather than an accuracy label, and
validation yield must not be reported as precision, recall, DOI accuracy, or
scholarly-identity accuracy without independent identity gold.

Step 7 consumes this paper-level registry together with Step 5 matches and
exports resolved CSV files without rerunning scholarly lookup. See
{doc}`14-csv-export` and {doc}`../data-contracts/resolved-csv`.
