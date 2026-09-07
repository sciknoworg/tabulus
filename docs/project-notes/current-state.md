# Current State

This page is an engineering snapshot of the rebuilt installable Tabulus
library. For normal usage, start with {doc}`../tutorial/00-overview` and the
installation page for your machine.

This snapshot records the implemented pipeline through Stage 6 on the GPU
cluster, as reported for this documentation update. The local documentation
checkout still contains the earlier Stage 4 model and commands through Stage 5;
the enriched extractor and Stage 6 source are on the cluster. CLI examples below
are limited to commands verified in this checkout.

## Runnable Stages

The rebuilt library currently exposes standalone command-line stages:

```bash
tabulus profile --pdf /path/to/paper.pdf --backend pipeline

tabulus reconstruct-tables \
  --crops /path/to/tabulus-output/table-crops/<paper> \
  --adapter paddleocr-vl \
  --device gpu:0

tabulus classify-reference-tables \
  --reconstruction /path/to/tabulus-output/table-crops/<paper>/reconstructions/paddleocr-vl

tabulus extract-bibliography \
  --pdf /path/to/paper.pdf \
  --out /path/to/artifact-root \
  --grobid-url http://localhost:8070

tabulus match-references \
  --selected /path/to/selected_reference_tables.json \
  --bibliography /path/to/artifact-root/references/bibliography.json
```

Bibliography extraction is implemented as a Python library API under
`src/tabulus/bibliography/`. The pipeline does not yet provide
resolved CSV export, run-report/QA bundle generation,
continued-table merging, standalone scientific table normalization, or
complete `tabulus run` orchestration.

## Implemented

`tabulus.mineru`
: MinerU-backed PDF profiling, existing-output discovery, recursive
  `*_content_list.json` lookup, table-region extraction, page/provenance
  normalization, and automatic canonical table-crop export.

`tabulus export-table-crops`
: Regenerates the canonical crop handoff from an existing MinerU output
  directory without rerunning MinerU.

`tabulus.table_ocr`
: Stage 2 table reconstruction over canonical MinerU crops. The package
  provides the adapter protocol, lazy registry, batch runner, shared
  HTML/Markdown parser, deterministic OTSL-to-HTML normalization, native and
  parsed artifact writing, prediction CSV export, and batch summary manifests.

`tabulus reconstruct-tables`
: Runs one registered reconstruction adapter over one or more canonical crop
  roots. It preserves table IDs, processes physical crops independently,
  reuses one adapter instance across the command, and writes `native/`,
  `parsed/`, `predictions/`, and `batch_summary.json`.

`tabulus.reference_tables`
: One deterministic regex/rule classifier applied independently to each
  reconstruction method's outputs. It identifies reference-containing
  reconstructed-table instances. It consumes reconstruction manifests and
  parsed artifacts, writes
  `reference_table_classification.json`, and keeps independent classifications
  separate from continuation-inherited decisions.

`tabulus.bibliography`
: GROBID-backed bibliography extraction for one original scientific PDF. It is
  available through `tabulus extract-bibliography` and the Python API. It posts
  the PDF to GROBID `processReferences`, preserves raw reference text, extracts
  DOI strings only when already present, and writes
  `references/bibliography.json`.

  The enriched extraction also preserves title, authors, year, venue, volume,
  issue, and pages when available. It performs no scholarly resolution; the
  ordered bibliography remains immutable evidence for downstream stages.

Stage 5 reference matching
: Deterministic linking of selected reference-like table cells to entries in
  `references/bibliography.json`. Matching preserves row-level provenance,
  unmatched tokens, and ambiguous candidates in
  `references/reference_matches.json` without modifying reconstruction
  prediction CSVs.

Stage 6 scholarly reference resolution
: Resolves each unique `(paper, bibliography_index)` once, using the union of
  referenced indices from every reconstruction method. Crossref DOI validation
  and bibliographic search, CORE fallback, and bounded LLM adjudication feed a
  deterministic evidence gate. The result is one paper-level
  `references/reference_resolution.json` registry. See
  {doc}`../tutorial/13-doi-resolution` for safeguards and resumability.

## Stage 2 Adapter Set

The current registered crop-consuming adapters are maintained in
{doc}`../tutorial/08-table-ocr`. Adapter-specific model revisions, prompts,
runtime versions, and upstream resources are documented in the External Tools
section.

MinerU `table_body` remains a native reconstruction candidate produced during
profiling rather than a crop-consuming adapter.

The core reconstruction policy is unchanged across adapters:

- every adapter receives the same canonical MinerU crop
- no adapter returns to the source PDF to choose a different table region
- no adapter output is semantically repaired to improve apparent quality
- no continued-table merging happens during reconstruction
- prediction CSVs are written only for `ok` results with exactly one parsed
  structured table

## Validation Boundary

The automated test suite is designed so most tests do not require heavyweight
GPU model execution. It covers:

- MinerU output discovery and profiling command behavior
- canonical table-crop export
- adapter registry metadata and lazy loading
- mocked behavior for all registered reconstruction adapters
- shared HTML/Markdown parsing and OTSL normalization
- batch reconstruction input handling and artifact writing
- reference-table classification heuristics and manifest writing
- GROBID TEI bibliography parsing, HTTP request construction, and bibliography
  artifact writing
- deterministic Stage 5 matching and paper-level Stage 6 resolution safeguards
- Stage 6 operational failure handling and checkpoint resumability

The reported full repository suite for the GPU-cluster implementation passes
with **477 passed**, and its reported `git diff --check` is clean. These are
the supplied implementation-validation results, not a local test rerun during
this documentation-only update.

Real-model GPU validations are operational engineering checks. They confirm
that adapters can load, run through the Tabulus CLI, and produce the expected
artifact layers in validated environments. They are not reconstruction
accuracy, recall, precision, F1, or model-ranking evidence.

## Output Boundary

Current reconstruction outputs are pre-reference-resolution artifacts:

```text
<crop-root>/
  reconstructions/
    <adapter>/
      native/
      parsed/
      predictions/
      batch_summary.json
      reference_table_classification.json
```

`native/` preserves adapter-native evidence. `parsed/` preserves the common
structured representation. `predictions/` contains raw reconstruction CSVs.
`reference_table_classification.json` is a downstream routing/classification
manifest and does not overwrite reconstruction predictions.

Current bibliography extraction writes:

```text
<artifact-root>/
  references/
    bibliography.json
```

This artifact is produced from the original PDF, not from MinerU crops or
prediction CSV files.

Current reference matching writes by default:

```text
<reconstruction-directory>/
  references/
    reference_matches.json
```

This artifact is produced from selected reference-like tables and
`references/bibliography.json`.

For the full filesystem contract, see {doc}`../data-contracts/run-directory`.

## Reference Resolution Architecture

Stage 3 table selection and Stage 4 bibliography extraction are parallel
branches from the paper. Stage 5 matches table cells to bibliography positions.
Stage 6 collects the union of matched indices across reconstruction methods,
deduplicates by bibliography index, and writes one paper-level registry:

```text
<paper-artifact-root>/references/
  bibliography.json
  reference_resolution.checkpoint.json  (while incomplete)
  reference_resolution.json             (after all targets complete)
```

The final statuses are `validated_with_doi`, `validated_without_doi`, and
`rejected`. Rejection means insufficient evidence for a safe scholarly identity;
operational failures abort/checkpoint the run instead. Stage 7 remains
unimplemented: it will deterministically join validated identities back to all
relevant table cells/reference occurrences and produce downstream exports.

## JVSTA Demonstration Status

Enriched Stage 4 extraction was rerun for all 10 demonstration papers. Compared
with the previous artifacts, bibliography entry counts, index ordering, and raw
citation strings stayed identical. Stage 5 positional compatibility was verified
without rerunning Stages 3 or 5. See
{doc}`../tutorial/11-bibliography-extraction` for extraction and year-recovery
rules, and {doc}`../evaluation/reference-matching-quality` for the frozen
Stage 5 coverage counts.

Miikkulainen 2013 has 2,390 extracted bibliography entries, of which only 42
have a GROBID-extracted title (about 1.8%); authors/year/venue/pages are much
more complete. This is an observation about the extracted citation metadata
and citation structure in this paper, not a universal claim about citation styles.

Earlier Stage 6 corpus runs are provisional/diagnostic, not final evaluation
results. A clean final rerun is underway with the enriched Stage 4 artifacts.
Puurunen - February 2005 is the first canary because it contains many difficult
reference structures used to harden Stage 6. Its inputs are:

- `stage4-bibliography-enriched/ald/Puurunen - February 2005/references/bibliography.json`
- the 16 existing Stage 5 reference-match artifacts

The canary has **1,072 unique target bibliography references**. Checkpoint
resume behavior has already been verified successfully in the live run. Final
Stage 6 resolution coverage and result percentages must wait until the clean
corpus rerun completes; validated identities are not human-gold-standard
correctness labels.

## Not Yet Rebuilt

The following remain planned or historical in the rebuilt library unless a
future implementation changes this page:

- final resolved CSV generation
- run report / QA bundle generation
- full `tabulus run` orchestration
- continued-table merging
- standalone scientific table normalization command
- corpus-scale bibliography validation
- Kreuzberg fallback

Historical thesis code and older evaluation material may still mention some of
these systems. Those references should not be read as current runnable
features of `src/tabulus`.
