# Current State

This page is an engineering snapshot of the rebuilt installable Tabulus
library. For normal usage, start with {doc}`../tutorial/00-overview` and the
installation page for your machine.

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
`src/tabulus/bibliography/`. The current implementation does not yet provide
DOI resolution, resolved CSV export, run-report/QA bundle generation,
continued-table merging, standalone scientific table normalization, a
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
: Reference-table classification for reconstructed tables. It consumes
  reconstruction manifests and parsed artifacts, writes
  `reference_table_classification.json`, and keeps independent classifications
  separate from continuation-inherited decisions.

`tabulus.bibliography`
: GROBID-backed bibliography extraction for one original scientific PDF. It is
  available through `tabulus extract-bibliography` and the Python API. It posts
  the PDF to GROBID `processReferences`, preserves raw reference text, extracts
  DOI strings only when already present, and writes
  `references/bibliography.json`.

Stage 5 reference matching
: Deterministic linking of selected reference-like table cells to entries in
  `references/bibliography.json`. Matching preserves row-level provenance,
  unmatched tokens, and ambiguous candidates in
  `references/reference_matches.json` without modifying reconstruction
  prediction CSVs.

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

Current reference matching writes:

```text
<artifact-root>/
  references/
    reference_matches.json
```

This artifact is produced from selected reference-like tables and
`references/bibliography.json`.

For the full filesystem contract, see {doc}`../data-contracts/run-directory`.

## Not Yet Rebuilt

The following remain planned or historical in the rebuilt library unless a
future implementation changes this page:

- DOI resolution
- final resolved CSV generation
- run report / QA bundle generation
- full `tabulus run` orchestration
- continued-table merging
- standalone scientific table normalization command
- corpus-scale bibliography validation
- Kreuzberg fallback or Crossref integration in the rebuilt installable
  library

Historical thesis code and older evaluation material may still mention some of
these systems. Those references should not be read as current runnable
features of `src/tabulus`.
