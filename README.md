<p align="center">
  <img src="./assets/logo.png" alt="Tabulus logo" width="170"/>
</p>

# Tabulus

<p align="center">
  <a href="https://tabulus.readthedocs.io/en/latest/">
    <img src="https://img.shields.io/readthedocs/tabulus/latest?label=Read%20the%20Docs&logo=readthedocs" alt="Read the Docs"/>
  </a>
  <a href="https://doi.org/10.5281/zenodo.20741284">
    <img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20741284-blue.svg" alt="Tabulus DOI"/>
  </a>
  <img src="https://img.shields.io/badge/python-%3E%3D3.10-blue" alt="Python >= 3.10"/>
</p>

Tabulus is a staged, inspectable pipeline for extracting structured,
reference-aware table data from scientific PDFs. It keeps each major processing
boundary on disk, so table crops, reconstruction outputs, bibliography entries,
reference matches, and paper-level reference-resolution decisions can be
inspected independently.

The rebuilt library currently implements the pipeline through Stage 6:

1. PDF profiling and table detection
2. table reconstruction from canonical crops
3. reference-table classification
4. GROBID-backed bibliography extraction
5. deterministic table-cell-to-bibliography matching
6. paper-level scholarly reference resolution

Stage 7 resolved CSV export, run-report generation, and a single end-to-end
`tabulus run` orchestrator are planned for the rebuilt library.

## Features

- **Staged command-line workflow:** run each pipeline stage independently and
  inspect its artifacts before continuing.
- **Stable filesystem contracts:** exchange `tables_index.json`, prediction
  CSVs, `bibliography.json`, `reference_matches.json`, and
  `reference_resolution.json` between modules.
- **Canonical table-crop handoff:** profile PDFs once, then compare multiple
  table-reconstruction adapters on the same crop inputs.
- **Adapter-based reconstruction:** use OCR, table-structure, and document
  vision-language-model adapters behind one Stage 2 interface.
- **Bibliography-aware processing:** extract bibliography entries from the
  original PDF, link table citation tokens to bibliography positions, and
  resolve scholarly identities once per paper-level bibliography index.
- **Conservative reference resolution:** separate extraction, matching, and
  scholarly identity resolution; keep operational failures distinct from
  scientific rejections.
- **Evaluation support:** evaluate raw reconstruction prediction CSVs against
  gold CSVs with Relative Mapping Similarity (RMS).

## Pipeline At A Glance

```text
PDF
 |
 +--> Stage 1: profile PDF / export canonical table crops
 |        |
 |        v
 |      Stage 2: reconstruct tables with one adapter
 |        |
 |        v
 |      Stage 3: classify reference-containing tables
 |
 +--> Stage 4: extract bibliography from the original PDF
          |
          v
selected_reference_tables.json + bibliography.json
          |
          v
Stage 5: reference_matches.json
          |
          v
Stage 6: references/reference_resolution.json
          |
          v
Stage 7: resolved CSV export (planned)
```

Stage 4 is a parallel PDF-level branch. It does not consume table crops or
reconstruction CSVs. Stage 6 resolves the union of Stage 5-linked bibliography
indices once per paper, rather than resolving each table occurrence separately.

## Installation

Clone the repository and install the current library from the checkout:

```bash
git clone https://github.com/sciknoworg/tabulus.git
cd tabulus
python -m pip install -e ".[dev]"
```

Then check the CLI:

```bash
tabulus --help
```

For machine-specific setup, use the documentation:

- [Windows / CPU setup](https://tabulus.readthedocs.io/en/latest/installation/windows-cpu.html)
- [GPU server setup](https://tabulus.readthedocs.io/en/latest/installation/gpu-server.html)
- [Python library setup](https://tabulus.readthedocs.io/en/latest/installation/python-library.html)

## Quick Start

The commands below show the implemented one-paper workflow. Replace paths and
adapter names with values for your local run.

```bash
PDF="/path/to/paper.pdf"
CROP_ROOT="/path/to/tabulus-output/table-crops/<paper>"
RECONSTRUCTION="$CROP_ROOT/reconstructions/tesseract-tatr"
ARTIFACT_ROOT="/path/to/tabulus-artifacts/<paper>"
```

Profile the PDF and export canonical table crops:

```bash
tabulus profile \
  --pdf "$PDF" \
  --backend pipeline \
  --method auto
```

Reconstruct the detected table crops with a CPU-capable adapter:

```bash
tabulus reconstruct-tables \
  --crops "$CROP_ROOT" \
  --adapter tesseract-tatr \
  --device cpu
```

Classify reconstructed tables for reference-containing content:

```bash
tabulus classify-reference-tables \
  --reconstruction "$RECONSTRUCTION"
```

Extract the bibliography from the original PDF using a running GROBID service:

```bash
tabulus extract-bibliography \
  --pdf "$PDF" \
  --out "$ARTIFACT_ROOT" \
  --grobid-url http://localhost:8070
```

Match selected table references to bibliography positions:

```bash
tabulus match-references \
  --selected "$RECONSTRUCTION/selected_reference_tables.json" \
  --bibliography "$ARTIFACT_ROOT/references/bibliography.json"
```

Resolve linked bibliography entries at paper level:

```bash
tabulus resolve-references \
  --bibliography "$ARTIFACT_ROOT/references/bibliography.json" \
  --reference-matches "$RECONSTRUCTION/references/reference_matches.json" \
  --out "$ARTIFACT_ROOT"
```

Stage 6 also requires Crossref, CORE, and OpenAI-compatible LLM configuration
through command-line options or environment variables. See the Stage 6 tutorial
for the full configuration contract.

## Main Artifacts

| Stage | Command | Main output |
| --- | --- | --- |
| Stage 1 | `tabulus profile` | `tabulus-output/table-crops/<paper>/tables_index.json` and crop images |
| Stage 2 | `tabulus reconstruct-tables` | `native/`, `parsed/`, `predictions/`, `batch_summary.json` |
| Stage 3 | `tabulus classify-reference-tables` | `reference_table_classification.json`, `selected_reference_tables.json` |
| Stage 4 | `tabulus extract-bibliography` | `references/bibliography.json` |
| Stage 5 | `tabulus match-references` | `references/reference_matches.json` |
| Stage 6 | `tabulus resolve-references` | `references/reference_resolution.json` |
| Stage 7 | planned | resolved table export |

The detailed file contracts live in the
[Data Contracts documentation](https://tabulus.readthedocs.io/en/latest/data-contracts/run-directory.html).

## Table Reconstruction Adapters

Stage 2 adapters consume canonical table crops and write the same normalized
artifact layers. The current registry includes lightweight OCR/table-structure
routes, document vision-language models, and model-specific parsers.

See the
[Stage 2 tutorial](https://tabulus.readthedocs.io/en/latest/tutorial/08-table-ocr.html)
for the current adapter list, device support, output layout, and TabulusBench
examples.

## TabulusBench

TabulusBench is the benchmark dataset used in the tutorial examples and
evaluation documentation. It is distributed separately on Zenodo:

- [TabulusBench on Zenodo](https://zenodo.org/records/20230340)

The tutorial uses paper `P4` as the canonical one-paper worked example. A
one-paper example means processing the complete relevant input for that paper,
not cherry-picking one table. Benchmark gold material is read-only and should
not be overwritten by Tabulus runs.

## Evaluation

The rebuilt library currently provides native table-reconstruction evaluation:

```bash
tabulus evaluate-table-reconstruction \
  --gold /path/to/gold.csv \
  --prediction /path/to/prediction.csv \
  --metric rms
```

This evaluates raw table reconstruction with Relative Mapping Similarity (RMS).
It is separate from reference matching quality, bibliography extraction quality,
and scholarly reference-resolution diagnostics.

## Repository Layout

```text
assets/          Logo and documentation imagery
dataset/         Dataset notes and links to external benchmark material
docs/            ReadTheDocs documentation
evaluation/      Retained research/evaluation utilities and outputs
legacy/          Historical code and experiment material
src/tabulus/     Current installable library and CLI
src/legacy_tabulus/ Retained legacy application code
tests/           Current library test suite
```

The supported public library surface is under `src/tabulus`. Retained legacy
and research directories are preserved for provenance, but should not be read
as the current public API unless the docs say so explicitly.

## Development

Install development dependencies and run the test suite:

```bash
python -m pip install -e ".[dev]"
pytest
```

Useful checks before committing documentation or code changes:

```bash
git diff --check
```

## Documentation

- [ReadTheDocs](https://tabulus.readthedocs.io/)
- [Core pipeline overview](https://tabulus.readthedocs.io/en/latest/tutorial/00-overview.html)
- [Data contracts](https://tabulus.readthedocs.io/en/latest/data-contracts/run-directory.html)
- [External tools](https://tabulus.readthedocs.io/en/latest/external-tools/mineru.html)
- [Evaluation overview](https://tabulus.readthedocs.io/en/latest/evaluation/overview.html)

## Citation

Citation information will be added after the associated thesis or software
release is published.

## License

See [LICENSE](LICENSE) for the current repository license. The package metadata
currently reads the license from that file.
