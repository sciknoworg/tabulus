# Reference Matching Quality

Reference matching quality concerns Step 5 table-cell-to-bibliography-position
links. Step 5 is deterministic and offline: it links citations found in
selected reference-table cells to ordered Step 4 bibliography entries. DOI
values present in `references/bibliography.json` are extraction evidence, not
independently resolved scholarly identities.

The production Step 5 command is:

```bash
tabulus match-references \
  --selected /path/to/selected_reference_tables.json \
  --bibliography /path/to/artifact-root/references/bibliography.json
```

The output is `references/reference_matches.json` inside the reconstruction
adapter directory unless `--out` is provided.

## Production Artifact Boundary

The production artifact records table-cell-level links, unmatched tokens, and
skipped-table diagnostics. Coverage counts in that artifact are operational
traceability statistics. They are not accuracy, precision, recall, or F1 unless
compared with suitable gold labels.

Step 5 links table citation occurrences to bibliography positions. Step 6 is
the separate paper-level scholarly identity resolution step. Production numeric
matching uses positions in the normalized Step 4 GROBID bibliography order.

## Controlled Gold-Standard Scope

The current controlled Step 5 evaluation uses two TabulusBench papers:

- P251: Miikkulainen et al. (2013), 44 physical table fragments
- P252: Puurunen (2005), 21 physical table fragments

Only tables with an explicit reference column contribute citation-link gold:

- P251: 44/44 tables
- P252: 13/21 tables
- total evaluated: 57 physical tables

The eight P252 fragments without an explicit reference column are retained as
non-contributing fragments for Step 5.

The canonical Step 5 gold files are:

```text
P251/reference_matching/gold.json
P252/reference_matching/gold.json
```

in the corresponding TabulusBench paper directories.

The full expanded TabulusBench contains 252 papers and 605 gold physical
reference-table fragments. Step 5 production matching can be run wherever Step
3 and Step 4 artifacts exist, but the current Step 5 accuracy gold covers only
P251 and P252. For other papers, coverage, unmatched-token, and consistency
statistics must not be called precision, recall, F1, or accuracy unless
suitable gold labels exist.

## Gold Construction Semantics

The Step 5 gold is derived controlled gold, not separately hand-annotated
citation-link gold. Links are deterministically derived from:

- expert-curated `reference_tables/tables/*/gold.csv`
- complete numbered `bibliography/gold.json`

Only explicit numeric bibliography-reference expressions occurring in explicit
reference columns are included. Numeric markers embedded in non-reference
content, such as table-local footnote markers `mark[1]` or `mark[2]`, are
excluded.

Gold size:

| Paper | Citation cells | Bibliography links |
| --- | ---: | ---: |
| P251 | 1,478 | 4,384 |
| P252 | 538 | 1,579 |
| Overall | 2,016 | 5,963 |

## Evaluation Setting

The controlled evaluation gives the production Step 5 matcher:

- expert-curated gold table representations
- complete gold numbered bibliographies

This intentionally isolates Step 5 from upstream Step 2 reconstruction errors
and Step 4 GROBID bibliography-segmentation errors. Step 4 can recover
bibliography content nearly perfectly while split/merge differences alter
numeric positions; therefore controlled Step 5 performance must not be
presented as end-to-end pipeline performance.

## Metrics

The controlled evaluation reports:

`reference-column detection`
: Whether the explicit reference column is detected for each contributing
  table.

`exact citation-cell accuracy`
: The predicted bibliography-index set for a citation cell must exactly equal
  the gold set.

`individual-link precision, recall, and F1`
: Link metrics over `(citation cell, bibliography index)` pairs.

## Current Controlled Result

| Paper | Tables | Ref. columns | Citation cells | Links | Exact-cell accuracy | Link P | Link R | Link F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P251 | 44 | 44/44 | 1,478 | 4,384 | 100.00% | 100.00% | 100.00% | 100.00% |
| P252 | 13 | 13/13 | 538 | 1,579 | 100.00% | 100.00% | 100.00% | 100.00% |
| Overall | 57 | 57/57 | 2,016 | 5,963 | 100.00% | 100.00% | 100.00% | 100.00% |

Aggregate link counts:

- TP = 5,963
- FP = 0
- FN = 0
- unmatched tokens = 0
- mismatched cells = 0

These numbers characterize the deterministic Step 5 matcher under controlled
gold-table and gold-bibliography inputs. They are not end-to-end pipeline
accuracy and do not include upstream reconstruction or bibliography-segmentation
errors.

## Gold JSON Schema

The TabulusBench Step 5 gold files use this structure:

`schema_version`
: Gold-file schema version.

`task`
: The benchmark task, corresponding to Step 5 reference matching.

`paper_id`
: Paper identifier, such as `P251` or `P252`.

`gold_type`
: Gold label type for controlled reference matching.

`scope`
: Scope of the gold file, including the contributing physical table fragments.

`semantics`
: Explanation that bibliography indices are positions in the complete gold
  numbered bibliography.

`construction`
: Provenance for deterministic gold construction from curated table CSVs and
  numbered bibliography gold.

`summary`
: Counts such as contributing tables, citation cells, bibliography links, and
  retained non-contributing fragments.

`tables`
: Per-table citation-link labels.

Each `tables[]` entry contains:

`table_id`
: Physical table identifier.

`table_key`
: TabulusBench table directory key.

`gold_csv`
: Path to the expert-curated table CSV used for gold construction.

`reference_column_index`
: Zero-based explicit reference-column index.

`reference_column_header`
: Header text for the reference column.

`citation_cells`
: Per-cell citation-link labels.

Each `citation_cells[]` entry contains:

`row_index`
: Row index in the gold table representation.

`column_index`
: Column index of the reference cell.

`value`
: Raw cell value used for gold construction.

`bibliography_indices`
: Gold bibliography positions linked by that cell.

`tables_without_explicit_reference_column`
: Retained physical fragments that do not contribute citation-link gold because
  they lack an explicit reference column.

This gold schema is separate from the production
{doc}`../data-contracts/reference-matches-json` artifact. Production Step 5
uses the normalized Step 4 bibliography order; controlled evaluation uses gold
bibliography numbering to isolate the matcher.

## Step 6 Diagnostic Boundary

Step 6 scholarly reference resolution uses a paper-level denominator: the union
of bibliography indices linked by applicable Step 5 artifacts, deduplicated by
bibliography index. Each unique bibliography entry is resolved once per paper,
not once per table cell, table fragment, or reconstruction adapter.

Resolution status counts and resolution coverage are diagnostics over that
paper-level denominator. They should not be presented as accuracy without a
human gold standard. Operational failures during resolution are distinct from
scientific rejections and should not be counted as rejected scholarly
references.
