# bibliography.json

`references/bibliography.json` records normalized bibliography entries from the
PDF-level bibliography branch. Stage 4 runs GROBID on the original PDF and
produces ordered, immutable extraction evidence for Stages 5 and 6.

The bibliography branch reads the original scientific PDF. It does not consume
MinerU table crops, reconstruction prediction CSVs, or reference-table
classification output. Raw bibliography strings should remain traceable, and
DOI values may be recorded when they are already present in the extracted
bibliography text. Crossref or other external DOI resolution belongs to a later
stage.

The core positional fields are illustrated below; this minimal example omits
the enriched metadata described afterward:

```json
{
  "bibliography_count": 1,
  "bibliography_source": "grobid",
  "entries": [
    {
      "index": 1,
      "raw": "Smith J. Example paper. 2020.",
      "doi": "10.1234/example",
      "source": "grobid"
    }
  ]
}
```

`bibliography_count`
: Number of bibliography entries in the artifact.

`bibliography_source`
: Source extractor for the artifact. The implemented GROBID path writes
  `grobid`.

`entries[].index`
: One-based bibliography position in the parsed GROBID TEI order.
  Stage 5's `numeric_position` method uses this order; it is not a scholarly
  identifier or a position inferred from external metadata.

`entries[].raw`
: Preserved raw reference text, preferring GROBID's raw-reference note when
  available.

`entries[].doi`
: DOI found deterministically in the bibliography text, or an empty string.
  This is not Crossref enrichment.

`entries[].source`
: Source extractor for the entry.

## Enriched Extraction Metadata

The enriched Stage 4 artifact also carries title, authors, year, venue, volume,
issue, and pages when extracted. Missing values remain missing. These fields
are citation evidence, not metadata validated by Crossref or CORE.

A structured GROBID year takes precedence. Without one, exactly one plausible
year in the raw citation may be recovered; zero or multiple plausible years
remain unresolved. Extraction also repairs GROBID's year-in-pages cases.
Compound/multi-work references retain their ambiguity.

Downstream consumers must preserve entry counts, index ordering, and raw
citation strings. The enriched rerun for all 10 JVSTA demonstration papers
preserved these exactly, verifying compatibility with existing Stage 5
positional links without rerunning Stages 3 or 5.
