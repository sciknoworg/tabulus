# bibliography.json

`references/bibliography.json` records normalized bibliography entries from the
PDF-level bibliography branch.

The bibliography branch reads the original scientific PDF. It does not consume
MinerU table crops, reconstruction prediction CSVs, or reference-table
classification output. Raw bibliography strings should remain traceable, and
DOI values may be recorded when they are already present in the extracted
bibliography text. Crossref or other external DOI resolution belongs to a later
stage.

The current GROBID implementation writes:

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

`entries[].raw`
: Preserved raw reference text, preferring GROBID's raw-reference note when
  available.

`entries[].doi`
: DOI found deterministically in the bibliography text, or an empty string.
  This is not Crossref enrichment.

`entries[].source`
: Source extractor for the entry.
