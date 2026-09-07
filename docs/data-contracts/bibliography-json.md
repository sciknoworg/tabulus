# bibliography.json

`references/bibliography.json` records normalized bibliography entries from the
PDF-level bibliography branch. Stage 4 runs GROBID on the original PDF and
writes ordered extraction evidence for Stage 5 reference matching.

The bibliography branch reads the original scientific PDF. It does not consume
MinerU table crops, reconstruction prediction CSVs, or reference-table
classification output. Raw bibliography strings remain traceable, and DOI
values are recorded only when they already appear in the extracted bibliography
text.

```json
{
  "bibliography_count": 1,
  "bibliography_source": "grobid",
  "entries": [
    {
      "index": 1,
      "raw": "Smith J. Example paper. 2020. doi:10.1234/example",
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
: One-based bibliography position in parsed GROBID TEI order. Stage 5's
  `numeric_position` method uses this order; it is not a scholarly identifier
  or a position inferred from external metadata.

`entries[].raw`
: Preserved raw reference text, preferring GROBID's raw-reference note when
  available and otherwise using normalized text from the TEI bibliography
  element.

`entries[].doi`
: First DOI found deterministically in the extracted bibliography text, or an
  empty string. This is local text extraction, not Crossref enrichment.

`entries[].source`
: Source extractor for the entry. The implemented GROBID path writes `grobid`.

## Boundary

Stage 4 is extraction only. It does not call Crossref, CORE, LLM providers,
embedding services, search engines, or any scholarly-identity resolver.

Downstream stages should treat entry order and raw reference strings as
extraction evidence. Stage 5 links table-cell references to these bibliography
positions without mutating the bibliography artifact. Paper-level scholarly
identity resolution is planned for a later stage and is not implemented in the
current `src/tabulus` package.
