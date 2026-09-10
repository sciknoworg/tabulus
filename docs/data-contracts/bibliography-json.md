# bibliography.json

`references/bibliography.json` records normalized bibliography entries from the
PDF-level bibliography branch. Step 4 runs GROBID on the original PDF and
writes ordered extraction evidence for Step 5 reference matching and Step 6
paper-level scholarly reference resolution.

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
      "raw": "Smith J. Example paper. Example Journal 12, 100-110 (2020). doi:10.1234/example",
      "doi": "10.1234/example",
      "source": "grobid",
      "title": "Example paper",
      "authors": ["Jane Smith"],
      "year": 2020,
      "venue": "Example Journal",
      "volume": "12",
      "issue": "",
      "pages": "100-110"
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
: One-based bibliography position in parsed GROBID TEI order. Step 5's
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

`entries[].title`
: Structured title parsed from GROBID TEI when available. Missing title
  evidence is preserved as an empty string.

`entries[].authors`
: Structured author names parsed from GROBID TEI when available. Missing
  author evidence is preserved as an empty list.

`entries[].year`
: Structured GROBID publication year when available. If GROBID supplies no
  structured year, Tabulus recovers a year only when exactly one plausible year
  appears in the raw citation. Zero or multiple plausible raw years are
  preserved as `null`.

`entries[].venue`
: Structured journal or series title parsed from GROBID TEI when available.

`entries[].volume`
: Structured volume locator parsed from GROBID TEI when available.

`entries[].issue`
: Structured issue or number locator parsed from GROBID TEI when available.

`entries[].pages`
: Structured page locator parsed from GROBID TEI when available. A narrow
  repair removes a page value when GROBID appears to have placed the only
  recovered publication year in the page field, and accepts a replacement page
  locator only when one simple page or page range occurs immediately before the
  raw publication year.

## Boundary

Step 4 is extraction only. It does not call Crossref, CORE, LLM providers,
embedding services, search engines, or any scholarly-identity resolver.

Downstream steps should treat entry order, raw reference strings, and
structured GROBID fields as immutable extraction evidence. Step 5 links
table-cell references to these bibliography positions without mutating the
bibliography artifact. Step 6 uses the linked bibliography entries as source
evidence while retrieving and validating scholarly candidates in a separate
paper-level artifact.
