# GROBID

## Official Resources

- [GROBID project repository](https://github.com/grobidOrg/grobid)
- [GROBID documentation](https://grobid.readthedocs.io/)

## Role In Tabulus

GROBID is the external scholarly-document parser used by Tabulus for Stage 4
bibliography extraction. It is machine-learning software for extracting,
parsing, and restructuring scholarly PDFs into TEI/XML. GROBID's own system
can support many scholarly-document tasks, including header parsing, reference
parsing, citation contexts, full-text structuring, and optional consolidation;
Tabulus uses only the bibliography-reference extraction boundary for the
implemented Stage 4 path.

Tabulus sends the original publication PDF to a running GROBID HTTP service and
normalizes the returned TEI bibliography into `references/bibliography.json`.
This branch is independent of MinerU table crops and reconstruction outputs.
Stage 4 does not resolve scholarly identities; Crossref, CORE, and LLM-backed
adjudication belong to Stage 6.

GROBID is used by the current CLI command:

```bash
tabulus extract-bibliography \
  --pdf /path/to/paper.pdf \
  --out /path/to/artifact-root \
  --grobid-url http://localhost:8070
```

The command also supports an optional HTTP timeout:

```bash
tabulus extract-bibliography \
  --pdf /path/to/paper.pdf \
  --out /path/to/artifact-root \
  --grobid-url http://localhost:8070 \
  --timeout-seconds 300
```

`--pdf` is the original scientific PDF. `--out` is the Tabulus artifact root;
the bibliography writer creates `references/bibliography.json` beneath it.
`--grobid-url` is the service root. The client appends
`/api/processReferences` internally.

Tabulus posts the PDF as multipart form data with `includeRawCitations=1` and
`consolidateCitations=0`. Raw citation strings are requested because downstream
Stage 5 matching treats them as extraction evidence. GROBID citation
consolidation is disabled because external scholarly metadata lookup and
identity resolution are separate from Stage 4 extraction.

The normalized bibliography entry preserves:

- `index`: one-based position in parsed GROBID TEI bibliography order
- `raw`: preserved raw reference text, preferring GROBID's raw-reference note
  when available
- `doi`: the first DOI found deterministically in the extracted bibliography
  text, or an empty string
- `source`: `grobid`
- `title`, `authors`, `year`, `venue`, `volume`, `issue`, and `pages` when
  GROBID supplies usable structured metadata or the parser can recover a
  single unambiguous missing year from the raw citation

Missing metadata remains missing. Stage 4 does not invent titles, authors,
years, venues, locators, DOIs, or scholarly identities. Structured GROBID years
win over raw-text recovery; if the raw citation has zero or multiple plausible
years, the year remains unresolved. A narrow parser repair handles the observed
case where GROBID places the only recovered year in the page field.

Tabulus does not implement a separate GROBID health-check command. Existing
workflow docs show checking a service with `curl` before extraction. During
extraction, HTTP errors, unreachable services, invalid PDF paths, empty
responses, and invalid TEI are surfaced as extraction failures rather than
converted into bibliography entries.

See also:

- {doc}`../tutorial/11-bibliography-extraction`
- {doc}`../data-contracts/bibliography-json`
- {doc}`../modules/bibliography-extractors`
- {doc}`../tutorial/12-reference-matching`
- {doc}`../tutorial/13-doi-resolution`
