# Crossref

## Official Resources

- [Crossref website](https://www.crossref.org/)
- [Crossref REST API documentation](https://www.crossref.org/documentation/retrieve-metadata/rest-api/)
- [Works endpoints and parameters in the Crossref API reference](https://api.crossref.org/swagger-ui/index.html)
- [Single-work DOI lookup: `/works/{doi}`](https://www.crossref.org/documentation/retrieve-metadata/rest-api/#endpoints)
- [Bibliographic search: `/works?query.bibliographic=...`](https://api.crossref.org/swagger-ui/index.html#/Works/get_works)

## Role In Tabulus

Crossref is Tabulus's primary scholarly metadata source in Step 6. It exposes
publisher-deposited work metadata through a JSON REST API. For each unique
bibliography index linked by Step 5, Tabulus first checks an existing DOI when
one was extracted in Step 4. If that lookup does not return a work, Tabulus
searches using the raw citation. The retrieved record is assessed against the
source bibliography evidence; a sufficiently validated DOI or strong, distinct
bibliographic match can be accepted without calling CORE. Otherwise Step 6
continues to the {doc}`core` fallback.

## API Endpoints Used

The client in `src/tabulus/reference_resolution/crossref.py` makes only these
HTTP GET requests to the default base URL `https://api.crossref.org`:

| Purpose | Request path | Parameters |
| --- | --- | --- |
| Existing DOI lookup | `/works/{url-encoded-doi}` | DOI in the path; no query parameters |
| Bibliographic candidate search | `/works?query.bibliographic={raw-citation}&rows=5&mailto={contact-email}` | URL-encoded raw citation, result limit, contact email |

The DOI lookup is attempted only when the Step 4 entry contains a DOI. A
Crossref 404 means that DOI was not validated there and permits bibliographic
search; other HTTP or invalid-response failures are operational errors. The
search limit is five candidates by default. Both requests identify Tabulus
through a `User-Agent` containing the contact email.

## Configuration

`tabulus resolve-references` accepts `--crossref-mailto`. If omitted, the CLI
reads `TABULUS_CROSSREF_MAILTO`. The contact email identifies requests to
Crossref; it is not a Crossref API key. `src/tabulus/cli.py` loads this setting,
`crossref.py` performs the requests, and `pipeline.py` and `assessment.py`
collect and assess the returned candidates.

For the complete command and final status semantics, see
{doc}`../tutorial/13-doi-resolution` and
{doc}`../modules/doi-resolvers`. The resulting paper-level artifact is
{doc}`../data-contracts/reference-resolution-json`.
