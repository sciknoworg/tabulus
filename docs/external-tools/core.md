# CORE

## Official Resources

- [CORE website](https://core.ac.uk/)
- [CORE API service and documentation](https://core.ac.uk/services/api)
- [CORE API v3 reference](https://api.core.ac.uk/docs/v3)
- [Works-search endpoint: `/v3/search/works/`](https://api.core.ac.uk/v3/search/works/)

## Role In Tabulus

CORE aggregates open-access scholarly works from repositories and journals and
offers searchable metadata through its API. Tabulus uses CORE as a conditional
Step 6 discovery source when Crossref has not provided sufficient evidence for
a cited bibliography entry. CORE work records are normalized into the same
candidate model and assessed by deterministic bibliographic scoring; a strong,
distinct match can be accepted. Otherwise the retrieved Crossref and CORE
evidence is available to bounded LLM-assisted adjudication. An LLM selection
must still pass Tabulus's final deterministic admissibility gate.

## API Endpoint Used

`src/tabulus/reference_resolution/core.py` makes an HTTP GET request to the
default CORE API v3 base URL:

```text
https://api.core.ac.uk/v3/search/works/?q={sanitized-citation}&limit=5
```

The `q` value comes from the raw bibliography citation after CORE-query
metacharacters are sanitized; `limit` defaults to five. The request uses
`Authorization: Bearer` with the configured API key. Tabulus uses this works
search endpoint for candidate retrieval; it does not call CORE's full-text
download or other API operations.

Retryable HTTP responses (429 and 500/502/503/504) and timeouts receive bounded
retries, with `Retry-After` respected where available. Exhausted retries,
unreachable services, and invalid responses remain operational failures; they
are not turned into scientific `rejected` decisions. The scholarly-search retry
in Step 6 is separate: LLM-assisted adjudication may request at most one new
search, which again checks Crossref before conditional CORE fallback.

## Configuration

The default `--core-api-key-env` value is `CORE_API_KEY`. This CLI option names
the environment variable containing the CORE API key; it does not take the key
itself. Set the variable in the execution environment and keep its value out of
commands, logs, and committed files. `src/tabulus/cli.py` reads the variable,
`core.py` performs the request, and `fallback.py` decides when CORE is needed
and assesses its candidates.

For the full Step 6 workflow and final statuses (`validated_with_doi`,
`validated_without_doi`, and `rejected`), see
{doc}`../tutorial/13-doi-resolution` and {doc}`../modules/doi-resolvers`.
Step 4 extraction through {doc}`grobid` does not call CORE.

## Reference

Knoth, Petr, Drahomira Herrmannova, Matteo Cancellieri, Lucas Anastasiou,
Nancy Pontika, Samuel Pearce, Bikash Gyawali, and David Pride. "CORE: A global
aggregation service for open access papers." *Scientific Data* 10, no. 1
(2023): 366. [https://doi.org/10.1038/s41597-023-02208-w](https://doi.org/10.1038/s41597-023-02208-w).
