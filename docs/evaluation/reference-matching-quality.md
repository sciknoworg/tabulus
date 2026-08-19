# Reference Matching Quality

Reference matching quality evaluates whether table references are linked to the correct bibliography entries and DOI values.

This should be measured after bibliography extraction and table reconstruction contracts are stable.

The scored artifact is `references/reference_matches.json`, optionally compared with curated row-level match labels. Metrics should distinguish:

- extracting the reference-like cell from the table
- matching that cell to the correct bibliography entry
- carrying through DOI values when available
- leaving unresolved rows traceable

Precision, recall, F1, coverage, and unresolved-row counts are useful depending on the annotation available.

Reference matching evaluation should not mutate prediction CSV files. Resolved CSV files are downstream outputs produced after matching and DOI enrichment.
