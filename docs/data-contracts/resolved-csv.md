# Resolved CSV

Resolved CSV files are final user-facing table exports after reference matching and DOI resolution.

A prediction CSV is the reconstructed table before enrichment and is the artifact compared with ground truth during table-reconstruction evaluation. A resolved CSV is created later only for a relevant/reference-containing table after bibliography extraction, reference matching, and DOI resolution.

The detected reference column should be renamed to `DOI`. When DOI values are found, the original reference cell is replaced with DOI values. When no DOI is found, the original value should remain traceable in the output. Non-reference tables may have prediction CSV artifacts without a resolved CSV.

Example:

```text
DOI,Value
10.1234/example,A
```
