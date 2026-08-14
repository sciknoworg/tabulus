# Resolved CSV

Resolved CSV files are final table exports.

The detected reference column should be renamed to `DOI`. When DOI values are found, the original reference cell is replaced with DOI values. When no DOI is found, the original value should remain unless a later policy decides otherwise.

Example:

```text
DOI,Value
10.1234/example,A
```
