# Step 6 Scholarly Identity Resolution Services

Step 6 retrieves scholarly-work metadata for bibliography entries linked by
Step 5. Crossref is queried first; CORE is a conditional fallback when Crossref
evidence does not establish a sufficiently supported identity. Step 4 GROBID
bibliography extraction remains a separate PDF-level operation.

```{toctree}
:maxdepth: 1

crossref
core
```
