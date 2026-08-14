# Step 5: Reference Section Detection

## Goal

Find the page where the bibliography or references section begins.

## Input

`layout/layout_items.json` and optional extracted page text.

## Output

`metadata/reference_section.json`.

## Module Contract

```json
{
  "refs_start_page": 12,
  "evidence": "References",
  "source": "layout_heading",
  "confidence": "medium",
  "status": "reference_section_detected"
}
```

## Default Implementation

The current MinerU runner scans layout text for headings such as `References`, `Bibliography`, `Literaturverzeichnis`, `Quellen`, and `Referenzen`.

## Verification

The step succeeds when it either reports a page number with evidence or records that no reference heading was found.
