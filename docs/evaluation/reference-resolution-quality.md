# Step 6 Scholarly Identity Resolution Quality

Step 6 evaluation compares paper-level scholarly identities with independent,
source-derived gold. It scores each unique cited bibliography entry once,
after Step 5 has linked table citations to bibliography positions. The
production artifact is {doc}`../data-contracts/reference-resolution-json`.

## Run the evaluator

```bash
tabulus evaluate-reference-resolution \
  --gold /path/to/paper/step6_reference_resolution/gold.json \
  --prediction /path/to/artifact-root/references/reference_resolution.json \
  --out /path/to/evaluation/step6-reference-resolution.json
```

The library API is `tabulus.evaluation.evaluate_reference_resolution()`.
`--out` is optional. The CLI reads the gold and prediction JSON without
changing the Step 6 artifact.

## Inputs and index correspondence

| Input | Contents |
| --- | --- |
| `--gold` | `step6_reference_resolution/gold.json`, with source-derived scholarly identity `records`. |
| `--prediction` | `references/reference_resolution.json`, with an `entries` list; each entry has `resolution.reference_index`. |

The evaluator joins `gold_index` to `resolution.reference_index`. It uses the
historical `step4_index` only when `gold_index` is absent. **This is an index
join, not a bibliography-alignment check.** Predictions must already be in the
gold-index namespace, or have a separately validated one-to-one alignment or
crosswalk. A stored `step4_index` alone does not establish that a production
Step 4 entry represents the same work.

## Scores

The report separates operational resolution yield from metadata agreement:

| Measure | Serialized fields |
| --- | --- |
| Coverage and status | `gold_targets`, `matched_predictions`, `missing_predictions`, `resolved_targets`, `rejected_targets`, `resolution_yield`, `status_counts` |
| DOI agreement | `doi_evaluable`, `doi_predictions`, `doi_exact_matches`, `doi_precision`, `doi_recall`, `doi_f1` |
| Other metadata | `<field>_evaluable`, `<field>_correct`, `<field>_accuracy` for title, authors, year, and venue |

`validated_with_doi` and `validated_without_doi` count as resolved;
`rejected` is a scientific outcome, not an operational failure. Resolution
yield is an output rate, **not an accuracy label**. A missing DOI in source
gold means *unknown*, not that no DOI exists: DOI comparisons use only records
with an asserted gold DOI, and DOI precision counts predictions within that
evaluable set.

Metadata comparisons follow the current evaluator:

| Field | Match rule |
| --- | --- |
| DOI | Exact after DOI URL/prefix and trailing-punctuation normalization. |
| Title | Normalized alphanumeric tokens; exact or `SequenceMatcher` ratio at least `0.95`. |
| Authors | Ordered normalized surnames; list values and strings separated by `and` are supported. |
| Year | Exact trimmed string. |
| Venue | Normalized tokens with common-prefix compatibility and at least 80% compatibility over the shorter token list. |

These scores measure scholarly identity agreement. LLM usage, retries,
provider failures, and remote latency are separate operational diagnostics.

## TabulusBench scope

Source-derived Step 6 gold currently contains **4,075 records** across the
deep subset: P251 (2,280), P252 (996), and P253 (799). P251/P252's 3,276
records form the earlier controlled one-to-one eligible identity subset.
Source annotation is not automatically the eligible denominator for every
production run: first verify bibliography correspondence. Split, merged, and
unmatched Step 4 entries remain in upstream alignment accounting, outside a
clean one-to-one component comparison. The public pair evaluator neither
builds nor validates that crosswalk.

Gold identities are independently sourced, not simply all Step 5 links;
uncovered links are not automatically Step 6 errors. Step 6 evaluation does
not score Step 5 citation-cell matching or the {doc}`step7-quality` physical
export. Step 7 consumes the resolved registry without re-resolving identities.
