# Step 4 Bibliography Extraction Quality

Step 4 evaluation scores bibliography extraction from the original PDF. In the
production pipeline, Step 4 runs independently of the table-processing branch
and writes {doc}`../data-contracts/bibliography-json` at
`references/bibliography.json`.

Step 4 evaluation is distinct from Step 5 citation matching and Step 6 scholarly
identity resolution. GROBID extraction evidence is not Crossref/CORE/LLM
resolution.

## Public Interface

The library API is `tabulus.evaluation.evaluate_bibliography()`. The CLI is:

```bash
tabulus evaluate-bibliography \
  --gold /path/to/paper/step4_bibliography/gold.json \
  --prediction /path/to/artifact-root/references/bibliography.json \
  --out /path/to/evaluation/step4-bibliography.json
```

`--out` is optional. The evaluator compares one curated bibliography gold file
with one production `references/bibliography.json` artifact.

## Gold And Prediction Inputs

The canonical gold path is:

```text
<paper_root>/step4_bibliography/gold.json
```

The evaluator expects a curated gold list of numbered references with raw
reference strings. The prediction input is the Step 4 bibliography JSON object
with an `entries` list. If `bibliography_count` is present in the prediction, it
must match `len(entries)`.

Evaluation reads both files and writes a separate metrics artifact only when
`--out` is supplied.

## Component-Aware Matching

The current evaluator implements the pair-level portion of TabulusBench's
component-aware v5 Step 4 matcher. It aligns gold and predicted bibliography
entries using raw-block coverage plus bibliographic component evidence.

The implementation considers:

- exact/normalized DOI evidence when present;
- exact-token author surname evidence;
- year evidence;
- title and venue containment using normalized character n-grams;
- volume/page locator evidence;
- complete raw-block coverage;
- one-to-one matches;
- split cases where one gold reference maps to multiple predicted entries;
- merge cases where multiple gold references map to one predicted entry;
- unmatched gold and unmatched prediction positions.

Title and DOI are positive corroborating evidence only. Their absence from a
citation style does not count against otherwise correct bibliography recovery.

## Metrics

The serialized result includes content-recovery, strict-entry, position, and
alignment diagnostics:

- `gold_entries`
- `predicted_entries`
- `content_recovered_gold`
- `content_supported_predictions`
- `content_precision`, `content_recall`, `content_f1`
- `strict_entry_true_positives`, `strict_entry_false_positives`,
  `strict_entry_false_negatives`
- `strict_entry_precision`, `strict_entry_recall`, `strict_entry_f1`
- `one_to_one_gold`
- `split_gold_groups`, `split_pred_entries`
- `merge_groups`, `merged_gold_entries`
- `unmatched_gold`, `unmatched_pred`
- `position_preserved_gold`, `position_preservation_rate`
- mean/median absolute position error for one-to-one matches
- `mean_alignment_score`
- `predicted_component_availability`
- alignment `operations` and unmatched positions

The evaluator also serializes implementation parameters such as the match
threshold, block-F1 threshold, maximum split/merge window, lookahead, and
component weights.

## TabulusBench Scope

TabulusBench provides authoritative Step 4 bibliography gold for 52 of the 252
papers, containing 6,947 numbered gold references. The remaining papers can be
processed by Step 4 in a full-corpus run, but they do not contribute to
bibliography gold-standard metrics unless a Step 4 gold file exists.

Canonical gold path:

```text
<paper_root>/step4_bibliography/gold.json
```

Step 4 quality should not be reported as Step 5 citation-link accuracy or Step 6
identity accuracy. Bibliography split/merge differences can affect downstream
numeric positions even when the citation content itself is largely recovered.
