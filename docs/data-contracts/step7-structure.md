
# Step 7 structural plans and ORKG projection

`export-resolved-csv --orkg` adds an ORKG projection to the physical resolved
CSVs. Step 7 runs per physical fragment. Scientific coordinates refer to the
Step 2 prediction CSV, including headers. Step 5 supplies explicit links.
Step 6 supplies scholarly identities. The structural planner receives the
source grid, citation occurrence counts and available structural evidence.

## Run the projection

```bash
tabulus export-resolved-csv \
  --reference-matches /path/to/reference_matches.json \
  --reference-resolution /path/to/reference_resolution.json \
  --orkg --out /path/to/run
```

Use `--reconstruction-root` when the Step 5 manifest points at a different
machine's reconstruction directory. Relative prediction paths are resolved
inside that local root. When available, `--tables-index` supplies Step 1
MinerU structure evidence. Adapter-native HTML or OTSL can also provide
span evidence. Only spans aligned to the prediction grid supply coordinates.

A rectangular grid with a recognized first-row header, distinct scientific
headings, a known reference column, complete cells and no detected structural
ambiguity uses the deterministic path. Mechanical row replication per
reference association requires no model. Complex or ambiguous tables use
structural planning. If the plan is unavailable or fails validation, the
ORKG manifest records an abstention and the physical export remains usable.

### Endpoint configuration

Values are resolved when a table needs the planner.

| Setting | Command-line override | Step 7 environment | Shared environment |
| --- | --- | --- | --- |
| Base URL | `--step7-llm-base-url` | `TABULUS_STEP7_LLM_BASE_URL` | `TABULUS_LLM_BASE_URL` |
| Model | `--step7-llm-model` | `TABULUS_STEP7_LLM_MODEL` | `TABULUS_LLM_MODEL` |
| Provider label | `--step7-llm-provider` | `TABULUS_STEP7_LLM_PROVIDER` | `TABULUS_LLM_PROVIDER` |
| API key | `--step7-llm-api-key-env NAME` | `TABULUS_STEP7_LLM_API_KEY` | `TABULUS_LLM_API_KEY` |

Use a dedicated key when using a different endpoint URL. The planner appends
`/chat/completions` to a base URL such as `https://chat-ai.academiccloud.de/v1`
for SAIA or `https://openrouter.ai/api/v1` for OpenRouter. A custom compatible
URL works with a model that returns a textual JSON response.
Step 7 waits up to 60 seconds per attempt by default. Set
`TABULUS_STEP7_LLM_TIMEOUT_SECONDS` to allow more time for a response.
For endpoints and models that support `reasoning.max_tokens`, set
`TABULUS_STEP7_LLM_REASONING_MAX_TOKENS` to reserve part of the 8,192-token
response budget for the JSON plan. The manifest records the configured
budget, finish reason and token usage when the endpoint returns them.

## Plan format

The planner returns an object with `schema_version: 1`, `header_rows` and
`columns`. Coordinates are zero-based. Intervals use `[start, end)`.

| Field | Operation |
| --- | --- |
| `header_rows` | Source header row indices, including the Step 5 header if available. |
| `columns` | Ordered objects with `label`, `source_columns`, `header_cells` and optional `datatype`. Header cells are source coordinate pairs from parent to child. An empty source list allows a context-derived column. Multiple source values compose in order with ` \| `. |
| `group_rows` | Non-data heading rows that can anchor context. Cited rows and unmatched citation evidence cannot be removed. |
| `groups` | Nested or disjoint scopes beginning at group headings. |
| `merge_rows` | Disjoint contiguous ranges of data rows that form one record. |
| `propagations` | `source: [row, column]`, output `column`, target `rows` and enclosing `group`. Optional `slice` selects exact source characters. `mode: "fill"` applies to blank values and `mode: "prefix"` adds context to populated values. |
| `blanks` | Output `column`, `rows` and `kind` of `intentional` or `unknown`. Inherited values use propagation. |
| `inherit_references` | Explicit anchor `source_row`, target `rows`, `group` and optional `occurrences` from the anchor's links. Use an aligned reference span or provide scientific group `evidence_cells`. Explicit and intervening citations block inheritance. |

The model can name logical columns and identify evidence. The executor
derives scientific values from source cells and retrieves identities from
the Step 6 registry. Each rule is local to a physical fragment. Group
context from earlier fragments falls outside the plan schema.

### Output and provenance

`orkg/manifest.json` records the route and reasons, source hash, plan,
aligned spans, logical records, cell lineage, explicit and inherited links,
omissions, runtime and model usage. Each eligible association yields one
CSV row with the scientific values and a validated DOI or complete validated
title, authors and year fallback. Unmatched citations and non-importable
identities remain visible in the manifest. The CSV uses typed scientific
headers and ORKG comparison metadata fields. Import acceptance is a separate
check.

## Evaluation

```bash
tabulus evaluate-step7 \
  --prediction /path/to/run/orkg/manifest.json \
  --benchmark-paper /path/to/tabulusbench/paper \
  --alignment /path/to/evaluation_crosswalk.json \
  --out /path/to/evaluation.json
```

`--benchmark-paper` reads Step 5 gold cells, Step 4 numbered bibliography
entries and independently validated Step 6 `gold_index` identities. It also
reads draft ORKG headers and available row-level provenance. P251–P253 are
development data. Their draft fields provide diagnostics. Independent
annotations are required for routing, header hierarchy and propagation
scores. Use distinct papers for held-out generalization.

For an independent annotation set, use `--gold` with schema version 1,
`split` of `development` or `held_out`, an optional `index_namespace` and
a `tables` list. Each table uses `table_key` and can annotate the following:

- `route` of `deterministic` or `llm_assisted`.
- Ordered `columns` with `label` and optional `source_columns`,
  `header_cells` and `datatype`.
- Ordered `records` with `source_rows`. Optional `expansion` is the expected
  number of logical associations. Optional `projection_rows` is the expected
  row count after identity filtering. A record may annotate `cells` in
  column order with `value`, `sources`, `propagations` and `blanks` as in the
  prediction manifest.
- `links` on the table or its records. Each link identifies `source_row`,
  `source_column`, `target_row`, `reference_index`, zero-based `occurrence`,
  and `kind` of `explicit` or `inherited`. Optional `identity` fields give
  independently validated canonical metadata. Set `inheritance_annotated`
  to false where inherited links lack labels.

An evaluation crosswalk can specify
`tables: {prediction_key: {table_key, rows, columns}}` and a paper-local
`bibliography_indices` map. Coordinate and bibliography map keys are decimal
strings for prediction positions and values are gold positions. Unmapped
positions are unmatched. A bibliography map is required when the prediction
namespace differs from the declared gold namespace. Independently prepared
controlled inputs can declare `index_namespace: "gold_index"` after verifying
their alignment. With that namespace, `--allow-uncovered-identities` retains
links beyond independent identity coverage as `identity_unavailable` in the
physical export. Production export does not read benchmark data.

The evaluator separates structural correctness, citation alignment, identity
accuracy and export coverage. It checks each CSV against its provenance
manifest. Identity accuracy among aligned links and end-to-end recall have
separate denominators. LLM invocations count logical requests by table.
Transport retries are internal.
