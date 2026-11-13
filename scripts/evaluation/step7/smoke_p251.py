"""Smoke-test Step 7 on two frozen P251 production reconstructions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabulus.resolved_export import ENRICHMENT_COLUMNS, export_resolved_csvs
from tabulus.resolved_export.csv_io import read_csv


CASES = {
    17: ("page_022_table_017", "deterministic"),
    26: ("page_033_table_026", "llm_assisted"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, required=True, help="TabulusBench P251 directory.")
    parser.add_argument("--out", type=Path, required=True, help="New output directory outside P251.")
    parser.add_argument(
        "--direct-only", action="store_true", help="Run only the deterministic fragment offline."
    )
    args = parser.parse_args()

    paper = args.paper_root.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if out.is_relative_to(paper):
        parser.error("--out must be outside the paper directory.")
    runs = paper / "tabulus_runs"
    adapter = "tesseract-tatr"
    reconstruction = runs / "step2/production/run_01" / adapter / "reconstruction"
    matches = runs / "step5/production/run_01" / adapter / "references/reference_matches.json"
    resolution = runs / "step6/production/run_01/references/reference_resolution.json"
    tables_index = runs / "step1/production/artifacts/table-crops/tables_index.json"

    selected = {17: CASES[17]} if args.direct_only else CASES
    manifest = json.loads(matches.read_text(encoding="utf-8"))
    tables = [table for table in manifest["matched_tables"] if table["table_id"] in selected]
    if {table["table_id"] for table in tables} != selected.keys():
        parser.error("The frozen P251 Step 5 manifest is missing a smoke-test fragment.")

    out.mkdir(parents=True, exist_ok=False)
    subset = out / "reference_matches_subset.json"
    subset_manifest = {
        **manifest,
        "matched_tables": tables,
        "skipped_tables": [],
        "reference_tables_selected": len(tables),
        "reference_tables_checked": len(tables),
        "reference_tables_skipped": 0,
    }
    subset.write_text(
        json.dumps(subset_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result = export_resolved_csvs(
        subset,
        resolution,
        output_dir=out,
        reconstruction_root=reconstruction,
        tables_index_path=tables_index,
        orkg=True,
    )
    physical = {table["table_id"]: table for table in result.tables}
    upstream = {table["table_id"]: table for table in tables}
    projection = json.loads(result.orkg_manifest_path.read_text(encoding="utf-8"))
    if {table["table_id"] for table in projection["tables"]} != selected.keys():
        raise SystemExit(f"Projection manifest has an unexpected fragment set: {result.orkg_manifest_path}")
    failures = []
    for table in projection["tables"]:
        table_id = table["table_id"]
        expected_key, expected_route = selected[table_id]
        print(f"{table['table_key']} | {table.get('route')} | {table['status']} | "
              f"{table['llm_invocations']} planner calls | {table.get('export_rows', 0)} ORKG rows")
        if table["table_key"] != expected_key or table.get("route") != expected_route:
            failures.append(f"{expected_key} did not use the expected {expected_route} route.")
        expected_calls = 0 if expected_route == "deterministic" else 1
        if table["llm_invocations"] != expected_calls:
            failures.append(f"{expected_key} expected {expected_calls} planner calls.")
        if table["status"] != "exported":
            failures.append(f"{expected_key} produced no projection. {table.get('error', '')}".strip())
        else:
            expected_links = sorted(
                (match["row_index"], occurrence, index)
                for match in upstream[table_id]["matches"] if not match["is_header"]
                for occurrence, index in enumerate(match["matched_reference_indices"])
            )
            observed_links = sorted(
                (link["source_row"], link["occurrence"], link["reference_index"])
                for record in table["records"] for link in record["links"]
                if link["kind"] == "explicit"
            )
            if expected_links != observed_links:
                failures.append(f"{expected_key} did not preserve its explicit citation links.")
        resolved = physical[table_id]
        source_rows = read_csv(Path(resolved["source_prediction"]))
        resolved_rows = read_csv(Path(resolved["resolved_csv"]))
        if source_rows != [row[:-len(ENRICHMENT_COLUMNS)] for row in resolved_rows]:
            failures.append(f"{expected_key} changed scientific cells in the physical export.")

    print(f"Manifest at {result.orkg_manifest_path}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
