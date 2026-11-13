
from __future__ import annotations

import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from tabulus.cli import main
from tabulus.evaluation.step7 import evaluate_step7
from tabulus.resolved_export import export_resolved_csvs
from tabulus.resolved_export.csv_io import read_csv
from tabulus.resolved_export.structural_llm import (
    OpenAICompatibleStructurePlanner,
    StructureConfigurationError,
)


def _json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def _inputs(tmp_path, rows, citations, native=None):
    reconstruction = tmp_path / "crop" / "reconstructions" / "adapter"
    prediction = reconstruction / "predictions" / "table_001.csv"
    prediction.parent.mkdir(parents=True)
    with prediction.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    if native:
        _json(reconstruction / "native" / "table_001.json", {
            "source_image": "page_011_table_001.jpg", "native_markdown": [native],
        })
    else:
        _json(reconstruction / "parsed" / "table_001.json", {
            "source_image": "page_011_table_001.jpg",
        })
    ref = len(rows[0]) - 1
    matches = _json(tmp_path / "relocated" / "references" / "reference_matches.json", {
        "matched_tables": [{
            "table_id": 1,
            "source_parsed": str(reconstruction / "parsed" / "table_001.json"),
            "source_prediction": "predictions/table_001.csv",
            "reference_column_index": ref,
            "matches": [
                {"row_index": 0, "value": rows[0][ref], "is_header": True},
                *(
                    {"row_index": row, "value": rows[row][ref], "matched_reference_indices": indices,
                     "matched_references": [f"Work {i}" for i in indices]}
                    for row, indices in citations.items()
                ),
            ],
        }],
    })
    resolutions = _json(tmp_path / "references" / "reference_resolution.json", {
        "entries": [
            {"resolution": {"reference_index": 1, "raw_reference": "Work 1",
                            "status": "validated_with_doi", "canonical_doi": "10.1000/example"}},
            {"resolution": {"reference_index": 2, "raw_reference": "Work 2",
                            "status": "validated_without_doi", "canonical_title": "Second work",
                            "canonical_authors": ["M. Jones", "A. Smith"], "canonical_year": 2020}},
            {"resolution": {"reference_index": 3, "raw_reference": "Work 3", "status": "rejected"}},
        ],
    })
    return prediction, matches, resolutions


def test_cli_direct_export_never_invokes_planner_and_preserves_relocated_source(tmp_path, monkeypatch):
    prediction, matches, resolutions = _inputs(
        tmp_path, [["Material", "Dose", "Refs"], [" Al₂O₃ ", "40", "1,2"], ["ZnO", "10", "3"]],
        {1: [1, 2], 2: [3]},
    )
    before = prediction.read_bytes()

    def unexpected(*args, **kwargs):
        pytest.fail("A direct structural mapping must not invoke an LLM")

    monkeypatch.setattr(OpenAICompatibleStructurePlanner, "__call__", unexpected)
    monkeypatch.setattr(sys, "argv", [
        "tabulus", "export-resolved-csv", "--reference-matches", str(matches),
        "--reference-resolution", str(resolutions), "--orkg",
    ])
    main()
    output = prediction.parent.parent / "resolved_reference_tables" / "orkg"
    manifest = _load(output / "manifest.json")["tables"][0]
    assert manifest["route"] == "deterministic"
    assert manifest["llm_invocations"] == 0
    assert manifest["export_rows"] == 2
    assert manifest["omissions"][0]["identity"]["status"] == "rejected"
    assert _csv(output / manifest["csv"])[1:] == [
        [" Al₂O₃ ", "40", "10.1000/example", "", "", ""],
        [" Al₂O₃ ", "40", "", "Second work", "M. Jones and A. Smith", "2020"],
    ]
    assert prediction.read_bytes() == before
    assert manifest["records"][1]["links"][0]["reference_index"] == 3


@pytest.fixture
def grouped(tmp_path):
    rows = [
        ["Group", "Conditions", "", "Refs"],
        ["", "Temperature", "Pressure", ""],
        ["Material A", "40", "1", "1,2"],
        ["", "45", "2", ""],
        ["Material B", "90", "3", "3"],
        ["", "95", "4", ""],
    ]
    native = (
        "<table><thead><tr><th>Group</th><th colspan='2'>Conditions</th><th>Refs</th></tr>"
        "<tr><th></th><th>Temperature</th><th>Pressure</th><th></th></tr></thead>"
        "<tr><td rowspan='2'>Material A</td><td>40</td><td>1</td><td rowspan='2'>1,2</td></tr>"
        "<tr><td>45</td><td>2</td></tr>"
        "<tr><td>Material B</td><td>90</td><td>3</td><td>3</td></tr>"
        "<tr><td></td><td>95</td><td>4</td><td></td></tr></table>"
    )
    inputs = _inputs(tmp_path, rows, {2: [1, 2], 4: [3]}, native)
    plan = {
        "schema_version": 1,
        "header_rows": [0, 1],
        "columns": [
            {"label": "Group", "header_cells": [[0, 0]], "source_columns": [0]},
            {"label": "Conditions / Temperature", "header_cells": [[0, 1], [1, 1]], "source_columns": [1]},
            {"label": "Conditions / Pressure", "header_cells": [[0, 1], [1, 2]], "source_columns": [2]},
        ],
        "propagations": [{"source": [2, 0], "column": 0, "rows": [3, 4], "group": [2, 4]}],
        "inherit_references": [{"source_row": 2, "rows": [3, 4], "group": [2, 4]}],
        "blanks": [{"column": 0, "rows": [5, 6], "kind": "intentional"}],
    }
    return inputs, plan


def test_assisted_plan_preserves_physical_and_records_inheritance(grouped, tmp_path):
    (prediction, matches, resolution), plan = grouped
    before = prediction.read_bytes()
    calls = []

    def planner(evidence):
        calls.append(evidence)
        assert any(s["source"] == [2, 3] and s["row_end"] == 4 for s in evidence["aligned_spans"])
        assert evidence["explicit_citations"] == [{"row": 2, "occurrences": 2}, {"row": 4, "occurrences": 1}]
        return plan, {"provider": "fixture", "usage": {"total_tokens": 120}}

    result = export_resolved_csvs(
        matches, resolution, output_dir=tmp_path / "output", orkg=True, structure_planner=planner,
    )
    manifest = _load(result.orkg_manifest_path)
    table = manifest["tables"][0]
    assert len(calls) == table["llm_invocations"] == 1
    assert table["route"] == "llm_assisted"
    assert table["status"] == "exported"
    assert table["plan"] == plan
    assert table["records"][1]["cells"][0]["value"] == "Material A"
    assert table["records"][1]["cells"][0]["sources"] == [[2, 0]]
    assert table["records"][1]["links"][1]["kind"] == "inherited"
    assert table["records"][1]["links"][1]["source_row"] == 2
    assert table["records"][3]["cells"][0]["blanks"] == [{"row": 5, "kind": "intentional"}]
    assert _csv(result.orkg_manifest_path.parent / table["csv"])[3][:3] == ["Material A", "45", "2"]
    assert prediction.read_bytes() == before

    # Independent reference annotations for the assisted output. The expected DOI
    # differs deliberately so reference resolution can fail with correct structure.
    gold = _json(tmp_path / "gold.json", {
        "schema_version": 1, "split": "held_out",
        "tables": [{
            "table_key": "page_011_table_001", "route": "llm_assisted",
            "columns": [
                {"label": "Group", "source_columns": [0], "header_cells": [[0, 0]]},
                {"label": "Conditions / Temperature", "header_cells": [[0, 1], [1, 1]]},
                {"label": "Conditions / Pressure", "header_cells": [[0, 1], [1, 2]]},
            ],
            "records": [
                {"source_rows": [2], "expansion": 2},
                {"source_rows": [3], "expansion": 2, "cells": [{
                    "value": "Material A", "sources": [[2, 0]],
                    "propagations": [{"target_row": 3, "source": [2, 0], "mode": "fill"}],
                }]},
                {"source_rows": [4], "expansion": 1},
                {"source_rows": [5], "expansion": 0},
            ],
            "links": [{
                "source_row": 2, "target_row": 3, "source_column": 3, "reference_index": 1,
                "occurrence": 0, "kind": "inherited", "identity": {"canonical_doi": "10.1000/other"},
            }],
        }],
    })
    metrics = evaluate_step7(result.orkg_manifest_path, gold_path=gold)
    assert metrics["routing"]["accuracy"] == 1
    assert metrics["structure_and_citations"]["row_correspondence"]["f1"] == 1
    assert metrics["structure_and_citations"]["value_propagation"]["f1"] == 1
    assert metrics["structure_and_citations"]["citations_inherited"]["recall"] == 1
    assert metrics["structure_and_citations"]["citations_inherited"]["false_positive"] == 1
    assert metrics["structural_comparisons"]["header_hierarchy"]["accuracy"] == 1
    assert metrics["structural_comparisons"]["cell_placement"]["accuracy"] == 1
    assert metrics["structural_comparisons"]["row_expansion"]["accuracy"] == 1
    assert metrics["identity"]["canonical_doi"]["conditional_accuracy"] == 0
    assert metrics["llm"]["reported_tokens"]["total_tokens"] == 120


@pytest.mark.parametrize("mutation, message", [
    ({"rewritten_rows": [["invented"]]}, "Invalid fields"),
    ({"propagations": [{"source": [2, 0], "column": 0, "rows": [3, 6], "group": [2, 6]}]},
     "nonblank destination"),
    ({"inherit_references": [{"source_row": 2, "rows": [3, 6], "group": [2, 6]}]},
     "intervening citation"),
])
def test_assisted_plan_rejects_invented_data_and_cross_group_copies(grouped, tmp_path, mutation, message):
    (_, matches, resolution), plan = grouped
    result = export_resolved_csvs(
        matches, resolution, output_dir=tmp_path / "output", orkg=True,
        structure_planner=lambda _: ({**plan, **mutation}, {}),
    )
    table = _load(result.orkg_manifest_path)["tables"][0]
    assert table["status"] == "abstained"
    assert message in table["error"]
    assert "csv" not in table
    assert Path(result.tables[0]["resolved_csv"]).is_file()


def test_assisted_plan_labels_blank_header_without_claiming_header_evidence(grouped, tmp_path):
    (_, matches, resolution), plan = grouped
    inferred = {**plan, "columns": [dict(column) for column in plan["columns"]]}
    inferred["columns"][2]["header_cells"] = [[0, 2]]
    rejected = export_resolved_csvs(
        matches, resolution, output_dir=tmp_path / "rejected", orkg=True,
        structure_planner=lambda _: (inferred, {}),
    )
    table = _load(rejected.orkg_manifest_path)["tables"][0]
    assert table["status"] == "abstained"
    assert "invalid header evidence at [0, 2]" in table["error"]

    inferred["columns"][2]["header_cells"] = []
    accepted = export_resolved_csvs(
        matches, resolution, output_dir=tmp_path / "accepted", orkg=True,
        structure_planner=lambda _: (inferred, {}),
    )
    table = _load(accepted.orkg_manifest_path)["tables"][0]
    assert table["status"] == "exported"
    assert table["columns"][2]["source_columns"] == [2]
    assert table["columns"][2]["header_cells"] == []


def test_nested_heading_groups_and_phase_context(tmp_path):
    rows = [
        ["System", "Phase", "Refs"],
        ["Category", "", ""],
        ["Process A", "Cubic", ""],
        ["", "300", "1"],
        ["", "", "1"],
        ["Process B", "Hexagonal", ""],
        ["", "400", "1"],
    ]
    _, matches, resolution = _inputs(tmp_path, rows, {3: [1], 4: [1], 6: [1]})
    plan = {
        "schema_version": 1,
        "header_rows": [0],
        "group_rows": [1, 2, 5],
        "groups": [[1, 7], [2, 5], [5, 7]],
        "columns": [
            {"label": "Category", "source_columns": [], "header_cells": [[0, 0]]},
            {"label": "System", "source_columns": [0], "header_cells": [[0, 0]]},
            {"label": "Phase", "source_columns": [1], "header_cells": [[0, 1]]},
        ],
        "propagations": [
            {"source": [1, 0], "column": 0, "rows": [3, 7], "group": [1, 7]},
            {"source": [2, 0], "column": 1, "rows": [3, 5], "group": [2, 5]},
            {"source": [5, 0], "column": 1, "rows": [6, 7], "group": [5, 7]},
            {"source": [2, 1], "column": 2, "rows": [3, 5], "group": [2, 5], "mode": "prefix"},
            {"source": [5, 1], "column": 2, "rows": [6, 7], "group": [5, 7], "mode": "prefix"},
        ],
    }
    result = export_resolved_csvs(
        matches, resolution, orkg=True, structure_planner=lambda _: (plan, {}),
    )
    table = _load(result.orkg_manifest_path)["tables"][0]
    assert table["status"] == "exported"
    assert [[cell["value"] for cell in record["cells"]] for record in table["records"]] == [
        ["Category", "Process A", "Cubic: 300"],
        ["Category", "Process A", ""],
        ["Category", "Process B", "Hexagonal: 400"],
    ]


def test_projection_integrity_is_separate_from_structural_metrics(tmp_path):
    _, matches, resolution = _inputs(tmp_path, [["Value", "Refs"], ["1", "1"]], {1: [1]})
    result = export_resolved_csvs(matches, resolution, orkg=True)
    gold = _json(tmp_path / "gold.json", {
        "schema_version": 1, "split": "held_out",
        "tables": [{"table_key": "page_011_table_001", "route": "deterministic"}],
    })
    projection = result.orkg_manifest_path.parent / "table_001_orkg.csv"
    projection.write_text(projection.read_text(encoding="utf-8").replace("10.1000/example", "10.1000/other"),
                          encoding="utf-8")
    scored = evaluate_step7(result.orkg_manifest_path, gold_path=gold)
    assert scored["routing"]["accuracy"] == 1
    assert scored["coverage"]["gold_fragments_exported"] == 0
    assert scored["coverage"]["counts"]["missing_or_inconsistent_csv"] == 1


def test_step7_csv_can_read_a_wide_enrichment_field(tmp_path):
    path = tmp_path / "large.csv"
    large = "λ" * 174_875
    path.write_text(f"col\n{large}\n", encoding="utf-8")
    assert read_csv(path) == [["col"], [large]]


def test_controlled_uncovered_identity_keeps_explicit_link(tmp_path):
    _, matches, resolution = _inputs(tmp_path, [["Value", "Refs"], ["sample", "4"]], {1: [4]})
    with pytest.raises(ValueError, match="controlled gold_index"):
        export_resolved_csvs(matches, resolution, allow_uncovered_identities=True)
    payload = _load(matches)
    payload["index_namespace"] = "gold_index"
    _json(matches, payload)
    result = export_resolved_csvs(matches, resolution, allow_uncovered_identities=True, orkg=True)
    physical = _csv(Path(result.tables[0]["resolved_csv"]))
    assert json.loads(physical[1][2]) == [4]
    assert json.loads(physical[1][3]) == ["identity_unavailable"]
    manifest = _load(result.orkg_manifest_path)
    assert manifest["index_namespace"] == "gold_index"
    table = manifest["tables"][0]
    assert table["records"][0]["links"][0]["reference_index"] == 4
    assert table["omissions"][0]["reason"] == "missing_identity"


def test_evaluator_requires_crosswalk_for_production_indices(tmp_path):
    _, matches, resolution = _inputs(tmp_path, [["Value", "Refs"], ["sample", "1"]], {1: [1]})
    result = export_resolved_csvs(matches, resolution, orkg=True)
    gold = _json(tmp_path / "gold.json", {
        "schema_version": 1, "split": "held_out", "index_namespace": "gold_index",
        "tables": [{"table_key": "page_011_table_001", "links": [{
            "source_row": 1, "target_row": 1, "source_column": 1, "occurrence": 0,
            "kind": "explicit", "reference_index": 7, "identity": {"canonical_doi": "10.1000/example"},
        }]}],
    })
    with pytest.raises(ValueError, match="crosswalk"):
        evaluate_step7(result.orkg_manifest_path, gold_path=gold)
    crosswalk = _json(tmp_path / "crosswalk.json", {"bibliography_indices": {"1": 7}})
    metrics = evaluate_step7(result.orkg_manifest_path, gold_path=gold, alignment_path=crosswalk)
    assert metrics["structure_and_citations"]["citations_explicit"]["recall"] == 1
    assert metrics["identity"]["canonical_doi"]["conditional_accuracy"] == 1


def test_validate_stale_row_and_bibliography_before_export(tmp_path):
    prediction, matches, resolution = _inputs(tmp_path, [["Value", "Refs"], ["1", "1"]], {1: [1]})
    output = tmp_path / "output"
    prediction.write_text("Value,Refs\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Step 5 citation differs"):
        export_resolved_csvs(matches, resolution, output_dir=output)
    assert not output.exists()
    prediction.write_text("Value,Refs\n1,1\n", encoding="utf-8")
    payload = _load(resolution)
    payload["entries"][0]["resolution"]["raw_reference"] = "A different bibliography entry"
    _json(resolution, payload)
    with pytest.raises(ValueError, match="Bibliography evidence differs"):
        export_resolved_csvs(matches, resolution, output_dir=output)
    assert not output.exists()


def test_shared_endpoint_and_independent_openrouter_config(monkeypatch):
    from tabulus.resolved_export import structural_llm

    for suffix in ("BASE_URL", "MODEL", "API_KEY", "PROVIDER"):
        monkeypatch.delenv(f"TABULUS_STEP7_LLM_{suffix}", raising=False)
    monkeypatch.delenv("TABULUS_STEP7_LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("TABULUS_STEP7_LLM_REASONING_MAX_TOKENS", raising=False)
    monkeypatch.setenv("TABULUS_LLM_BASE_URL", "https://chat-ai.academiccloud.de/v1")
    monkeypatch.setenv("TABULUS_LLM_MODEL", "shared-model")
    monkeypatch.setenv("TABULUS_LLM_API_KEY", "shared-key")
    monkeypatch.setenv("TABULUS_LLM_PROVIDER", "saia")
    requests = []

    def respond(url, *, api_key, payload, timeout_seconds):
        requests.append((url, api_key, payload["model"], timeout_seconds))
        return {"choices": [{"message": {"content": '{"schema_version": 1}'}}]}

    monkeypatch.setattr(structural_llm, "_request_json", respond)
    _, metadata = OpenAICompatibleStructurePlanner()({"rows": []})
    assert metadata["provider"] == "saia"
    assert requests[-1] == ("https://chat-ai.academiccloud.de/v1/chat/completions", "shared-key", "shared-model", 60)
    openrouter = OpenAICompatibleStructurePlanner(
        base_url="https://openrouter.ai/api/v1", model="vendor/model", provider="openrouter",
    )
    with pytest.raises(StructureConfigurationError):
        openrouter({"rows": []})
    assert len(requests) == 1
    monkeypatch.setenv("TABULUS_STEP7_LLM_API_KEY", "independent-key")
    monkeypatch.setenv("TABULUS_STEP7_LLM_TIMEOUT_SECONDS", "180")
    _, metadata = openrouter({"rows": []})
    assert metadata["provider"] == "openrouter"
    assert requests[-1] == ("https://openrouter.ai/api/v1/chat/completions", "independent-key", "vendor/model", 180)
    assert metadata["timeout_seconds"] == 180
    monkeypatch.setenv("TABULUS_STEP7_LLM_TIMEOUT_SECONDS", "NaN")
    with pytest.raises(StructureConfigurationError, match="positive number"):
        openrouter({"rows": []})
    assert len(requests) == 2


def test_reasoning_budget_and_truncated_response_are_recorded(grouped, tmp_path, monkeypatch):
    from tabulus.resolved_export import structural_llm

    _, matches, resolution = grouped[0]
    monkeypatch.setenv("TABULUS_STEP7_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("TABULUS_STEP7_LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
    monkeypatch.setenv("TABULUS_STEP7_LLM_API_KEY", "fixture-key")
    monkeypatch.setenv("TABULUS_STEP7_LLM_REASONING_MAX_TOKENS", "2048")
    monkeypatch.delenv("TABULUS_STEP7_LLM_TIMEOUT_SECONDS", raising=False)

    def respond(url, *, api_key, payload, timeout_seconds):
        assert payload["max_tokens"] == 8192
        assert payload["reasoning"] == {"max_tokens": 2048}
        return {
            "id": "truncated-fixture",
            "choices": [{"finish_reason": "length", "message": {"content": ""}}],
            "usage": {"completion_tokens": 8192, "completion_tokens_details": {"reasoning_tokens": 8192}},
        }

    monkeypatch.setattr(structural_llm, "_request_json", respond)
    result = export_resolved_csvs(matches, resolution, output_dir=tmp_path / "output", orkg=True)
    table = _load(result.orkg_manifest_path)["tables"][0]
    assert table["status"] == "abstained"
    assert "exceeded the response limit" in table["error"]
    assert table["llm"]["finish_reason"] == "length"
    assert table["llm"]["usage"]["reasoning_tokens"] == 8192
    assert table["llm"]["reasoning_max_tokens"] == 2048

    monkeypatch.setenv("TABULUS_STEP7_LLM_REASONING_MAX_TOKENS", "8192")
    with pytest.raises(StructureConfigurationError, match="between 1 and 8191"):
        OpenAICompatibleStructurePlanner()({"rows": []})
