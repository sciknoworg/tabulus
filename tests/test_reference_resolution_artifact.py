from __future__ import annotations

import json

import pytest

from tabulus.reference_resolution import (
    CrossrefAssessment,
    CrossrefAssessmentStatus,
    LLMAdjudicationResponse,
    LLMDecision,
    LLMDecisionType,
    LLMUsage,
    ReferenceResolution,
    ReferenceResolutionTrace,
    ResolutionStatus,
    ScholarlyResolution,
    ScholarlyResolutionStatus,
    build_reference_resolution_payload,
    default_reference_resolution_path,
    write_reference_resolution_artifact,
)


def _initial_resolution(
    index: int,
) -> ScholarlyResolution:
    crossref = CrossrefAssessment(
        reference_index=index,
        status=(
            CrossrefAssessmentStatus
            .NEEDS_CORE
        ),
        selected_candidate=None,
        selected_score=None,
        ranked_candidates=(),
        reason="Test Crossref evidence.",
    )

    return ScholarlyResolution(
        reference_index=index,
        raw_reference=f"Reference {index}.",
        status=ScholarlyResolutionStatus.NEEDS_LLM,
        selected_candidate=None,
        selected_score=None,
        crossref_assessment=crossref,
        core_assessment=None,
        reason="Test initial resolution.",
    )


def _llm_response() -> LLMAdjudicationResponse:
    return LLMAdjudicationResponse(
        decision=LLMDecision(
            decision=LLMDecisionType.REJECT_ALL,
            confidence=0.9,
            evidence=("No candidate supported.",),
        ),
        model="qwen3.6-35b-a3b",
        response_id="response-test",
        finish_reason="stop",
        usage=LLMUsage(
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
        ),
    )


def _trace(
    index: int,
    status: ResolutionStatus,
    *,
    llm_used: bool = False,
    retry_used: bool = False,
) -> ReferenceResolutionTrace:
    doi = (
        f"10.1000/{index}"
        if status
        == ResolutionStatus.VALIDATED_WITH_DOI
        else ""
    )

    resolution = ReferenceResolution(
        reference_index=index,
        raw_reference=f"Reference {index}.",
        status=status,
        canonical_doi=doi,
        canonical_title=(
            f"Resolved work {index}"
            if status != ResolutionStatus.REJECTED
            else ""
        ),
        source=(
            "crossref"
            if status
            == ResolutionStatus.VALIDATED_WITH_DOI
            else (
                "core"
                if status
                == ResolutionStatus.VALIDATED_WITHOUT_DOI
                else ""
            )
        ),
        confidence=(
            0.95
            if status != ResolutionStatus.REJECTED
            else None
        ),
        reason="Test final decision.",
    )

    return ReferenceResolutionTrace(
        resolution=resolution,
        initial_scholarly_resolution=(
            _initial_resolution(index)
        ),
        first_llm_response=(
            _llm_response()
            if llm_used
            else None
        ),
        retry_query=(
            "improved bibliographic query"
            if retry_used
            else ""
        ),
    )


def test_payload_sorts_entries_and_counts_final_statuses() -> None:
    payload = build_reference_resolution_payload(
        [
            _trace(
                3,
                ResolutionStatus.REJECTED,
                llm_used=True,
            ),
            _trace(
                1,
                ResolutionStatus.VALIDATED_WITH_DOI,
            ),
            _trace(
                2,
                ResolutionStatus.VALIDATED_WITHOUT_DOI,
                llm_used=True,
                retry_used=True,
            ),
        ]
    )

    assert payload["schema_version"] == 1
    assert payload["resolution_count"] == 3

    assert payload["status_counts"] == {
        "validated_with_doi": 1,
        "validated_without_doi": 1,
        "rejected": 1,
    }

    assert payload["llm_adjudicated_count"] == 2
    assert payload["retry_count"] == 1

    assert [
        entry["resolution"]["reference_index"]
        for entry in payload["entries"]
    ] == [1, 2, 3]


def test_payload_rejects_duplicate_bibliography_indices() -> None:
    with pytest.raises(
        ValueError,
        match="Duplicate",
    ):
        build_reference_resolution_payload(
            [
                _trace(
                    1,
                    ResolutionStatus.REJECTED,
                ),
                _trace(
                    1,
                    ResolutionStatus.REJECTED,
                ),
            ]
        )


def test_payload_rejects_non_final_unresolved_status() -> None:
    with pytest.raises(
        ValueError,
        match="cannot contain status",
    ):
        build_reference_resolution_payload(
            [
                _trace(
                    1,
                    ResolutionStatus.UNRESOLVED,
                )
            ]
        )


def test_writer_uses_canonical_references_path(
    tmp_path,
) -> None:
    expected = (
        tmp_path
        / "references"
        / "reference_resolution.json"
    )

    assert (
        default_reference_resolution_path(
            tmp_path
        )
        == expected
    )

    written = write_reference_resolution_artifact(
        [
            _trace(
                1,
                ResolutionStatus.VALIDATED_WITH_DOI,
            ),
            _trace(
                2,
                ResolutionStatus.REJECTED,
                llm_used=True,
            ),
        ],
        tmp_path,
    )

    assert written == expected
    assert written.is_file()

    payload = json.loads(
        written.read_text(
            encoding="utf-8"
        )
    )

    assert payload["resolution_count"] == 2
    assert payload["status_counts"][
        "validated_with_doi"
    ] == 1
    assert payload["status_counts"][
        "rejected"
    ] == 1

    # Atomic writer should not leave temporary siblings behind.
    assert list(
        written.parent.glob(
            ".reference_resolution.json.*.tmp"
        )
    ) == []


def test_artifact_records_llm_provenance_but_no_api_configuration() -> None:
    payload = build_reference_resolution_payload(
        [
            _trace(
                7,
                ResolutionStatus.REJECTED,
                llm_used=True,
                retry_used=True,
            )
        ]
    )

    serialized = json.dumps(
        payload
    )

    entry = payload["entries"][0]

    assert (
        entry["first_llm_response"]["model"]
        == "qwen3.6-35b-a3b"
    )
    assert (
        entry["first_llm_response"]["usage"][
            "total_tokens"
        ]
        == 70
    )

    assert "api_key" not in serialized
    assert "authorization" not in serialized.casefold()



def _serialized_final_trace(
    reference_index: int,
    *,
    status: str = "validated_with_doi",
) -> dict:
    return {
        "resolution": {
            "reference_index": reference_index,
            "raw_reference": (
                f"Reference {reference_index}"
            ),
            "status": status,
            "canonical_doi": (
                f"10.1000/{reference_index}"
                if status
                == "validated_with_doi"
                else ""
            ),
            "canonical_title": "",
            "canonical_authors": [],
            "canonical_year": None,
            "canonical_venue": "",
            "source": "",
            "confidence": None,
            "reason": "",
        },
        "initial_scholarly_resolution": {},
        "first_llm_response": None,
        "retry_used": False,
        "retry_query": "",
        "retry_crossref_assessment": None,
        "retry_core_assessment": None,
        "second_llm_response": None,
    }


def test_reference_resolution_checkpoint_round_trip(
    tmp_path,
) -> None:
    from tabulus.reference_resolution.artifact import (
        default_reference_resolution_checkpoint_path,
        load_reference_resolution_checkpoint,
        write_reference_resolution_checkpoint,
    )

    entries = (
        _serialized_final_trace(4),
        _serialized_final_trace(
            2,
            status="rejected",
        ),
    )

    output = (
        write_reference_resolution_checkpoint(
            entries,
            tmp_path,
            run_fingerprint="abc123",
            target_indices=(
                1,
                2,
                3,
                4,
                5,
            ),
        )
    )

    assert output == (
        default_reference_resolution_checkpoint_path(
            tmp_path
        )
    )

    loaded = (
        load_reference_resolution_checkpoint(
            tmp_path,
            expected_run_fingerprint="abc123",
            expected_target_indices=(
                1,
                2,
                3,
                4,
                5,
            ),
        )
    )

    assert [
        entry["resolution"][
            "reference_index"
        ]
        for entry in loaded
    ] == [
        2,
        4,
    ]


def test_reference_resolution_checkpoint_rejects_wrong_run_or_targets(
    tmp_path,
) -> None:
    import pytest

    from tabulus.reference_resolution.artifact import (
        load_reference_resolution_checkpoint,
        write_reference_resolution_checkpoint,
    )

    write_reference_resolution_checkpoint(
        (
            _serialized_final_trace(1),
        ),
        tmp_path,
        run_fingerprint="run-a",
        target_indices=(
            1,
            2,
            3,
        ),
    )

    with pytest.raises(
        ValueError,
        match="run fingerprint",
    ):
        load_reference_resolution_checkpoint(
            tmp_path,
            expected_run_fingerprint="run-b",
            expected_target_indices=(
                1,
                2,
                3,
            ),
        )

    with pytest.raises(
        ValueError,
        match="target set",
    ):
        load_reference_resolution_checkpoint(
            tmp_path,
            expected_run_fingerprint="run-a",
            expected_target_indices=(
                1,
                2,
                4,
            ),
        )


def test_reference_resolution_checkpoint_rejects_unresolved_entry(
    tmp_path,
) -> None:
    import pytest

    from tabulus.reference_resolution.artifact import (
        write_reference_resolution_checkpoint,
    )

    with pytest.raises(
        ValueError,
        match="only final scientific statuses",
    ):
        write_reference_resolution_checkpoint(
            (
                _serialized_final_trace(
                    1,
                    status="unresolved",
                ),
            ),
            tmp_path,
            run_fingerprint="abc",
            target_indices=(
                1,
                2,
            ),
        )


def test_reference_resolution_payload_from_serialized_entries() -> None:
    from tabulus.reference_resolution.artifact import (
        build_reference_resolution_payload_from_entries,
    )

    first = _serialized_final_trace(
        2,
        status="rejected",
    )
    second = _serialized_final_trace(
        1,
        status="validated_with_doi",
    )

    second[
        "first_llm_response"
    ] = {
        "decision": {}
    }
    second[
        "retry_used"
    ] = True

    payload = (
        build_reference_resolution_payload_from_entries(
            (
                first,
                second,
            )
        )
    )

    assert payload[
        "resolution_count"
    ] == 2

    assert payload[
        "status_counts"
    ] == {
        "validated_with_doi": 1,
        "validated_without_doi": 0,
        "rejected": 1,
    }

    assert payload[
        "llm_adjudicated_count"
    ] == 1

    assert payload[
        "retry_count"
    ] == 1

    assert [
        entry["resolution"][
            "reference_index"
        ]
        for entry in payload[
            "entries"
        ]
    ] == [
        1,
        2,
    ]


def test_stage6_run_fingerprint_is_match_order_independent(
    tmp_path,
) -> None:
    from tabulus.reference_resolution.artifact import (
        build_reference_resolution_run_fingerprint,
    )

    bibliography = (
        tmp_path
        / "bibliography.json"
    )
    first = (
        tmp_path
        / "a.json"
    )
    second = (
        tmp_path
        / "b.json"
    )

    bibliography.write_text(
        '{"entries":[]}',
        encoding="utf-8",
    )
    first.write_text(
        '{"a":1}',
        encoding="utf-8",
    )
    second.write_text(
        '{"b":2}',
        encoding="utf-8",
    )

    forward = (
        build_reference_resolution_run_fingerprint(
            bibliography,
            (
                first,
                second,
            ),
        )
    )

    reverse = (
        build_reference_resolution_run_fingerprint(
            bibliography,
            (
                second,
                first,
            ),
        )
    )

    assert forward == reverse



def test_stage6_run_fingerprint_changes_with_resolver_configuration(
    tmp_path,
) -> None:
    from tabulus.reference_resolution.artifact import (
        build_reference_resolution_run_fingerprint,
    )

    bibliography = (
        tmp_path
        / "bibliography.json"
    )
    matches = (
        tmp_path
        / "reference_matches.json"
    )

    bibliography.write_text(
        '{"entries":[]}',
        encoding="utf-8",
    )
    matches.write_text(
        '{"matched_tables":[]}',
        encoding="utf-8",
    )

    first = build_reference_resolution_run_fingerprint(
        bibliography,
        (
            matches,
        ),
        resolver_configuration=(
            "provider=openai-compatible;"
            "base_url=https://example.test/v1;"
            "model=model-a;"
            "thinking=false"
        ),
    )

    second = build_reference_resolution_run_fingerprint(
        bibliography,
        (
            matches,
        ),
        resolver_configuration=(
            "provider=openai-compatible;"
            "base_url=https://example.test/v1;"
            "model=model-b;"
            "thinking=false"
        ),
    )

    assert first != second
