from __future__ import annotations

from tabulus.reference_resolution import (
    CoreAssessmentStatus,
    CoreRateLimit,
    CoreSearchResponse,
    CrossrefRetrieval,
    ReferenceEvidence,
    ResolutionCandidate,
    ScholarlyResolutionStatus,
    assess_core,
    resolve_crossref_then_core,
)


def _evidence(
    *,
    index: int = 1,
) -> ReferenceEvidence:
    return ReferenceEvidence(
        reference_index=index,
        raw_reference=(
            "Smith J. Atomic layer deposition of example films. "
            "Example Journal 12, 100-110 (2020)."
        ),
        title=(
            "Atomic layer deposition of example films"
        ),
        authors=("J. Smith",),
        year=2020,
        venue="Example Journal",
        volume="12",
        pages="100-110",
    )


def _candidate(
    *,
    source: str,
    doi: str = "10.1000/example",
) -> ResolutionCandidate:
    return ResolutionCandidate(
        source=source,
        source_id=doi or "work-1",
        doi=doi,
        title=(
            "Atomic Layer Deposition of Example Films"
        ),
        authors=("Smith, J.",),
        year=2020,
        venue="Example Journal",
        volume="12",
        pages="100-110",
    )


def _crossref_retrieval(
    *,
    index: int = 1,
    candidates=(),
) -> CrossrefRetrieval:
    return CrossrefRetrieval(
        reference_index=index,
        raw_reference="Example raw reference.",
        existing_doi="",
        existing_doi_checked=False,
        existing_doi_candidate=None,
        bibliographic_search_performed=True,
        bibliographic_candidates=tuple(candidates),
    )


def _core_response(
    *candidates,
) -> CoreSearchResponse:
    return CoreSearchResponse(
        candidates=tuple(candidates),
        total_hits=len(candidates),
        search_id="test-search",
        limit=5,
        offset=0,
        rate_limit=CoreRateLimit(
            limit=500,
            remaining=499,
        ),
    )


def test_assess_core_accepts_strong_candidate() -> None:
    evidence = _evidence()
    candidate = _candidate(
        source="core",
    )

    result = assess_core(
        evidence,
        _core_response(candidate),
    )

    assert result.status == (
        CoreAssessmentStatus
        .VALIDATED_CORE_MATCH
    )
    assert result.selected_candidate == candidate
    assert result.selected_score is not None
    assert result.selected_score.strong_match is True


def test_assess_core_no_candidates_requires_llm() -> None:
    result = assess_core(
        _evidence(),
        _core_response(),
    )

    assert result.status == (
        CoreAssessmentStatus.NEEDS_LLM
    )
    assert result.selected_candidate is None


def test_assess_core_close_candidates_requires_llm() -> None:
    evidence = _evidence()

    best = _candidate(
        source="core",
        doi="10.1000/best",
    )

    close = ResolutionCandidate(
        source="core",
        source_id="close",
        doi="10.1000/close",
        title=(
            "Atomic Layer Deposition of Example Films"
        ),
        authors=("Smith, J.",),
        year=2020,
        venue="Example Journal",
        volume="99",
        pages="100-110",
    )

    result = assess_core(
        evidence,
        _core_response(close, best),
    )

    assert result.status == (
        CoreAssessmentStatus.NEEDS_LLM
    )
    assert len(result.ranked_candidates) == 2


class _FakeCoreClient:
    def __init__(self):
        self.calls: list[str] = []
        self.responses: dict[
            str,
            CoreSearchResponse,
        ] = {}

    def search_works(
        self,
        reference_text: str,
    ) -> CoreSearchResponse:
        self.calls.append(reference_text)
        return self.responses.get(
            reference_text,
            _core_response(),
        )


def test_crossref_validated_reference_skips_core() -> None:
    evidence = _evidence()
    crossref_candidate = _candidate(
        source="crossref",
    )

    core = _FakeCoreClient()

    results = resolve_crossref_then_core(
        [evidence],
        [
            _crossref_retrieval(
                candidates=(crossref_candidate,),
            )
        ],
        core,
    )

    assert core.calls == []
    assert results[0].status == (
        ScholarlyResolutionStatus
        .VALIDATED_CROSSREF
    )
    assert (
        results[0].selected_candidate.source
        == "crossref"
    )


def test_crossref_failure_falls_through_to_core() -> None:
    evidence = _evidence()
    core_candidate = _candidate(
        source="core",
    )

    core = _FakeCoreClient()
    core.responses[
        evidence.raw_reference
    ] = _core_response(core_candidate)

    results = resolve_crossref_then_core(
        [evidence],
        [
            _crossref_retrieval(
                candidates=(),
            )
        ],
        core,
    )

    assert core.calls == [
        evidence.raw_reference
    ]
    assert results[0].status == (
        ScholarlyResolutionStatus
        .VALIDATED_CORE
    )
    assert (
        results[0].selected_candidate.source
        == "core"
    )


def test_unresolved_crossref_and_core_routes_to_llm() -> None:
    evidence = _evidence()

    core = _FakeCoreClient()

    results = resolve_crossref_then_core(
        [evidence],
        [
            _crossref_retrieval(
                candidates=(),
            )
        ],
        core,
    )

    assert results[0].status == (
        ScholarlyResolutionStatus.NEEDS_LLM
    )
    assert results[0].selected_candidate is None
    assert (
        results[0].core_assessment is not None
    )
    assert (
        results[0]
        .core_assessment
        .status
        == CoreAssessmentStatus.NEEDS_LLM
    )


def test_core_can_validate_scholarly_work_without_doi() -> None:
    evidence = _evidence()

    core_candidate = _candidate(
        source="core",
        doi="",
    )

    result = assess_core(
        evidence,
        _core_response(core_candidate),
    )

    assert result.status == (
        CoreAssessmentStatus
        .VALIDATED_CORE_MATCH
    )
    assert result.selected_candidate is not None
    assert result.selected_candidate.doi == ""



def test_core_failure_reports_bibliography_index() -> None:
    import pytest

    from tabulus.reference_resolution.core import CoreError

    class FailingCore:
        def search_works(
            self,
            reference_text,
        ):
            raise CoreError(
                "CORE request returned HTTP 500 "
                "after 4 attempts."
            )

    with pytest.raises(
        CoreError,
        match="bibliography index 1",
    ):
        resolve_crossref_then_core(
            [_evidence()],
            [
                _crossref_retrieval(
                    candidates=(),
                )
            ],
            FailingCore(),
        )
