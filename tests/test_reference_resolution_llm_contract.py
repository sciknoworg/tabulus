from __future__ import annotations

import pytest

from tabulus.reference_resolution import (
    CandidateScore,
    CoreAssessment,
    CoreAssessmentStatus,
    CrossrefAssessment,
    CrossrefAssessmentStatus,
    LLMDecisionType,
    RankedCandidate,
    ReferenceEvidence,
    ResolutionCandidate,
    ScholarlyResolution,
    ScholarlyResolutionStatus,
    build_llm_adjudication_case,
    parse_llm_decision,
)


def _score(
    value: float,
) -> CandidateScore:
    return CandidateScore(
        score=value,
        comparable_weight=0.85,
        field_scores=(
            ("title", value),
            ("authors", 1.0),
            ("year", 1.0),
        ),
        sufficient_evidence=True,
        strong_match=False,
    )


def _candidate(
    *,
    source: str,
    doi: str,
    title: str,
) -> ResolutionCandidate:
    return ResolutionCandidate(
        source=source,
        source_id=doi,
        doi=doi,
        title=title,
        authors=("Smith, J.",),
        year=2020,
    )


def _evidence() -> ReferenceEvidence:
    return ReferenceEvidence(
        reference_index=7,
        raw_reference="Smith J. Example citation. 2020.",
        authors=("J. Smith",),
        year=2020,
    )


def _resolution() -> ScholarlyResolution:
    crossref_candidate = _candidate(
        source="crossref",
        doi="10.1000/a",
        title="Candidate A",
    )

    core_candidate = _candidate(
        source="core",
        doi="10.1000/b",
        title="Candidate B",
    )

    crossref = CrossrefAssessment(
        reference_index=7,
        status=CrossrefAssessmentStatus.NEEDS_CORE,
        selected_candidate=None,
        selected_score=None,
        ranked_candidates=(
            RankedCandidate(
                crossref_candidate,
                _score(0.70),
            ),
        ),
        reason="Ambiguous.",
    )

    core = CoreAssessment(
        reference_index=7,
        status=CoreAssessmentStatus.NEEDS_LLM,
        selected_candidate=None,
        selected_score=None,
        ranked_candidates=(
            RankedCandidate(
                core_candidate,
                _score(0.72),
            ),
        ),
        reason="Ambiguous.",
    )

    return ScholarlyResolution(
        reference_index=7,
        raw_reference=_evidence().raw_reference,
        status=ScholarlyResolutionStatus.NEEDS_LLM,
        selected_candidate=None,
        selected_score=None,
        crossref_assessment=crossref,
        core_assessment=core,
        reason="Needs LLM adjudication.",
    )


def test_build_llm_case_exposes_only_retrieved_candidates() -> None:
    case = build_llm_adjudication_case(
        _evidence(),
        _resolution(),
    )

    assert [
        candidate.candidate_id
        for candidate in case.candidates
    ] == [
        "crossref:1",
        "core:1",
    ]

    assert [
        candidate.candidate.doi
        for candidate in case.candidates
    ] == [
        "10.1000/a",
        "10.1000/b",
    ]


def test_llm_case_rejects_already_validated_reference() -> None:
    resolution = _resolution()

    validated = ScholarlyResolution(
        reference_index=resolution.reference_index,
        raw_reference=resolution.raw_reference,
        status=ScholarlyResolutionStatus.VALIDATED_CORE,
        selected_candidate=_candidate(
            source="core",
            doi="10.1000/b",
            title="Candidate B",
        ),
        selected_score=_score(0.9),
        crossref_assessment=resolution.crossref_assessment,
        core_assessment=resolution.core_assessment,
        reason="Validated.",
    )

    with pytest.raises(
        ValueError,
        match="only permitted",
    ):
        build_llm_adjudication_case(
            _evidence(),
            validated,
        )


def test_parse_select_candidate_accepts_supplied_candidate() -> None:
    case = build_llm_adjudication_case(
        _evidence(),
        _resolution(),
    )

    decision = parse_llm_decision(
        {
            "decision": "select_candidate",
            "candidate_id": "core:1",
            "confidence": 0.93,
            "evidence": [
                "Author and year agree.",
                "Candidate B best explains the citation.",
            ],
        },
        case,
    )

    assert decision.decision == (
        LLMDecisionType.SELECT_CANDIDATE
    )
    assert decision.candidate_id == "core:1"
    assert decision.confidence == 0.93


def test_parse_select_candidate_rejects_invented_candidate() -> None:
    case = build_llm_adjudication_case(
        _evidence(),
        _resolution(),
    )

    with pytest.raises(
        ValueError,
        match="not supplied",
    ):
        parse_llm_decision(
            {
                "decision": "select_candidate",
                "candidate_id": "crossref:99",
            },
            case,
        )


def test_parse_retry_search_requires_query() -> None:
    case = build_llm_adjudication_case(
        _evidence(),
        _resolution(),
    )

    with pytest.raises(
        ValueError,
        match="requires search_query",
    ):
        parse_llm_decision(
            {
                "decision": "retry_search",
            },
            case,
        )


def test_parse_retry_search_accepts_bounded_query() -> None:
    case = build_llm_adjudication_case(
        _evidence(),
        _resolution(),
    )

    decision = parse_llm_decision(
        {
            "decision": "retry_search",
            "search_query": (
                "Smith 2020 Example Journal volume 12"
            ),
            "confidence": 0.7,
            "evidence": [
                "Original citation is underspecified."
            ],
        },
        case,
    )

    assert decision.decision == (
        LLMDecisionType.RETRY_SEARCH
    )
    assert decision.search_query == (
        "Smith 2020 Example Journal volume 12"
    )


def test_parse_reject_all() -> None:
    case = build_llm_adjudication_case(
        _evidence(),
        _resolution(),
    )

    decision = parse_llm_decision(
        {
            "decision": "reject_all",
            "confidence": 0.88,
            "evidence": [
                "Neither candidate matches the citation."
            ],
        },
        case,
    )

    assert decision.decision == (
        LLMDecisionType.REJECT_ALL
    )


def test_parse_rejects_arbitrary_doi_field() -> None:
    case = build_llm_adjudication_case(
        _evidence(),
        _resolution(),
    )

    with pytest.raises(
        ValueError,
        match="unsupported fields",
    ):
        parse_llm_decision(
            {
                "decision": "select_candidate",
                "candidate_id": "core:1",
                "doi": "10.9999/hallucinated",
            },
            case,
        )


def test_parse_rejects_invalid_confidence() -> None:
    case = build_llm_adjudication_case(
        _evidence(),
        _resolution(),
    )

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        parse_llm_decision(
            {
                "decision": "reject_all",
                "confidence": 1.5,
            },
            case,
        )


def test_llm_case_deduplicates_same_doi_across_sources() -> None:
    evidence = _evidence()

    crossref_candidate = _candidate(
        source="crossref",
        doi="10.1000/shared",
        title="Same work",
    )

    core_candidate = _candidate(
        source="core",
        doi="10.1000/shared",
        title="Same work from CORE",
    )

    crossref = CrossrefAssessment(
        reference_index=7,
        status=CrossrefAssessmentStatus.NEEDS_CORE,
        selected_candidate=None,
        selected_score=None,
        ranked_candidates=(
            RankedCandidate(
                crossref_candidate,
                _score(0.70),
            ),
        ),
        reason="Ambiguous.",
    )

    core = CoreAssessment(
        reference_index=7,
        status=CoreAssessmentStatus.NEEDS_LLM,
        selected_candidate=None,
        selected_score=None,
        ranked_candidates=(
            RankedCandidate(
                core_candidate,
                _score(0.71),
            ),
        ),
        reason="Ambiguous.",
    )

    resolution = ScholarlyResolution(
        reference_index=7,
        raw_reference=evidence.raw_reference,
        status=ScholarlyResolutionStatus.NEEDS_LLM,
        selected_candidate=None,
        selected_score=None,
        crossref_assessment=crossref,
        core_assessment=core,
        reason="Needs LLM.",
    )

    case = build_llm_adjudication_case(
        evidence,
        resolution,
    )

    assert len(case.candidates) == 1
    assert (
        case.candidates[0].candidate.doi
        == "10.1000/shared"
    )





def test_parse_normalizes_single_evidence_string() -> None:
    case = build_llm_adjudication_case(
        _evidence(),
        _resolution(),
    )

    decision = parse_llm_decision(
        {
            "decision": "select_candidate",
            "candidate_id": "core:1",
            "confidence": 0.9,
            "evidence": "Author and year support this candidate.",
        },
        case,
    )

    assert decision.evidence == (
        "Author and year support this candidate.",
    )


def test_parse_still_rejects_structured_evidence_objects() -> None:
    case = build_llm_adjudication_case(
        _evidence(),
        _resolution(),
    )

    with pytest.raises(
        ValueError,
        match="only strings",
    ):
        parse_llm_decision(
            {
                "decision": "reject_all",
                "evidence": [
                    {
                        "reason": "unsupported"
                    }
                ],
            },
            case,
        )



def test_parse_normalizes_numeric_confidence_string() -> None:
    case = build_llm_adjudication_case(
        _evidence(),
        _resolution(),
    )

    decision = parse_llm_decision(
        {
            "decision": "select_candidate",
            "candidate_id": "core:1",
            "confidence": "0.91",
        },
        case,
    )

    assert decision.confidence == 0.91


def test_parse_normalizes_percentage_confidence_string() -> None:
    case = build_llm_adjudication_case(
        _evidence(),
        _resolution(),
    )

    decision = parse_llm_decision(
        {
            "decision": "reject_all",
            "confidence": "87%",
        },
        case,
    )

    assert decision.confidence == 0.87


def test_parse_rejects_qualitative_confidence_string() -> None:
    case = build_llm_adjudication_case(
        _evidence(),
        _resolution(),
    )

    with pytest.raises(
        ValueError,
        match="numeric",
    ):
        parse_llm_decision(
            {
                "decision": "reject_all",
                "confidence": "high",
            },
            case,
        )



def test_parse_normalizes_translated_evidence_key() -> None:
    case = build_llm_adjudication_case(
        _evidence(),
        _resolution(),
    )

    decision = parse_llm_decision(
        {
            "decision": "reject_all",
            "\u8bc1\u636e": [
                "Candidate metadata is insufficient."
            ],
        },
        case,
    )

    assert decision.evidence == (
        "Candidate metadata is insufficient.",
    )


def test_parse_rejects_duplicate_evidence_aliases() -> None:
    case = build_llm_adjudication_case(
        _evidence(),
        _resolution(),
    )

    with pytest.raises(
        ValueError,
        match="both evidence",
    ):
        parse_llm_decision(
            {
                "decision": "reject_all",
                "evidence": [
                    "English evidence."
                ],
                "\u8bc1\u636e": [
                    "Translated evidence."
                ],
            },
            case,
        )


def test_parse_still_rejects_translated_identity_fields() -> None:
    case = build_llm_adjudication_case(
        _evidence(),
        _resolution(),
    )

    with pytest.raises(
        ValueError,
        match="unsupported fields",
    ):
        parse_llm_decision(
            {
                "decision": "reject_all",
                "\u51b3\u5b9a": "reject_all",
            },
            case,
        )



def test_llm_case_omits_document_context_when_absent() -> None:
    case = build_llm_adjudication_case(
        _evidence(),
        _resolution(),
    )

    payload = case.to_dict()

    # Preserve the existing no-context contract for the ablation baseline.
    assert "document_contexts" not in payload


def test_llm_case_exposes_document_context_as_secondary_evidence() -> None:
    from tabulus.reference_resolution.reference_context import (
        ReferenceContext,
    )

    context = ReferenceContext(
        page_index=3,
        block_index=16,
        block_type="paragraph",
        bbox=(504, 640, 916, 820),
        citation_marker="7",
        citation_count=1,
        text=(
            "Smith et al.<sup>7</sup> reported the "
            "corresponding deposition process."
        ),
    )

    case = build_llm_adjudication_case(
        _evidence(),
        _resolution(),
        document_contexts=(
            context,
        ),
    )

    payload = case.to_dict()

    assert payload[
        "document_contexts"
    ] == [
        context.to_dict()
    ]

    assert any(
        "secondary evidence"
        in rule
        for rule in payload["rules"]
    )

    assert any(
        "must not be used to invent"
        in rule
        for rule in payload["rules"]
    )


def test_parse_normalizes_whitespace_around_evidence_key() -> None:
    case = build_llm_adjudication_case(
        _evidence(),
        _resolution(),
    )

    decision = parse_llm_decision(
        {
            "decision": "reject_all",
            " evidence ": [
                "Candidate metadata is insufficient."
            ],
        },
        case,
    )

    assert decision.evidence == (
        "Candidate metadata is insufficient.",
    )


def test_parse_still_rejects_whitespace_around_identity_fields() -> None:
    case = build_llm_adjudication_case(
        _evidence(),
        _resolution(),
    )

    with pytest.raises(
        ValueError,
        match="unsupported fields",
    ):
        parse_llm_decision(
            {
                "decision": "select_candidate",
                " candidate_id ": "core:1",
                "evidence": [
                    "Candidate appears supported."
                ],
            },
            case,
        )


def test_parse_rejects_duplicate_whitespace_evidence_fields() -> None:
    case = build_llm_adjudication_case(
        _evidence(),
        _resolution(),
    )

    with pytest.raises(
        ValueError,
        match="duplicate evidence",
    ):
        parse_llm_decision(
            {
                "decision": "reject_all",
                "evidence": [
                    "First explanation."
                ],
                " evidence": [
                    "Second explanation."
                ],
            },
            case,
        )
