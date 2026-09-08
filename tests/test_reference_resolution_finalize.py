from __future__ import annotations

from tabulus.reference_resolution import (
    CandidateScore,
    CoreAssessment,
    CoreAssessmentStatus,
    CoreRateLimit,
    CoreSearchResponse,
    CrossrefAssessment,
    CrossrefAssessmentStatus,
    LLMAdjudicationResponse,
    LLMDecision,
    LLMDecisionType,
    LLMUsage,
    RankedCandidate,
    ReferenceEvidence,
    ResolutionCandidate,
    ResolutionStatus,
    ScholarlyResolution,
    ScholarlyResolutionStatus,
    finalize_reference_resolution,
)


def _evidence() -> ReferenceEvidence:
    return ReferenceEvidence(
        reference_index=7,
        raw_reference="Smith J. Example work. 2020.",
        title="Example work",
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
    title: str = "Example work",
    year: int = 2020,
    volume: str = "12",
    pages: str = "100-110",
) -> ResolutionCandidate:
    return ResolutionCandidate(
        source=source,
        source_id=doi or f"{source}-work",
        doi=doi,
        title=title,
        authors=("Smith, J.",),
        year=year,
        venue="Example Journal",
        volume=volume,
        pages=pages,
    )


def _score(
    value: float = 0.70,
    *,
    strong: bool = False,
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
        strong_match=strong,
    )


def _needs_llm() -> ScholarlyResolution:
    crossref = CrossrefAssessment(
        reference_index=7,
        status=CrossrefAssessmentStatus.NEEDS_CORE,
        selected_candidate=None,
        selected_score=None,
        ranked_candidates=(
            RankedCandidate(
                _candidate(
                    source="crossref",
                    doi="10.1000/weak-crossref",
                    title="Possible example work",
                    year=2020,
                ),
                _score(0.70),
            ),
        ),
        reason="Weak Crossref evidence.",
    )

    core = CoreAssessment(
        reference_index=7,
        status=CoreAssessmentStatus.NEEDS_LLM,
        selected_candidate=None,
        selected_score=None,
        ranked_candidates=(
            RankedCandidate(
                _candidate(
                    source="core",
                    doi="10.1000/weak-core",
                    title="Another possible work",
                    year=2020,
                ),
                _score(0.69),
            ),
        ),
        reason="Weak CORE evidence.",
    )

    return ScholarlyResolution(
        reference_index=7,
        raw_reference=_evidence().raw_reference,
        status=ScholarlyResolutionStatus.NEEDS_LLM,
        selected_candidate=None,
        selected_score=None,
        crossref_assessment=crossref,
        core_assessment=core,
        reason="Needs LLM.",
    )


def _validated_crossref(
    *,
    doi: str = "10.1000/example",
) -> ScholarlyResolution:
    candidate = _candidate(
        source="crossref",
        doi=doi,
    )

    score = _score(
        1.0,
        strong=True,
    )

    crossref = CrossrefAssessment(
        reference_index=7,
        status=(
            CrossrefAssessmentStatus
            .VALIDATED_SEARCH_MATCH
        ),
        selected_candidate=candidate,
        selected_score=score,
        ranked_candidates=(
            RankedCandidate(
                candidate,
                score,
            ),
        ),
        reason="Validated.",
    )

    return ScholarlyResolution(
        reference_index=7,
        raw_reference=_evidence().raw_reference,
        status=(
            ScholarlyResolutionStatus
            .VALIDATED_CROSSREF
        ),
        selected_candidate=candidate,
        selected_score=score,
        crossref_assessment=crossref,
        core_assessment=None,
        reason="Crossref validated.",
    )


def _validated_core_without_doi() -> ScholarlyResolution:
    candidate = _candidate(
        source="core",
        doi="",
    )

    score = _score(
        1.0,
        strong=True,
    )

    crossref = CrossrefAssessment(
        reference_index=7,
        status=CrossrefAssessmentStatus.NEEDS_CORE,
        selected_candidate=None,
        selected_score=None,
        ranked_candidates=(),
        reason="No Crossref result.",
    )

    core = CoreAssessment(
        reference_index=7,
        status=(
            CoreAssessmentStatus
            .VALIDATED_CORE_MATCH
        ),
        selected_candidate=candidate,
        selected_score=score,
        ranked_candidates=(
            RankedCandidate(
                candidate,
                score,
            ),
        ),
        reason="CORE validated.",
    )

    return ScholarlyResolution(
        reference_index=7,
        raw_reference=_evidence().raw_reference,
        status=ScholarlyResolutionStatus.VALIDATED_CORE,
        selected_candidate=candidate,
        selected_score=score,
        crossref_assessment=crossref,
        core_assessment=core,
        reason="CORE validated.",
    )


def _llm_response(
    decision: LLMDecision,
) -> LLMAdjudicationResponse:
    return LLMAdjudicationResponse(
        decision=decision,
        model="qwen3.6-35b-a3b",
        response_id="test",
        finish_reason="stop",
        usage=LLMUsage(
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
        ),
    )


class _FakeLLM:
    def __init__(
        self,
        *decisions: LLMDecision,
    ):
        self.responses = [
            _llm_response(decision)
            for decision in decisions
        ]
        self.calls = []

    def adjudicate(
        self,
        case,
    ):
        self.calls.append(case)

        if not self.responses:
            raise AssertionError(
                "Unexpected extra LLM adjudication."
            )

        return self.responses.pop(0)


class _FakeCrossref:
    def __init__(
        self,
        candidates=(),
    ):
        self.candidates = tuple(candidates)
        self.calls = []

    def search_bibliographic(
        self,
        reference_text,
    ):
        self.calls.append(reference_text)
        return self.candidates


class _FakeCore:
    def __init__(
        self,
        candidates=(),
    ):
        self.candidates = tuple(candidates)
        self.calls = []

    def search_works(
        self,
        reference_text,
    ):
        self.calls.append(reference_text)

        return CoreSearchResponse(
            candidates=self.candidates,
            total_hits=len(self.candidates),
            search_id="test",
            limit=5,
            offset=0,
            rate_limit=CoreRateLimit(),
        )


def test_deterministic_crossref_validation_skips_llm() -> None:
    llm = _FakeLLM()

    result = finalize_reference_resolution(
        _evidence(),
        _validated_crossref(),
        crossref_client=_FakeCrossref(),
        core_client=_FakeCore(),
        llm_client=llm,
    )

    assert result.resolution.status == (
        ResolutionStatus.VALIDATED_WITH_DOI
    )
    assert (
        result.resolution.canonical_doi
        == "10.1000/example"
    )
    assert llm.calls == []
    assert result.retry_used is False


def test_deterministic_core_can_validate_without_doi() -> None:
    result = finalize_reference_resolution(
        _evidence(),
        _validated_core_without_doi(),
        crossref_client=_FakeCrossref(),
        core_client=_FakeCore(),
        llm_client=_FakeLLM(),
    )

    assert result.resolution.status == (
        ResolutionStatus.VALIDATED_WITHOUT_DOI
    )
    assert result.resolution.canonical_doi == ""
    assert result.resolution.source == "core"


def test_first_llm_can_select_retrieved_candidate() -> None:
    llm = _FakeLLM(
        LLMDecision(
            decision=LLMDecisionType.SELECT_CANDIDATE,
            candidate_id="core:1",
            confidence=0.91,
            evidence=("Best supported candidate.",),
        )
    )

    result = finalize_reference_resolution(
        _evidence(),
        _needs_llm(),
        crossref_client=_FakeCrossref(),
        core_client=_FakeCore(),
        llm_client=llm,
    )

    assert result.resolution.status == (
        ResolutionStatus.VALIDATED_WITH_DOI
    )
    assert result.resolution.source == "core"
    assert result.resolution.confidence == 0.91
    assert result.retry_used is False


def test_first_llm_reject_all_produces_rejected_status() -> None:
    llm = _FakeLLM(
        LLMDecision(
            decision=LLMDecisionType.REJECT_ALL,
            confidence=0.9,
            evidence=("No supplied candidate is supported.",),
        )
    )

    result = finalize_reference_resolution(
        _evidence(),
        _needs_llm(),
        crossref_client=_FakeCrossref(),
        core_client=_FakeCore(),
        llm_client=llm,
    )

    assert result.resolution.status == (
        ResolutionStatus.REJECTED
    )
    assert result.retry_used is False


def test_retry_can_be_validated_deterministically_by_crossref() -> None:
    query = "Smith Example work 2020 Example Journal 12 100-110"

    llm = _FakeLLM(
        LLMDecision(
            decision=LLMDecisionType.RETRY_SEARCH,
            search_query=query,
            confidence=0.8,
            evidence=("Use complete bibliographic locators.",),
        )
    )

    crossref = _FakeCrossref(
        candidates=(
            _candidate(
                source="crossref",
                doi="10.1000/recovered",
            ),
        )
    )

    core = _FakeCore()

    result = finalize_reference_resolution(
        _evidence(),
        _needs_llm(),
        crossref_client=crossref,
        core_client=core,
        llm_client=llm,
    )

    assert result.resolution.status == (
        ResolutionStatus.VALIDATED_WITH_DOI
    )
    assert (
        result.resolution.canonical_doi
        == "10.1000/recovered"
    )
    assert result.retry_query == query
    assert crossref.calls == [query]
    assert core.calls == []
    assert len(llm.calls) == 1


def test_retry_falls_through_to_core() -> None:
    query = "Smith Example work 2020"

    llm = _FakeLLM(
        LLMDecision(
            decision=LLMDecisionType.RETRY_SEARCH,
            search_query=query,
        )
    )

    core = _FakeCore(
        candidates=(
            _candidate(
                source="core",
                doi="",
            ),
        )
    )

    result = finalize_reference_resolution(
        _evidence(),
        _needs_llm(),
        crossref_client=_FakeCrossref(),
        core_client=core,
        llm_client=llm,
    )

    assert result.resolution.status == (
        ResolutionStatus.VALIDATED_WITHOUT_DOI
    )
    assert result.resolution.source == "core"
    assert core.calls == [query]


def test_second_llm_can_select_candidate_after_retry() -> None:
    query = "Smith 2020 possible example"

    llm = _FakeLLM(
        LLMDecision(
            decision=LLMDecisionType.RETRY_SEARCH,
            search_query=query,
        ),
        LLMDecision(
            decision=LLMDecisionType.SELECT_CANDIDATE,
            candidate_id="core:1",
            confidence=0.82,
        ),
    )

    retry_core_candidate = _candidate(
        source="core",
        doi="10.1000/retry-core",
        title="Possible example publication",
        year=2020,
        volume="99",
        pages="999-1009",
    )

    result = finalize_reference_resolution(
        _evidence(),
        _needs_llm(),
        crossref_client=_FakeCrossref(),
        core_client=_FakeCore(
            candidates=(
                retry_core_candidate,
            )
        ),
        llm_client=llm,
    )

    assert result.resolution.status == (
        ResolutionStatus.VALIDATED_WITH_DOI
    )
    assert result.second_llm_response is not None
    assert len(llm.calls) == 2


def test_second_retry_request_is_rejected_after_budget_exhausted() -> None:
    first_query = "Smith 2020 Example Journal"

    llm = _FakeLLM(
        LLMDecision(
            decision=LLMDecisionType.RETRY_SEARCH,
            search_query=first_query,
        ),
        LLMDecision(
            decision=LLMDecisionType.RETRY_SEARCH,
            search_query="try something else again",
        ),
    )

    result = finalize_reference_resolution(
        _evidence(),
        _needs_llm(),
        crossref_client=_FakeCrossref(),
        core_client=_FakeCore(),
        llm_client=llm,
    )

    assert result.resolution.status == (
        ResolutionStatus.REJECTED
    )
    assert result.retry_used is True
    assert len(llm.calls) == 2
    assert "retry" in result.resolution.reason.casefold()



def test_llm_failure_reports_bibliography_index() -> None:
    import pytest

    from tabulus.reference_resolution import (
        OpenAICompatibleLLMError,
    )

    class FailingLLM:
        def adjudicate(
            self,
            case,
        ):
            raise OpenAICompatibleLLMError(
                "invalid model output"
            )

    with pytest.raises(
        OpenAICompatibleLLMError,
        match="bibliography index 7",
    ):
        finalize_reference_resolution(
            _evidence(),
            _needs_llm(),
            crossref_client=_FakeCrossref(),
            core_client=_FakeCore(),
            llm_client=FailingLLM(),
        )



def test_second_llm_failure_reports_bibliography_index() -> None:
    import pytest

    from tabulus.reference_resolution import (
        OpenAICompatibleLLMError,
    )

    query = "Smith Example work 2020"

    class FailingSecondLLM:
        def __init__(self):
            self.calls = 0

        def adjudicate(
            self,
            case,
        ):
            self.calls += 1

            if self.calls == 1:
                return _llm_response(
                    LLMDecision(
                        decision=LLMDecisionType.RETRY_SEARCH,
                        search_query=query,
                    )
                )

            raise OpenAICompatibleLLMError(
                "invalid final model output"
            )

    with pytest.raises(
        OpenAICompatibleLLMError,
        match="bibliography index 7",
    ):
        finalize_reference_resolution(
            _evidence(),
            _needs_llm(),
            crossref_client=_FakeCrossref(),
            core_client=_FakeCore(),
            llm_client=FailingSecondLLM(),
        )



def test_finalize_passes_document_context_to_llm() -> None:
    from types import SimpleNamespace

    from tabulus.reference_resolution.reference_context import (
        ReferenceContext,
    )

    context = ReferenceContext(
        page_index=4,
        block_index=2,
        block_type="paragraph",
        bbox=(10, 20, 30, 40),
        citation_marker="7",
        citation_count=1,
        text=(
            "The citing article discusses the specific "
            "process attributed to reference 7."
        ),
    )

    class CapturingLLM:
        def __init__(self):
            self.cases = []

        def adjudicate(
            self,
            case,
        ):
            self.cases.append(
                case
            )

            return SimpleNamespace(
                decision=LLMDecision(
                    decision=(
                        LLMDecisionType.REJECT_ALL
                    ),
                    evidence=(
                        "No candidate is sufficiently supported.",
                    ),
                )
            )

    llm = CapturingLLM()

    trace = finalize_reference_resolution(
        _evidence(),
        _needs_llm(),
        crossref_client=object(),
        core_client=object(),
        llm_client=llm,
        document_contexts=(
            context,
        ),
    )

    assert trace.resolution.status == (
        ResolutionStatus.REJECTED
    )

    assert len(
        llm.cases
    ) == 1

    assert (
        llm.cases[0].document_contexts
        == (
            context,
        )
    )

    payload = llm.cases[0].to_dict()

    assert payload[
        "document_contexts"
    ][0]["text"] == context.text



def test_finalize_rejects_non_atomic_bibliography_entry() -> None:
    evidence = ReferenceEvidence(
        reference_index=7,
        raw_reference=(
            "B. Y. Maa and P. D. Dapkus, Appl. Phys. Lett. "
            "58, 1762 (1991). "
            "B. Y. Maa and P. D. Dapkus, Appl. Phys. Lett. "
            "58, 2261 (1991)."
        ),
        authors=(
            "B. Y. Maa",
            "P. D. Dapkus",
        ),
        year=1991,
        venue="Applied Physics Letters",
        volume="58",
    )

    trace = finalize_reference_resolution(
        evidence,
        _validated_crossref(),
        crossref_client=object(),
        core_client=object(),
        llm_client=object(),
    )

    assert trace.resolution.status == (
        ResolutionStatus.REJECTED
    )

    assert (
        "Non-atomic bibliography entry"
        in trace.resolution.reason
    )



def test_llm_selection_requires_sufficient_deterministic_evidence() -> None:
    from types import SimpleNamespace

    weak_candidate = ResolutionCandidate(
        source="crossref",
        source_id="10.1000/editorial",
        doi="10.1000/editorial",
        title="Editorial Board",
        authors=(),
        year=2020,
        venue="Example Journal",
        volume="12",
        pages="iii",
    )

    weak_score = CandidateScore(
        score=0.81,
        comparable_weight=0.37,
        field_scores=(
            ("year", 1.0),
            ("venue", 1.0),
            ("volume", 1.0),
            ("pages", 0.0),
        ),
        sufficient_evidence=False,
        strong_match=False,
    )

    crossref = CrossrefAssessment(
        reference_index=7,
        status=CrossrefAssessmentStatus.NEEDS_CORE,
        selected_candidate=None,
        selected_score=None,
        ranked_candidates=(
            RankedCandidate(
                weak_candidate,
                weak_score,
            ),
        ),
        reason="Insufficient evidence.",
    )

    core = CoreAssessment(
        reference_index=7,
        status=CoreAssessmentStatus.NEEDS_LLM,
        selected_candidate=None,
        selected_score=None,
        ranked_candidates=(),
        reason="No stronger CORE candidate.",
    )

    scholarly = ScholarlyResolution(
        reference_index=7,
        raw_reference=_evidence().raw_reference,
        status=ScholarlyResolutionStatus.NEEDS_LLM,
        selected_candidate=None,
        selected_score=None,
        crossref_assessment=crossref,
        core_assessment=core,
        reason="Needs LLM.",
    )

    class SelectingLLM:
        def adjudicate(
            self,
            case,
        ):
            return SimpleNamespace(
                decision=LLMDecision(
                    decision=(
                        LLMDecisionType.SELECT_CANDIDATE
                    ),
                    candidate_id="crossref:1",
                    confidence=0.95,
                    evidence=(
                        "Candidate appears plausible.",
                    ),
                )
            )

    trace = finalize_reference_resolution(
        _evidence(),
        scholarly,
        crossref_client=object(),
        core_client=object(),
        llm_client=SelectingLLM(),
    )

    assert trace.resolution.status == (
        ResolutionStatus.REJECTED
    )

    assert (
        "minimum bibliographic-evidence"
        in trace.resolution.reason
    )


def test_llm_selection_cannot_override_explicit_doi_conflict() -> None:
    from types import SimpleNamespace

    evidence = ReferenceEvidence(
        reference_index=7,
        raw_reference="Smith J. Example work. 2020.",
        doi="10.1000/source",
        title="Example work",
        authors=("J. Smith",),
        year=2020,
    )

    candidate = ResolutionCandidate(
        source="crossref",
        source_id="10.1000/different",
        doi="10.1000/different",
        title="Example work",
        authors=("Smith, J.",),
        year=2020,
    )

    score = CandidateScore(
        score=0.50,
        comparable_weight=1.75,
        field_scores=(
            ("doi", 0.0),
            ("title", 1.0),
            ("authors", 1.0),
            ("year", 1.0),
        ),
        sufficient_evidence=True,
        strong_match=False,
    )

    crossref = CrossrefAssessment(
        reference_index=7,
        status=CrossrefAssessmentStatus.NEEDS_CORE,
        selected_candidate=None,
        selected_score=None,
        ranked_candidates=(
            RankedCandidate(
                candidate,
                score,
            ),
        ),
        reason="DOI conflict.",
    )

    core = CoreAssessment(
        reference_index=7,
        status=CoreAssessmentStatus.NEEDS_LLM,
        selected_candidate=None,
        selected_score=None,
        ranked_candidates=(),
        reason="No alternative.",
    )

    scholarly = ScholarlyResolution(
        reference_index=7,
        raw_reference=evidence.raw_reference,
        status=ScholarlyResolutionStatus.NEEDS_LLM,
        selected_candidate=None,
        selected_score=None,
        crossref_assessment=crossref,
        core_assessment=core,
        reason="Needs LLM.",
    )

    class SelectingLLM:
        def adjudicate(
            self,
            case,
        ):
            return SimpleNamespace(
                decision=LLMDecision(
                    decision=(
                        LLMDecisionType.SELECT_CANDIDATE
                    ),
                    candidate_id="crossref:1",
                    confidence=0.99,
                    evidence=(
                        "Other metadata agrees.",
                    ),
                )
            )

    trace = finalize_reference_resolution(
        evidence,
        scholarly,
        crossref_client=object(),
        core_client=object(),
        llm_client=SelectingLLM(),
    )

    assert trace.resolution.status == (
        ResolutionStatus.REJECTED
    )

    assert (
        "explicitly conflicts with the source DOI"
        in trace.resolution.reason
    )
