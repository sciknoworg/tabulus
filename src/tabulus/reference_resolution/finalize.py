from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from tabulus.reference_resolution.assessment import (
    CrossrefAssessment,
    CrossrefAssessmentStatus,
    assess_crossref,
    rank_candidates,
)
from tabulus.reference_resolution.core import (
    CoreError,
    CoreSearchResponse,
)
from tabulus.reference_resolution.fallback import (
    CoreAssessment,
    CoreAssessmentStatus,
    ScholarlyResolution,
    ScholarlyResolutionStatus,
    assess_core,
)
from tabulus.reference_resolution.llm_client import (
    LLMAdjudicationResponse,
    OpenAICompatibleLLMError,
)
from tabulus.reference_resolution.llm_contract import (
    LLMAdjudicationCase,
    LLMDecisionType,
    build_llm_adjudication_case,
)
from tabulus.reference_resolution.models import (
    ReferenceEvidence,
    ReferenceResolution,
    ResolutionCandidate,
    ResolutionStatus,
)
from tabulus.reference_resolution.pipeline import (
    CrossrefRetrieval,
    is_non_atomic_reference,
)
from tabulus.reference_resolution.reference_context import (
    ReferenceContext,
)
from tabulus.reference_resolution.scoring import (
    normalize_doi,
)


class CrossrefSearchRetriever(Protocol):
    def search_bibliographic(
        self,
        reference_text: str,
    ) -> tuple[ResolutionCandidate, ...]:
        ...


class CoreSearchRetriever(Protocol):
    def search_works(
        self,
        reference_text: str,
    ) -> CoreSearchResponse:
        ...


class LLMAdjudicator(Protocol):
    def adjudicate(
        self,
        case: LLMAdjudicationCase,
    ) -> LLMAdjudicationResponse:
        ...


@dataclass(frozen=True)
class ReferenceResolutionTrace:
    """Complete Stage 6 provenance for one bibliography entry."""

    resolution: ReferenceResolution
    initial_scholarly_resolution: ScholarlyResolution
    first_llm_response: LLMAdjudicationResponse | None = None
    retry_query: str = ""
    retry_crossref_assessment: CrossrefAssessment | None = None
    retry_core_assessment: CoreAssessment | None = None
    second_llm_response: LLMAdjudicationResponse | None = None

    @property
    def retry_used(self) -> bool:
        return bool(self.retry_query)

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolution": self.resolution.to_dict(),
            "initial_scholarly_resolution": (
                self.initial_scholarly_resolution.to_dict()
            ),
            "first_llm_response": (
                self.first_llm_response.to_dict()
                if self.first_llm_response is not None
                else None
            ),
            "retry_used": self.retry_used,
            "retry_query": self.retry_query,
            "retry_crossref_assessment": (
                self.retry_crossref_assessment.to_dict()
                if self.retry_crossref_assessment is not None
                else None
            ),
            "retry_core_assessment": (
                self.retry_core_assessment.to_dict()
                if self.retry_core_assessment is not None
                else None
            ),
            "second_llm_response": (
                self.second_llm_response.to_dict()
                if self.second_llm_response is not None
                else None
            ),
        }


def _validated_resolution(
    evidence: ReferenceEvidence,
    candidate: ResolutionCandidate,
    *,
    confidence: float | None,
    reason: str,
) -> ReferenceResolution:
    doi = normalize_doi(
        candidate.doi
    )

    status = (
        ResolutionStatus.VALIDATED_WITH_DOI
        if doi
        else ResolutionStatus.VALIDATED_WITHOUT_DOI
    )

    return ReferenceResolution(
        reference_index=evidence.reference_index,
        raw_reference=evidence.raw_reference,
        status=status,
        canonical_doi=doi,
        canonical_title=candidate.title,
        canonical_authors=candidate.authors,
        canonical_year=candidate.year,
        canonical_venue=candidate.venue,
        source=candidate.source,
        confidence=confidence,
        reason=reason,
    )


def _rejected_resolution(
    evidence: ReferenceEvidence,
    *,
    reason: str,
) -> ReferenceResolution:
    return ReferenceResolution(
        reference_index=evidence.reference_index,
        raw_reference=evidence.raw_reference,
        status=ResolutionStatus.REJECTED,
        reason=reason,
    )


def _llm_selection_admissibility_error(
    case: LLMAdjudicationCase,
    response: LLMAdjudicationResponse,
) -> str:
    """Return why an LLM-selected candidate is unsafe, or ``""``.

    The model may adjudicate among supplied scholarly candidates, but it
    cannot lower Tabulus's minimum bibliographic-evidence requirement.

    A candidate is therefore admissible only when:

    * it has no explicit DOI contradiction with the source evidence; and
    * deterministic scoring found sufficient bibliographic evidence.

    ``strong_match`` is deliberately not required: otherwise the LLM would
    add no value beyond deterministic resolution.
    """

    selected = case.candidate_by_id(
        response.decision.candidate_id
    )

    if selected is None:
        return (
            "LLM selected a candidate outside the "
            "bounded candidate set."
        )

    score = selected.deterministic_score
    field_scores = dict(
        score.field_scores
    )

    if (
        "doi" in field_scores
        and field_scores["doi"] == 0.0
    ):
        return (
            "selected candidate explicitly conflicts "
            "with the source DOI."
        )

    if not score.sufficient_evidence:
        return (
            "selected candidate does not satisfy the "
            "deterministic minimum bibliographic-evidence "
            "requirement."
        )

    return ""


def _candidate_from_llm_selection(
    case: LLMAdjudicationCase,
    response: LLMAdjudicationResponse,
) -> ResolutionCandidate:
    selected = case.candidate_by_id(
        response.decision.candidate_id
    )

    # parse_llm_decision() already guarantees this, but retain the explicit
    # invariant here because finalization is the trust boundary.
    if selected is None:
        raise ValueError(
            "Validated LLM response refers to an unavailable candidate."
        )

    return selected.candidate


def _retry_crossref_assessment(
    evidence: ReferenceEvidence,
    query: str,
    client: CrossrefSearchRetriever,
) -> CrossrefAssessment:
    candidates = client.search_bibliographic(
        query
    )

    retrieval = CrossrefRetrieval(
        reference_index=evidence.reference_index,
        raw_reference=query,
        existing_doi="",
        existing_doi_checked=False,
        existing_doi_candidate=None,
        bibliographic_search_performed=True,
        bibliographic_candidates=candidates,
    )

    return assess_crossref(
        evidence,
        retrieval,
    )


def _combined_retry_resolution(
    evidence: ReferenceEvidence,
    initial: ScholarlyResolution,
    retry_crossref: CrossrefAssessment,
    retry_core: CoreAssessment,
) -> ScholarlyResolution:
    """Combine initial and retry candidates for final LLM adjudication."""

    crossref_candidates = [
        item.candidate
        for item in retry_crossref.ranked_candidates
    ] + [
        item.candidate
        for item in initial.crossref_assessment.ranked_candidates
    ]

    initial_core_ranked = (
        initial.core_assessment.ranked_candidates
        if initial.core_assessment is not None
        else ()
    )

    core_candidates = [
        item.candidate
        for item in retry_core.ranked_candidates
    ] + [
        item.candidate
        for item in initial_core_ranked
    ]

    combined_crossref = CrossrefAssessment(
        reference_index=evidence.reference_index,
        status=CrossrefAssessmentStatus.NEEDS_CORE,
        selected_candidate=None,
        selected_score=None,
        ranked_candidates=rank_candidates(
            evidence,
            crossref_candidates,
        ),
        reason=(
            "Combined initial and retry Crossref candidates "
            "for final LLM adjudication."
        ),
    )

    combined_core = CoreAssessment(
        reference_index=evidence.reference_index,
        status=CoreAssessmentStatus.NEEDS_LLM,
        selected_candidate=None,
        selected_score=None,
        ranked_candidates=rank_candidates(
            evidence,
            core_candidates,
        ),
        reason=(
            "Combined initial and retry CORE candidates "
            "for final LLM adjudication."
        ),
    )

    return ScholarlyResolution(
        reference_index=evidence.reference_index,
        raw_reference=evidence.raw_reference,
        status=ScholarlyResolutionStatus.NEEDS_LLM,
        selected_candidate=None,
        selected_score=None,
        crossref_assessment=combined_crossref,
        core_assessment=combined_core,
        reason=(
            "Deterministic validation remained insufficient "
            "after one bounded retry search."
        ),
    )


def finalize_reference_resolution(
    evidence: ReferenceEvidence,
    scholarly_resolution: ScholarlyResolution,
    *,
    crossref_client: CrossrefSearchRetriever,
    core_client: CoreSearchRetriever,
    llm_client: LLMAdjudicator,
    document_contexts: tuple[ReferenceContext, ...] = (),
) -> ReferenceResolutionTrace:
    """Produce the final Stage 6 decision for one bibliography entry.

    Deterministically validated Crossref/CORE matches are accepted directly.

    References requiring LLM adjudication receive one evidence-bounded model
    decision. The model may select a retrieved candidate, reject all
    candidates, or request exactly one improved scholarly search.

    A retry searches Crossref first and CORE only if Crossref remains
    insufficient. If deterministic validation still fails, one final LLM
    adjudication is allowed over the combined initial and retry candidate
    evidence. A second request to retry is rejected because the retry budget
    has been exhausted.

    Network/provider exceptions are intentionally not converted into
    ``rejected`` decisions. They propagate to the caller so operational failure
    cannot cause Stage 7 to delete a potentially valid scientific row.
    """

    if (
        evidence.reference_index
        != scholarly_resolution.reference_index
    ):
        raise ValueError(
            "Reference evidence and scholarly resolution "
            "must describe the same bibliography index."
        )

    # Stage 6 resolves exactly one scholarly work per bibliography entry.
    # A raw entry containing multiple complete citation-like publication
    # tails is therefore unsafe even if one retrieved candidate happens to
    # match part of it. Prefer a conservative rejection to an arbitrary DOI.
    if is_non_atomic_reference(
        evidence.raw_reference
    ):
        return ReferenceResolutionTrace(
            resolution=_rejected_resolution(
                evidence,
                reason=(
                    "Non-atomic bibliography entry contains multiple "
                    "complete citation-like publication records or "
                    "document-layout contamination; single-work "
                    "scholarly resolution is unsafe."
                ),
            ),
            initial_scholarly_resolution=(
                scholarly_resolution
            ),
        )

    # Already validated deterministically by Crossref or CORE.
    if scholarly_resolution.status in {
        ScholarlyResolutionStatus.VALIDATED_CROSSREF,
        ScholarlyResolutionStatus.VALIDATED_CORE,
    }:
        candidate = scholarly_resolution.selected_candidate

        if candidate is None:
            raise ValueError(
                "Validated scholarly resolution has no selected candidate."
            )

        confidence = (
            scholarly_resolution.selected_score.score
            if scholarly_resolution.selected_score is not None
            else None
        )

        return ReferenceResolutionTrace(
            resolution=_validated_resolution(
                evidence,
                candidate,
                confidence=confidence,
                reason=scholarly_resolution.reason,
            ),
            initial_scholarly_resolution=scholarly_resolution,
        )

    if (
        scholarly_resolution.status
        != ScholarlyResolutionStatus.NEEDS_LLM
    ):
        raise ValueError(
            "Unsupported scholarly resolution status."
        )

    # First LLM adjudication.
    first_case = build_llm_adjudication_case(
        evidence,
        scholarly_resolution,
        document_contexts=document_contexts,
    )

    try:
        first_response = llm_client.adjudicate(
            first_case
        )
    except OpenAICompatibleLLMError as error:
        raise OpenAICompatibleLLMError(
            "LLM adjudication failed for bibliography "
            f"index {evidence.reference_index}: {error}"
        ) from error

    first_decision = first_response.decision

    if (
        first_decision.decision
        == LLMDecisionType.SELECT_CANDIDATE
    ):
        admissibility_error = (
            _llm_selection_admissibility_error(
                first_case,
                first_response,
            )
        )

        if admissibility_error:
            return ReferenceResolutionTrace(
                resolution=_rejected_resolution(
                    evidence,
                    reason=(
                        "LLM-selected candidate failed the "
                        "final deterministic admissibility gate: "
                        f"{admissibility_error}"
                    ),
                ),
                initial_scholarly_resolution=(
                    scholarly_resolution
                ),
                first_llm_response=first_response,
            )

        candidate = _candidate_from_llm_selection(
            first_case,
            first_response,
        )

        return ReferenceResolutionTrace(
            resolution=_validated_resolution(
                evidence,
                candidate,
                confidence=first_decision.confidence,
                reason=(
                    "Candidate selected by evidence-bounded "
                    "LLM adjudication and passed the final "
                    "deterministic admissibility gate."
                ),
            ),
            initial_scholarly_resolution=scholarly_resolution,
            first_llm_response=first_response,
        )

    if (
        first_decision.decision
        == LLMDecisionType.REJECT_ALL
    ):
        return ReferenceResolutionTrace(
            resolution=_rejected_resolution(
                evidence,
                reason=(
                    "Crossref and CORE did not provide a "
                    "deterministically validated match, and the "
                    "evidence-bounded LLM rejected all candidates."
                ),
            ),
            initial_scholarly_resolution=scholarly_resolution,
            first_llm_response=first_response,
        )

    # Exactly one bounded retry.
    retry_query = first_decision.search_query

    retry_crossref = _retry_crossref_assessment(
        evidence,
        retry_query,
        crossref_client,
    )

    if not retry_crossref.needs_core:
        candidate = retry_crossref.selected_candidate

        if candidate is None:
            raise ValueError(
                "Validated retry Crossref assessment "
                "has no selected candidate."
            )

        confidence = (
            retry_crossref.selected_score.score
            if retry_crossref.selected_score is not None
            else None
        )

        return ReferenceResolutionTrace(
            resolution=_validated_resolution(
                evidence,
                candidate,
                confidence=confidence,
                reason=(
                    "Reference was validated by Crossref "
                    "after one LLM-generated retry search."
                ),
            ),
            initial_scholarly_resolution=scholarly_resolution,
            first_llm_response=first_response,
            retry_query=retry_query,
            retry_crossref_assessment=retry_crossref,
        )

    try:
        retry_core_response = core_client.search_works(
            retry_query
        )
    except CoreError as error:
        raise CoreError(
            "CORE retry lookup failed for bibliography "
            f"index {evidence.reference_index}: {error}"
        ) from error

    retry_core = assess_core(
        evidence,
        retry_core_response,
    )

    if not retry_core.needs_llm:
        candidate = retry_core.selected_candidate

        if candidate is None:
            raise ValueError(
                "Validated retry CORE assessment "
                "has no selected candidate."
            )

        confidence = (
            retry_core.selected_score.score
            if retry_core.selected_score is not None
            else None
        )

        return ReferenceResolutionTrace(
            resolution=_validated_resolution(
                evidence,
                candidate,
                confidence=confidence,
                reason=(
                    "Reference was validated by CORE "
                    "after one LLM-generated retry search."
                ),
            ),
            initial_scholarly_resolution=scholarly_resolution,
            first_llm_response=first_response,
            retry_query=retry_query,
            retry_crossref_assessment=retry_crossref,
            retry_core_assessment=retry_core,
        )

    # Deterministic retry also failed. Give the model one final adjudication
    # over both the original and retry candidate evidence.
    combined = _combined_retry_resolution(
        evidence,
        scholarly_resolution,
        retry_crossref,
        retry_core,
    )

    second_case = build_llm_adjudication_case(
        evidence,
        combined,
        document_contexts=document_contexts,
    )

    try:
        second_response = llm_client.adjudicate(
            second_case
        )
    except OpenAICompatibleLLMError as error:
        raise OpenAICompatibleLLMError(
            "Final LLM adjudication failed for bibliography "
            f"index {evidence.reference_index}: {error}"
        ) from error

    second_decision = second_response.decision

    if (
        second_decision.decision
        == LLMDecisionType.SELECT_CANDIDATE
    ):
        admissibility_error = (
            _llm_selection_admissibility_error(
                second_case,
                second_response,
            )
        )

        if admissibility_error:
            return ReferenceResolutionTrace(
                resolution=_rejected_resolution(
                    evidence,
                    reason=(
                        "Final LLM-selected candidate failed the "
                        "deterministic admissibility gate: "
                        f"{admissibility_error}"
                    ),
                ),
                initial_scholarly_resolution=scholarly_resolution,
                first_llm_response=first_response,
                retry_query=retry_query,
                retry_crossref_assessment=retry_crossref,
                retry_core_assessment=retry_core,
                second_llm_response=second_response,
            )

        candidate = _candidate_from_llm_selection(
            second_case,
            second_response,
        )

        return ReferenceResolutionTrace(
            resolution=_validated_resolution(
                evidence,
                candidate,
                confidence=second_decision.confidence,
                reason=(
                    "Candidate selected by final evidence-bounded "
                    "LLM adjudication after one scholarly-search "
                    "retry and passed the deterministic "
                    "admissibility gate."
                ),
            ),
            initial_scholarly_resolution=scholarly_resolution,
            first_llm_response=first_response,
            retry_query=retry_query,
            retry_crossref_assessment=retry_crossref,
            retry_core_assessment=retry_core,
            second_llm_response=second_response,
        )

    if (
        second_decision.decision
        == LLMDecisionType.REJECT_ALL
    ):
        reason = (
            "No candidate could be validated after Crossref, CORE, "
            "one bounded search retry, and final LLM adjudication."
        )
    else:
        reason = (
            "No candidate could be validated after the single permitted "
            "search retry; the LLM requested another retry after the "
            "retry budget had been exhausted."
        )

    return ReferenceResolutionTrace(
        resolution=_rejected_resolution(
            evidence,
            reason=reason,
        ),
        initial_scholarly_resolution=scholarly_resolution,
        first_llm_response=first_response,
        retry_query=retry_query,
        retry_crossref_assessment=retry_crossref,
        retry_core_assessment=retry_core,
        second_llm_response=second_response,
    )
