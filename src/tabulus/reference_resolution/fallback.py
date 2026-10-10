from __future__ import annotations

import logging

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Protocol

from tabulus.reference_resolution.assessment import (
    DEFAULT_CANDIDATE_MARGIN,
    CrossrefAssessment,
    CrossrefAssessmentStatus,
    RankedCandidate,
    assess_crossref,
    assess_crossref_retrievals,
    rank_candidates,
)
from tabulus.reference_resolution.core import (
    CoreError,
    CoreSearchResponse,
)
from tabulus.reference_resolution.models import (
    CandidateScore,
    ReferenceEvidence,
    ResolutionCandidate,
)
from tabulus.reference_resolution.pipeline import (
    CrossrefRetrieval,
    is_publisher_boilerplate_reference,
)


class CoreRetriever(Protocol):
    """Minimal CORE interface required by the fallback resolver."""

    def search_works(
        self,
        reference_text: str,
    ) -> CoreSearchResponse:
        ...



class CrossrefSearchRetriever(Protocol):
    """Crossref search interface used only for operational CORE recovery."""

    def search_bibliographic(
        self,
        reference_text: str,
    ) -> tuple[ResolutionCandidate, ...]:
        ...


class CoreAssessmentStatus(str, Enum):
    """Outcome of deterministic CORE candidate validation."""

    VALIDATED_CORE_MATCH = "validated_core_match"
    NEEDS_LLM = "needs_llm"


@dataclass(frozen=True)
class CoreAssessment:
    """Deterministic assessment of one CORE search response."""

    reference_index: int
    status: CoreAssessmentStatus
    selected_candidate: ResolutionCandidate | None
    selected_score: CandidateScore | None
    ranked_candidates: tuple[RankedCandidate, ...]
    reason: str

    @property
    def needs_llm(self) -> bool:
        return (
            self.status
            == CoreAssessmentStatus.NEEDS_LLM
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_index": self.reference_index,
            "status": self.status.value,
            "selected_candidate": (
                self.selected_candidate.to_dict()
                if self.selected_candidate is not None
                else None
            ),
            "selected_score": (
                self.selected_score.to_dict()
                if self.selected_score is not None
                else None
            ),
            "ranked_candidates": [
                candidate.to_dict()
                for candidate in self.ranked_candidates
            ],
            "reason": self.reason,
        }


class ScholarlyResolutionStatus(str, Enum):
    """Current two-source Stage 6 resolution state."""

    VALIDATED_CROSSREF = "validated_crossref"
    VALIDATED_CORE = "validated_core"
    NEEDS_LLM = "needs_llm"


@dataclass(frozen=True)
class ScholarlyResolution:
    """Combined Crossref → CORE resolution evidence for one reference."""

    reference_index: int
    raw_reference: str
    status: ScholarlyResolutionStatus
    selected_candidate: ResolutionCandidate | None
    selected_score: CandidateScore | None
    crossref_assessment: CrossrefAssessment
    core_assessment: CoreAssessment | None
    reason: str

    @property
    def selected_source(self) -> str:
        if self.selected_candidate is None:
            return ""
        return self.selected_candidate.source

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_index": self.reference_index,
            "raw_reference": self.raw_reference,
            "status": self.status.value,
            "selected_source": self.selected_source,
            "selected_candidate": (
                self.selected_candidate.to_dict()
                if self.selected_candidate is not None
                else None
            ),
            "selected_score": (
                self.selected_score.to_dict()
                if self.selected_score is not None
                else None
            ),
            "crossref_assessment": (
                self.crossref_assessment.to_dict()
            ),
            "core_assessment": (
                self.core_assessment.to_dict()
                if self.core_assessment is not None
                else None
            ),
            "reason": self.reason,
        }


def assess_core(
    evidence: ReferenceEvidence,
    response: CoreSearchResponse,
    *,
    candidate_margin: float = DEFAULT_CANDIDATE_MARGIN,
) -> CoreAssessment:
    """Validate CORE candidates using the same deterministic evidence policy.

    CORE is a fallback discovery source, not an authority that is trusted
    automatically. The best candidate must satisfy the strong-match criteria
    and be sufficiently distinct from any plausible runner-up.

    If deterministic evidence remains weak or ambiguous, the reference is
    explicitly routed to the later LLM adjudication step.
    """

    if candidate_margin < 0:
        raise ValueError(
            "candidate_margin must be non-negative."
        )

    ranked = rank_candidates(
        evidence,
        response.candidates,
    )

    if not ranked:
        return CoreAssessment(
            reference_index=evidence.reference_index,
            status=CoreAssessmentStatus.NEEDS_LLM,
            selected_candidate=None,
            selected_score=None,
            ranked_candidates=(),
            reason=(
                "CORE produced no scholarly-work candidates."
            ),
        )

    top = ranked[0]

    if not top.score.strong_match:
        return CoreAssessment(
            reference_index=evidence.reference_index,
            status=CoreAssessmentStatus.NEEDS_LLM,
            selected_candidate=None,
            selected_score=None,
            ranked_candidates=ranked,
            reason=(
                "The highest-ranked CORE candidate did not "
                "satisfy the strong-match criteria."
            ),
        )

    if len(ranked) > 1:
        runner_up = ranked[1]

        if runner_up.score.sufficient_evidence:
            margin = (
                top.score.score
                - runner_up.score.score
            )

            if margin < candidate_margin:
                return CoreAssessment(
                    reference_index=evidence.reference_index,
                    status=CoreAssessmentStatus.NEEDS_LLM,
                    selected_candidate=None,
                    selected_score=None,
                    ranked_candidates=ranked,
                    reason=(
                        "Multiple CORE candidates are too "
                        "close to distinguish reliably."
                    ),
                )

    return CoreAssessment(
        reference_index=evidence.reference_index,
        status=CoreAssessmentStatus.VALIDATED_CORE_MATCH,
        selected_candidate=top.candidate,
        selected_score=top.score,
        ranked_candidates=ranked,
        reason=(
            "The highest-ranked CORE candidate satisfied "
            "the strong-match criteria and was sufficiently "
            "distinct from alternatives."
        ),
    )


LOGGER = logging.getLogger(__name__)


def _recover_after_core_failure(
    evidence: ReferenceEvidence,
    initial_crossref: CrossrefAssessment,
    crossref_client: CrossrefSearchRetriever,
    *,
    candidate_margin: float,
) -> ScholarlyResolution | None:
    """Try one structured Crossref search after persistent CORE failure.

    The recovery query is derived only from structured source evidence.
    Candidates remain subject to the unchanged deterministic scorer and,
    when necessary, the existing evidence-bounded LLM admissibility gate.
    """

    title = str(evidence.title or "").strip()

    if not title:
        return None

    parts = [title]

    if evidence.authors:
        parts.append(
            str(evidence.authors[0]).strip()
        )

    if evidence.year is not None:
        parts.append(
            str(evidence.year)
        )

    # A title alone is deliberately insufficient to trigger an additional
    # provider search after an operational failure.
    if len(parts) < 2:
        return None

    query = " ".join(
        part for part in parts if part
    )

    candidates = crossref_client.search_bibliographic(
        query
    )

    retrieval = CrossrefRetrieval(
        reference_index=evidence.reference_index,
        raw_reference=query,
        existing_doi="",
        existing_doi_checked=False,
        existing_doi_candidate=None,
        bibliographic_search_performed=True,
        bibliographic_candidates=tuple(
            candidates
        ),
    )

    recovery = assess_crossref(
        evidence,
        retrieval,
        candidate_margin=candidate_margin,
    )

    if not recovery.needs_core:
        return ScholarlyResolution(
            reference_index=evidence.reference_index,
            raw_reference=evidence.raw_reference,
            status=(
                ScholarlyResolutionStatus
                .VALIDATED_CROSSREF
            ),
            selected_candidate=(
                recovery.selected_candidate
            ),
            selected_score=(
                recovery.selected_score
            ),
            crossref_assessment=recovery,
            core_assessment=None,
            reason=(
                "CORE failed operationally, and a bounded structured "
                "Crossref recovery search provided sufficient "
                "deterministic validation."
            ),
        )

    merged_candidates = [
        item.candidate
        for item in recovery.ranked_candidates
    ] + [
        item.candidate
        for item in initial_crossref.ranked_candidates
    ]

    combined = CrossrefAssessment(
        reference_index=evidence.reference_index,
        status=CrossrefAssessmentStatus.NEEDS_CORE,
        selected_candidate=None,
        selected_score=None,
        ranked_candidates=rank_candidates(
            evidence,
            merged_candidates,
        ),
        reason=(
            "CORE failed operationally. A bounded structured "
            "Crossref recovery search produced candidate evidence "
            "but did not satisfy deterministic strong-match criteria."
        ),
    )

    return ScholarlyResolution(
        reference_index=evidence.reference_index,
        raw_reference=evidence.raw_reference,
        status=ScholarlyResolutionStatus.NEEDS_LLM,
        selected_candidate=None,
        selected_score=None,
        crossref_assessment=combined,
        core_assessment=None,
        reason=(
            "CORE was unavailable for this query; the unresolved "
            "Crossref evidence is routed to bounded LLM adjudication."
        ),
    )


def resolve_crossref_then_core(
    targets: Iterable[ReferenceEvidence],
    crossref_retrievals: Iterable[CrossrefRetrieval],
    core_client: CoreRetriever,
    *,
    crossref_client: CrossrefSearchRetriever | None = None,
    candidate_margin: float = DEFAULT_CANDIDATE_MARGIN,
) -> tuple[ScholarlyResolution, ...]:
    """Resolve references through Crossref first and CORE only when needed.

    Crossref-validated references never trigger a CORE request. References
    that Crossref cannot validate are searched in CORE and assessed with the
    same deterministic scoring policy.

    References still unresolved after CORE are preserved with complete
    Crossref and CORE evidence for the later LLM adjudication stage.
    """

    target_map: dict[int, ReferenceEvidence] = {}

    for target in targets:
        if target.reference_index in target_map:
            raise ValueError(
                "Duplicate Stage 6 target bibliography index: "
                f"{target.reference_index}"
            )

        target_map[target.reference_index] = target

    crossref_assessments = (
        assess_crossref_retrievals(
            target_map.values(),
            crossref_retrievals,
            candidate_margin=candidate_margin,
        )
    )

    assessment_map = {
        assessment.reference_index: assessment
        for assessment in crossref_assessments
    }

    core_indices = tuple(
        index
        for index in sorted(target_map)
        if assessment_map[index].needs_core
    )
    core_total = len(core_indices)
    core_position = 0

    LOGGER.info(
        "[CORE] %d/%d references require CORE fallback",
        core_total,
        len(target_map),
    )

    results: list[ScholarlyResolution] = []

    for index in sorted(target_map):
        evidence = target_map[index]
        crossref = assessment_map[index]

        if not crossref.needs_core:
            results.append(
                ScholarlyResolution(
                    reference_index=index,
                    raw_reference=evidence.raw_reference,
                    status=(
                        ScholarlyResolutionStatus
                        .VALIDATED_CROSSREF
                    ),
                    selected_candidate=(
                        crossref.selected_candidate
                    ),
                    selected_score=(
                        crossref.selected_score
                    ),
                    crossref_assessment=crossref,
                    core_assessment=None,
                    reason=(
                        "Crossref provided sufficient "
                        "deterministic validation."
                    ),
                )
            )
            continue

        core_position += 1

        LOGGER.info(
            "[CORE] %d/%d ref=%d",
            core_position,
            core_total,
            evidence.reference_index,
        )

        if is_publisher_boilerplate_reference(
            evidence.raw_reference
        ):
            core = CoreAssessment(
                reference_index=evidence.reference_index,
                status=CoreAssessmentStatus.NEEDS_LLM,
                selected_candidate=None,
                selected_score=None,
                ranked_candidates=(),
                reason=(
                    "Standardized publisher disclaimer is not "
                    "a scholarly-work reference; CORE lookup skipped."
                ),
            )

            results.append(
                ScholarlyResolution(
                    reference_index=index,
                    raw_reference=evidence.raw_reference,
                    status=ScholarlyResolutionStatus.NEEDS_LLM,
                    selected_candidate=None,
                    selected_score=None,
                    crossref_assessment=crossref,
                    core_assessment=core,
                    reason=(
                        "Publisher boilerplate requires deterministic "
                        "non-scholarly rejection."
                    ),
                )
            )

            LOGGER.info(
                "[CORE] %d/%d ref=%d -> skipped publisher boilerplate",
                core_position,
                core_total,
                evidence.reference_index,
            )

            continue

        try:
            core_response = core_client.search_works(
                evidence.raw_reference
            )
        except CoreError as error:
            recovered = None

            if crossref_client is not None:
                recovered = _recover_after_core_failure(
                    evidence,
                    crossref,
                    crossref_client,
                    candidate_margin=candidate_margin,
                )

            if recovered is not None:
                LOGGER.warning(
                    "[CORE] ref=%d failed operationally; "
                    "using bounded structured Crossref recovery",
                    evidence.reference_index,
                )

                results.append(
                    recovered
                )
                continue

            raise CoreError(
                "CORE lookup failed for bibliography "
                f"index {evidence.reference_index}: {error}"
            ) from error

        core = assess_core(
            evidence,
            core_response,
            candidate_margin=candidate_margin,
        )

        LOGGER.info(
            "[CORE] %d/%d ref=%d -> %s",
            core_position,
            core_total,
            evidence.reference_index,
            core.status.value,
        )

        if not core.needs_llm:
            results.append(
                ScholarlyResolution(
                    reference_index=index,
                    raw_reference=evidence.raw_reference,
                    status=(
                        ScholarlyResolutionStatus
                        .VALIDATED_CORE
                    ),
                    selected_candidate=(
                        core.selected_candidate
                    ),
                    selected_score=core.selected_score,
                    crossref_assessment=crossref,
                    core_assessment=core,
                    reason=(
                        "Crossref was insufficient, but CORE "
                        "provided sufficient deterministic "
                        "validation."
                    ),
                )
            )
            continue

        results.append(
            ScholarlyResolution(
                reference_index=index,
                raw_reference=evidence.raw_reference,
                status=ScholarlyResolutionStatus.NEEDS_LLM,
                selected_candidate=None,
                selected_score=None,
                crossref_assessment=crossref,
                core_assessment=core,
                reason=(
                    "Neither Crossref nor CORE provided "
                    "sufficient deterministic evidence; "
                    "LLM adjudication is required."
                ),
            )
        )

    return tuple(results)
