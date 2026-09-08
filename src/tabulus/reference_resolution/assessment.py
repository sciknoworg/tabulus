from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from tabulus.reference_resolution.models import (
    CandidateScore,
    ReferenceEvidence,
    ResolutionCandidate,
)
from tabulus.reference_resolution.pipeline import (
    CrossrefRetrieval,
)
from tabulus.reference_resolution.scoring import (
    normalize_doi,
    score_candidate,
)


DEFAULT_CANDIDATE_MARGIN = 0.08


class CrossrefAssessmentStatus(str, Enum):
    """Outcome of the Crossref portion of Stage 6."""

    VALIDATED_EXISTING_DOI = "validated_existing_doi"
    VALIDATED_SEARCH_MATCH = "validated_search_match"
    NEEDS_CORE = "needs_core"


@dataclass(frozen=True)
class RankedCandidate:
    """One Crossref candidate plus deterministic validation evidence."""

    candidate: ResolutionCandidate
    score: CandidateScore

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "score": self.score.to_dict(),
        }


@dataclass(frozen=True)
class CrossrefAssessment:
    """Decision produced after deterministic Crossref validation."""

    reference_index: int
    status: CrossrefAssessmentStatus
    selected_candidate: ResolutionCandidate | None
    selected_score: CandidateScore | None
    ranked_candidates: tuple[RankedCandidate, ...]
    reason: str

    @property
    def needs_core(self) -> bool:
        return (
            self.status
            == CrossrefAssessmentStatus.NEEDS_CORE
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
                item.to_dict()
                for item in self.ranked_candidates
            ],
            "reason": self.reason,
        }


def rank_candidates(
    evidence: ReferenceEvidence,
    candidates: Iterable[ResolutionCandidate],
) -> tuple[RankedCandidate, ...]:
    """Score and deterministically rank external work candidates."""

    ranked = [
        RankedCandidate(
            candidate=candidate,
            score=score_candidate(
                evidence,
                candidate,
            ),
        )
        for candidate in candidates
    ]

    ranked.sort(
        key=lambda item: (
            -item.score.score,
            -item.score.comparable_weight,
            normalize_doi(
                item.candidate.doi
            ),
            item.candidate.source_id.casefold(),
            item.candidate.title.casefold(),
        )
    )

    return tuple(ranked)


def assess_crossref(
    evidence: ReferenceEvidence,
    retrieval: CrossrefRetrieval,
    *,
    candidate_margin: float = DEFAULT_CANDIDATE_MARGIN,
) -> CrossrefAssessment:
    """Determine whether Crossref evidence is sufficient to stop resolution.

    Existing DOI lookup is treated separately from bibliographic search.
    An exact DOI returned by Crossref validates the identifier because the DOI
    itself is unique evidence.

    Bibliographic search candidates must satisfy the deterministic strong-match
    criteria and, when another plausible candidate exists, must be clearly
    better than the runner-up. Otherwise the reference proceeds to CORE.
    """

    if evidence.reference_index != retrieval.reference_index:
        raise ValueError(
            "Reference evidence and Crossref retrieval "
            "must describe the same bibliography index."
        )

    if candidate_margin < 0:
        raise ValueError(
            "candidate_margin must be non-negative."
        )

    existing = retrieval.existing_doi_candidate

    if existing is not None:
        expected_doi = normalize_doi(
            evidence.doi
        )
        candidate_doi = normalize_doi(
            existing.doi
        )

        existing_score = score_candidate(
            evidence,
            existing,
        )

        if (
            expected_doi
            and candidate_doi
            and expected_doi == candidate_doi
        ):
            return CrossrefAssessment(
                reference_index=evidence.reference_index,
                status=(
                    CrossrefAssessmentStatus
                    .VALIDATED_EXISTING_DOI
                ),
                selected_candidate=existing,
                selected_score=existing_score,
                ranked_candidates=(),
                reason=(
                    "Existing DOI was found in Crossref "
                    "and the returned DOI agrees exactly."
                ),
            )

        # A DOI endpoint returning a different identifier would be
        # inconsistent evidence. Do not accept it; continue with the
        # bibliographic-search evidence already collected by the pipeline.

    ranked = rank_candidates(
        evidence,
        retrieval.bibliographic_candidates,
    )

    if not ranked:
        return CrossrefAssessment(
            reference_index=evidence.reference_index,
            status=CrossrefAssessmentStatus.NEEDS_CORE,
            selected_candidate=None,
            selected_score=None,
            ranked_candidates=(),
            reason=(
                "Crossref produced no bibliographic candidate "
                "that could be assessed."
            ),
        )

    top = ranked[0]

    if not top.score.strong_match:
        return CrossrefAssessment(
            reference_index=evidence.reference_index,
            status=CrossrefAssessmentStatus.NEEDS_CORE,
            selected_candidate=None,
            selected_score=None,
            ranked_candidates=ranked,
            reason=(
                "The highest-ranked Crossref candidate "
                "did not satisfy the strong-match criteria."
            ),
        )

    if len(ranked) > 1:
        runner_up = ranked[1]

        # Only treat a runner-up as meaningfully competitive when it contains
        # sufficient bibliographic evidence. A weak or essentially
        # unscorable second result should not block a strong top candidate.
        if runner_up.score.sufficient_evidence:
            margin = (
                top.score.score
                - runner_up.score.score
            )

            if margin < candidate_margin:
                return CrossrefAssessment(
                    reference_index=evidence.reference_index,
                    status=(
                        CrossrefAssessmentStatus
                        .NEEDS_CORE
                    ),
                    selected_candidate=None,
                    selected_score=None,
                    ranked_candidates=ranked,
                    reason=(
                        "Multiple Crossref candidates are "
                        "too close to distinguish reliably."
                    ),
                )

    return CrossrefAssessment(
        reference_index=evidence.reference_index,
        status=(
            CrossrefAssessmentStatus
            .VALIDATED_SEARCH_MATCH
        ),
        selected_candidate=top.candidate,
        selected_score=top.score,
        ranked_candidates=ranked,
        reason=(
            "The highest-ranked Crossref candidate "
            "satisfied the strong-match criteria and "
            "was sufficiently distinct from alternatives."
        ),
    )


def assess_crossref_retrievals(
    targets: Iterable[ReferenceEvidence],
    retrievals: Iterable[CrossrefRetrieval],
    *,
    candidate_margin: float = DEFAULT_CANDIDATE_MARGIN,
) -> tuple[CrossrefAssessment, ...]:
    """Assess a collection of Crossref retrieval results by bibliography index."""

    target_map: dict[int, ReferenceEvidence] = {}

    for target in targets:
        if target.reference_index in target_map:
            raise ValueError(
                "Duplicate Stage 6 target bibliography index: "
                f"{target.reference_index}"
            )

        target_map[target.reference_index] = target

    retrieval_map: dict[int, CrossrefRetrieval] = {}

    for retrieval in retrievals:
        if retrieval.reference_index in retrieval_map:
            raise ValueError(
                "Duplicate Crossref retrieval bibliography index: "
                f"{retrieval.reference_index}"
            )

        retrieval_map[
            retrieval.reference_index
        ] = retrieval

    if set(target_map) != set(retrieval_map):
        missing_retrievals = sorted(
            set(target_map)
            - set(retrieval_map)
        )
        unexpected_retrievals = sorted(
            set(retrieval_map)
            - set(target_map)
        )

        raise ValueError(
            "Crossref retrieval indices do not match "
            "Stage 6 targets. "
            f"Missing={missing_retrievals}; "
            f"unexpected={unexpected_retrievals}"
        )

    return tuple(
        assess_crossref(
            target_map[index],
            retrieval_map[index],
            candidate_margin=candidate_margin,
        )
        for index in sorted(target_map)
    )
