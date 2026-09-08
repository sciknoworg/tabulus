from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from tabulus.reference_resolution.fallback import (
    ScholarlyResolution,
    ScholarlyResolutionStatus,
)
from tabulus.reference_resolution.models import (
    CandidateScore,
    ReferenceEvidence,
    ResolutionCandidate,
)
from tabulus.reference_resolution.reference_context import (
    ReferenceContext,
)


DEFAULT_LLM_CANDIDATES_PER_SOURCE = 5


class LLMDecisionType(str, Enum):
    """Legal actions available to the Stage 6 LLM adjudicator."""

    SELECT_CANDIDATE = "select_candidate"
    RETRY_SEARCH = "retry_search"
    REJECT_ALL = "reject_all"


@dataclass(frozen=True)
class LLMCandidate:
    """One evidence-bounded candidate exposed to the LLM."""

    candidate_id: str
    candidate: ResolutionCandidate
    deterministic_score: CandidateScore

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate": self.candidate.to_dict(),
            "deterministic_score": (
                self.deterministic_score.to_dict()
            ),
        }


@dataclass(frozen=True)
class LLMAdjudicationCase:
    """Provider-neutral request contract for one unresolved reference."""

    reference_index: int
    raw_reference: str
    evidence: ReferenceEvidence
    candidates: tuple[LLMCandidate, ...]
    document_contexts: tuple[ReferenceContext, ...] = ()

    def candidate_by_id(
        self,
        candidate_id: str,
    ) -> LLMCandidate | None:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        return None

    def to_dict(self) -> dict[str, Any]:
        rules = [
            (
                "Select only a candidate_id supplied in "
                "the candidates array."
            ),
            (
                "Never invent or return a DOI, title, author, "
                "or publication that is not represented by "
                "the supplied candidates."
            ),
            (
                "Use retry_search only when the citation can "
                "plausibly be reformulated into a better "
                "bibliographic search query."
            ),
            (
                "Use reject_all when the supplied evidence "
                "does not support any candidate and a better "
                "search query cannot be justified."
            ),
        ]

        payload: dict[str, Any] = {
            "schema_version": 1,
            "reference_index": self.reference_index,
            "raw_reference": self.raw_reference,
            "structured_evidence": self.evidence.to_dict(),
            "candidates": [
                candidate.to_dict()
                for candidate in self.candidates
            ],
            "allowed_decisions": [
                decision.value
                for decision in LLMDecisionType
            ],
            "rules": rules,
        }

        if self.document_contexts:
            payload["document_contexts"] = [
                context.to_dict()
                for context in self.document_contexts
            ]

            rules.extend(
                [
                    (
                        "Document contexts are secondary evidence "
                        "from the citing paper and may support or "
                        "weaken a candidate match."
                    ),
                    (
                        "Document contexts must not be used to "
                        "invent a publication, DOI, or candidate "
                        "outside the supplied candidates array."
                    ),
                ]
            )

        return payload


@dataclass(frozen=True)
class LLMDecision:
    """Strictly validated output from an LLM adjudicator."""

    decision: LLMDecisionType
    candidate_id: str = ""
    search_query: str = ""
    confidence: float | None = None
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "candidate_id": self.candidate_id,
            "search_query": self.search_query,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
        }


def _candidate_identity(
    candidate: ResolutionCandidate,
) -> tuple[str, str, str]:
    """Identity used for duplicate suppression across scholarly sources.

    When a DOI exists it is the primary scholarly-work identity, regardless of
    whether the candidate came from Crossref or CORE. Candidates without a DOI
    fall back to provider identifier plus normalized title.
    """

    doi = candidate.doi.strip().casefold()

    if doi:
        return (
            "doi",
            doi,
            "",
        )

    return (
        candidate.source.casefold(),
        candidate.source_id.casefold(),
        candidate.title.casefold(),
    )


def build_llm_adjudication_case(
    evidence: ReferenceEvidence,
    resolution: ScholarlyResolution,
    *,
    candidates_per_source: int = DEFAULT_LLM_CANDIDATES_PER_SOURCE,
    document_contexts: tuple[ReferenceContext, ...] = (),
) -> LLMAdjudicationCase:
    """Build the bounded evidence package shown to an LLM.

    Only references unresolved by deterministic Crossref and CORE validation
    may enter this step. Candidate identifiers are generated by Tabulus; the
    model never chooses a DOI or scholarly entity outside this candidate set.
    """

    if resolution.reference_index != evidence.reference_index:
        raise ValueError(
            "Reference evidence and scholarly resolution "
            "must describe the same bibliography index."
        )

    if resolution.status != ScholarlyResolutionStatus.NEEDS_LLM:
        raise ValueError(
            "LLM adjudication is only permitted for "
            "references with status needs_llm."
        )

    if candidates_per_source <= 0:
        raise ValueError(
            "candidates_per_source must be greater than zero."
        )

    collected: list[LLMCandidate] = []
    seen: set[tuple[str, str, str]] = set()

    sources = (
        (
            "crossref",
            resolution.crossref_assessment.ranked_candidates,
        ),
        (
            "core",
            (
                resolution.core_assessment.ranked_candidates
                if resolution.core_assessment is not None
                else ()
            ),
        ),
    )

    for source, ranked in sources:
        source_rank = 0

        for ranked_candidate in ranked:
            identity = _candidate_identity(
                ranked_candidate.candidate
            )

            if identity in seen:
                continue

            seen.add(identity)
            source_rank += 1

            if source_rank > candidates_per_source:
                break

            collected.append(
                LLMCandidate(
                    candidate_id=(
                        f"{source}:{source_rank}"
                    ),
                    candidate=ranked_candidate.candidate,
                    deterministic_score=ranked_candidate.score,
                )
            )

    return LLMAdjudicationCase(
        reference_index=evidence.reference_index,
        raw_reference=evidence.raw_reference,
        evidence=evidence,
        candidates=tuple(collected),
        document_contexts=tuple(
            document_contexts
        ),
    )


def parse_llm_decision(
    payload: Any,
    case: LLMAdjudicationCase,
) -> LLMDecision:
    """Strictly validate structured LLM output.

    Unknown keys are rejected so a model cannot smuggle an invented DOI or
    other unsanctioned publication metadata into the Stage 6 artifact.
    """

    if not isinstance(payload, dict):
        raise ValueError(
            "LLM decision must be a JSON object."
        )

    # Some JSON-mode models may translate the auxiliary evidence key
    # despite otherwise respecting the requested schema. Normalize only
    # the explicitly observed alias. Identity-bearing fields such as
    # decision and candidate_id remain strictly schema-bound.
    payload = dict(payload)

    # Evidence is explanatory, non-identity-bearing metadata. Some models
    # occasionally emit harmless leading or trailing whitespace around this
    # JSON key. Normalize only this auxiliary field; identity/control fields
    # such as decision and candidate_id remain strictly schema-bound.
    whitespace_evidence_keys = [
        key
        for key in payload
        if (
            isinstance(key, str)
            and key != "evidence"
            and key.strip() == "evidence"
        )
    ]

    if whitespace_evidence_keys:
        if (
            "evidence" in payload
            or len(whitespace_evidence_keys) > 1
        ):
            raise ValueError(
                "LLM decision must not contain duplicate "
                "evidence fields."
            )

        whitespace_key = whitespace_evidence_keys[0]
        payload["evidence"] = payload.pop(
            whitespace_key
        )

    translated_evidence_key = "\u8bc1\u636e"

    if translated_evidence_key in payload:
        if "evidence" in payload:
            raise ValueError(
                "LLM decision must not contain both evidence "
                "and its translated alias."
            )

        payload["evidence"] = payload.pop(
            translated_evidence_key
        )

    allowed_keys = {
        "decision",
        "candidate_id",
        "search_query",
        "confidence",
        "evidence",
    }

    unknown_keys = set(payload) - allowed_keys

    if unknown_keys:
        raise ValueError(
            "LLM decision contains unsupported fields: "
            + ", ".join(sorted(unknown_keys))
        )

    raw_decision = payload.get("decision")

    try:
        decision = LLMDecisionType(
            raw_decision
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "LLM decision has an unsupported decision value."
        ) from error

    candidate_id = str(
        payload.get("candidate_id") or ""
    ).strip()

    search_query = str(
        payload.get("search_query") or ""
    ).strip()

    raw_confidence = payload.get("confidence")

    confidence: float | None

    if raw_confidence is None:
        confidence = None

    elif isinstance(raw_confidence, bool):
        raise ValueError(
            "LLM confidence must be numeric between 0 and 1."
        )

    elif isinstance(
        raw_confidence,
        (int, float),
    ):
        confidence = float(
            raw_confidence
        )

    elif isinstance(
        raw_confidence,
        str,
    ):
        value = raw_confidence.strip()

        if not value:
            confidence = None

        elif value.endswith("%"):
            try:
                confidence = (
                    float(
                        value[:-1].strip()
                    )
                    / 100.0
                )
            except ValueError as error:
                raise ValueError(
                    "LLM confidence must be numeric "
                    "between 0 and 1."
                ) from error

        else:
            try:
                confidence = float(
                    value
                )
            except ValueError as error:
                raise ValueError(
                    "LLM confidence must be numeric "
                    "between 0 and 1."
                ) from error

    else:
        raise ValueError(
            "LLM confidence must be numeric between 0 and 1."
        )

    if (
        confidence is not None
        and not 0.0 <= confidence <= 1.0
    ):
        raise ValueError(
            "LLM confidence must be between 0 and 1."
        )

    raw_evidence = payload.get(
        "evidence",
        [],
    )

    # ``evidence`` is explanatory provenance rather than an identity-bearing
    # trust-boundary field. Some otherwise valid JSON-mode models emit one
    # explanation as a string instead of a one-element array. Normalize that
    # harmless variation while keeping non-string structured content invalid.
    if raw_evidence is None:
        evidence_values = []
    elif isinstance(raw_evidence, str):
        evidence_values = [
            raw_evidence
        ]
    elif isinstance(raw_evidence, list):
        evidence_values = raw_evidence
    else:
        raise ValueError(
            "LLM evidence must be a string or a list of strings."
        )

    evidence_items: list[str] = []

    for item in evidence_values:
        if not isinstance(item, str):
            raise ValueError(
                "LLM evidence must contain only strings."
            )

        value = item.strip()

        if value:
            evidence_items.append(value)

    if decision == LLMDecisionType.SELECT_CANDIDATE:
        if not candidate_id:
            raise ValueError(
                "select_candidate requires candidate_id."
            )

        if case.candidate_by_id(
            candidate_id
        ) is None:
            raise ValueError(
                "LLM selected a candidate_id that was "
                "not supplied by Tabulus."
            )

        if search_query:
            raise ValueError(
                "select_candidate must not include search_query."
            )

    elif decision == LLMDecisionType.RETRY_SEARCH:
        if candidate_id:
            raise ValueError(
                "retry_search must not include candidate_id."
            )

        if not search_query:
            raise ValueError(
                "retry_search requires search_query."
            )

    elif decision == LLMDecisionType.REJECT_ALL:
        if candidate_id:
            raise ValueError(
                "reject_all must not include candidate_id."
            )

        if search_query:
            raise ValueError(
                "reject_all must not include search_query."
            )

    return LLMDecision(
        decision=decision,
        candidate_id=candidate_id,
        search_query=search_query,
        confidence=confidence,
        evidence=tuple(evidence_items),
    )
