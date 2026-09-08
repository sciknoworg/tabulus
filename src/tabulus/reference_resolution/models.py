from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ResolutionStatus(str, Enum):
    """Final validation state for one bibliography entry."""

    VALIDATED_WITH_DOI = "validated_with_doi"
    VALIDATED_WITHOUT_DOI = "validated_without_doi"
    UNRESOLVED = "unresolved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ReferenceEvidence:
    """Structured evidence available for one source bibliography entry.

    Fields other than ``raw_reference`` are optional because scientific
    bibliographies are frequently incomplete. Missing fields must not count as
    disagreement during candidate scoring.
    """

    reference_index: int
    raw_reference: str
    doi: str = ""
    title: str = ""
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_index": self.reference_index,
            "raw_reference": self.raw_reference,
            "doi": self.doi,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "venue": self.venue,
            "volume": self.volume,
            "issue": self.issue,
            "pages": self.pages,
        }


@dataclass(frozen=True)
class ResolutionCandidate:
    """One scholarly-work candidate returned by an external source."""

    source: str
    source_id: str = ""
    doi: str = ""
    title: str = ""
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_id": self.source_id,
            "doi": self.doi,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "venue": self.venue,
            "volume": self.volume,
            "issue": self.issue,
            "pages": self.pages,
            "url": self.url,
        }


@dataclass(frozen=True)
class CandidateScore:
    """Deterministic comparison between source evidence and one candidate."""

    score: float
    comparable_weight: float
    field_scores: tuple[tuple[str, float], ...]
    sufficient_evidence: bool
    strong_match: bool

    @property
    def comparable_fields(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.field_scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "comparable_weight": self.comparable_weight,
            "field_scores": {
                name: value
                for name, value in self.field_scores
            },
            "comparable_fields": list(self.comparable_fields),
            "sufficient_evidence": self.sufficient_evidence,
            "strong_match": self.strong_match,
        }


@dataclass(frozen=True)
class ReferenceResolution:
    """Persistable Stage 6 decision for one unique bibliography entry."""

    reference_index: int
    raw_reference: str
    status: ResolutionStatus
    canonical_doi: str = ""
    canonical_title: str = ""
    canonical_authors: tuple[str, ...] = ()
    canonical_year: int | None = None
    canonical_venue: str = ""
    source: str = ""
    confidence: float | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_index": self.reference_index,
            "raw_reference": self.raw_reference,
            "status": self.status.value,
            "canonical_doi": self.canonical_doi,
            "canonical_title": self.canonical_title,
            "canonical_authors": list(self.canonical_authors),
            "canonical_year": self.canonical_year,
            "canonical_venue": self.canonical_venue,
            "source": self.source,
            "confidence": self.confidence,
            "reason": self.reason,
        }
