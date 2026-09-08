from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from tabulus.reference_resolution.models import (
    CandidateScore,
    ReferenceEvidence,
    ResolutionCandidate,
)


FIELD_WEIGHTS: dict[str, float] = {
    "doi": 1.00,
    "title": 0.35,
    "authors": 0.25,
    "year": 0.15,
    "venue": 0.10,
    "volume": 0.05,
    "issue": 0.03,
    "pages": 0.07,
}

STRONG_MATCH_THRESHOLD = 0.85
TITLE_STRONG_THRESHOLD = 0.90
AUTHOR_STRONG_THRESHOLD = 0.75


def normalize_doi(value: str) -> str:
    text = str(value or "").strip().casefold()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "doi.org/",
        "doi:",
    ):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text.rstrip(").,;]")


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).casefold()
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def normalize_locator(value: str) -> str:
    text = str(value or "").casefold()
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", "", text).strip(".,;")


def _text_similarity(left: str, right: str) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)

    if not left_norm or not right_norm:
        return 0.0

    if left_norm == right_norm:
        return 1.0

    return SequenceMatcher(
        None,
        left_norm,
        right_norm,
        autojunk=False,
    ).ratio()


def _title_is_probable_container_title(
    evidence: ReferenceEvidence,
) -> bool:
    """Detect when GROBID's title is actually a containing book title.

    Chapter/contribution citations commonly appear as::

        Author, in Container Title, edited by ..., pp. 1-20.

    In that structure the text following ``in`` identifies the container,
    not the cited contribution. It must therefore not be treated as
    work-title evidence during scholarly identity resolution.
    """

    if not evidence.title or not evidence.raw_reference:
        return False

    raw = str(evidence.raw_reference)

    match = re.search(
        r"(?:^|[,;])\s*in\s+"
        r"(?P<container>.+?)"
        r"\s*,?\s*edited\s+by\b",
        raw,
        flags=re.IGNORECASE,
    )

    if match is None:
        return False

    if re.search(
        r"\bpp?\.\s*\d",
        raw,
        flags=re.IGNORECASE,
    ) is None:
        return False

    container = normalize_text(
        match.group("container")
    )
    title = normalize_text(
        evidence.title
    )

    if not container or not title:
        return False

    return (
        title == container
        or title in container
        or container in title
    )


def _author_family_token(author: str) -> str:
    """Return a normalized family-name token.

    External scholarly APIs do not use one universal author-name order.
    Examples include ``Jane Smith``, ``J. Smith``, and ``Smith, J.``.
    When a comma is present, the text before the comma is treated as the
    family name. Otherwise the final normalized token is used.
    """

    raw = str(author or "").strip()

    if not raw:
        return ""

    if "," in raw:
        family = raw.split(",", 1)[0]
        normalized = normalize_text(family)

        if not normalized:
            return ""

        return normalized.split()[-1]

    normalized = normalize_text(raw)

    if not normalized:
        return ""

    return normalized.split()[-1]


def _author_similarity(
    evidence_authors: tuple[str, ...],
    candidate_authors: tuple[str, ...],
) -> float:
    evidence = {
        token
        for author in evidence_authors
        if (token := _author_family_token(author))
    }
    candidate = {
        token
        for author in candidate_authors
        if (token := _author_family_token(author))
    }

    if not evidence or not candidate:
        return 0.0

    intersection = len(evidence & candidate)
    precision = intersection / len(candidate)
    recall = intersection / len(evidence)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def _exact_text(left: str, right: str) -> float:
    left_norm = normalize_locator(left)
    right_norm = normalize_locator(right)
    if not left_norm or not right_norm:
        return 0.0
    return 1.0 if left_norm == right_norm else 0.0


def _has_explicit_numbered_edition(
    raw_reference: str,
) -> bool:
    """Return whether the source explicitly identifies a numbered edition.

    Edition-specific monograph citations identify a particular bibliographic
    manifestation. A candidate from another publication year must therefore
    not be accepted merely because author and title agree.
    """

    return (
        re.search(
            r"\b\d+\s*(?:st|nd|rd|th)\s+"
            r"ed(?:ition)?\.?"
            r"(?=\s|[,;:)\]\u0352]|$)",
            str(raw_reference or ""),
            flags=re.IGNORECASE,
        )
        is not None
    )


def _has_explicit_edition_year_conflict(
    evidence: ReferenceEvidence,
    candidate: ResolutionCandidate,
) -> bool:
    if not _has_explicit_numbered_edition(
        evidence.raw_reference
    ):
        return False

    if (
        evidence.year is None
        or candidate.year is None
    ):
        return False

    return evidence.year != candidate.year


def _field_scores(
    evidence: ReferenceEvidence,
    candidate: ResolutionCandidate,
) -> list[tuple[str, float]]:
    scores: list[tuple[str, float]] = []

    if evidence.doi and candidate.doi:
        scores.append(
            (
                "doi",
                1.0
                if normalize_doi(evidence.doi)
                == normalize_doi(candidate.doi)
                else 0.0,
            )
        )

    if (
        evidence.title
        and candidate.title
        and not _title_is_probable_container_title(
            evidence
        )
    ):
        scores.append(
            ("title", _text_similarity(evidence.title, candidate.title))
        )

    if evidence.authors and candidate.authors:
        scores.append(
            (
                "authors",
                _author_similarity(
                    evidence.authors,
                    candidate.authors,
                ),
            )
        )

    if evidence.year is not None and candidate.year is not None:
        scores.append(
            ("year", 1.0 if evidence.year == candidate.year else 0.0)
        )

    if evidence.venue and candidate.venue:
        scores.append(
            ("venue", _text_similarity(evidence.venue, candidate.venue))
        )

    if evidence.volume and candidate.volume:
        scores.append(
            ("volume", _exact_text(evidence.volume, candidate.volume))
        )

    if evidence.issue and candidate.issue:
        scores.append(
            ("issue", _exact_text(evidence.issue, candidate.issue))
        )

    if evidence.pages and candidate.pages:
        scores.append(
            ("pages", _exact_text(evidence.pages, candidate.pages))
        )

    return scores


def _has_sufficient_evidence(
    field_scores: dict[str, float],
) -> bool:
    # An exact DOI is independently decisive.
    if field_scores.get("doi") == 1.0:
        return True

    # A highly similar title plus at least one independent bibliographic clue
    # is sufficient even when author metadata is incomplete.
    if field_scores.get("title", 0.0) >= TITLE_STRONG_THRESHOLD:
        independent = {
            "authors",
            "year",
            "venue",
            "volume",
            "issue",
            "pages",
        }
        if independent & field_scores.keys():
            return True

    # Title-less citations are common. In that case require author agreement,
    # year, and at least one publication locator rather than trusting
    # author/year alone.
    if (
        field_scores.get("authors", 0.0) >= AUTHOR_STRONG_THRESHOLD
        and field_scores.get("year") == 1.0
        and any(
            field_scores.get(field, 0.0) > 0.0
            for field in ("venue", "volume", "issue", "pages")
        )
    ):
        return True

    return False


def score_candidate(
    evidence: ReferenceEvidence,
    candidate: ResolutionCandidate,
) -> CandidateScore:
    """Compare one external candidate with the available citation evidence.

    Only fields present on both sides contribute to the normalized score.
    Missing metadata therefore does not count as disagreement.

    A high numerical score alone is not sufficient for automatic acceptance:
    ``sufficient_evidence`` additionally requires a bibliographically
    discriminating combination of agreeing fields.
    """

    scores = _field_scores(evidence, candidate)

    if not scores:
        return CandidateScore(
            score=0.0,
            comparable_weight=0.0,
            field_scores=(),
            sufficient_evidence=False,
            strong_match=False,
        )

    weighted_sum = sum(
        FIELD_WEIGHTS[field] * value
        for field, value in scores
    )
    comparable_weight = sum(
        FIELD_WEIGHTS[field]
        for field, _ in scores
    )

    normalized_score = (
        weighted_sum / comparable_weight
        if comparable_weight
        else 0.0
    )

    field_score_map = dict(scores)
    sufficient = _has_sufficient_evidence(field_score_map)

    # A numbered-edition citation identifies a particular bibliographic
    # manifestation. Do not equate it with another edition/year merely from
    # author/title agreement. An exact source DOI remains independently
    # decisive in the Crossref DOI-validation path.
    edition_year_conflict = (
        _has_explicit_edition_year_conflict(
            evidence,
            candidate,
        )
    )

    if edition_year_conflict:
        sufficient = False

    # An explicitly conflicting DOI can never be a strong match.
    doi_conflict = (
        "doi" in field_score_map
        and field_score_map["doi"] == 0.0
    )

    strong = (
        not doi_conflict
        and not edition_year_conflict
        and sufficient
        and normalized_score >= STRONG_MATCH_THRESHOLD
    )

    return CandidateScore(
        score=round(normalized_score, 6),
        comparable_weight=round(comparable_weight, 6),
        field_scores=tuple(scores),
        sufficient_evidence=sufficient,
        strong_match=strong,
    )
