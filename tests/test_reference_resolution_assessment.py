from __future__ import annotations

import pytest

from tabulus.reference_resolution import (
    CrossrefAssessmentStatus,
    CrossrefRetrieval,
    ReferenceEvidence,
    ResolutionCandidate,
    assess_crossref,
    assess_crossref_retrievals,
    rank_candidates,
)


def _strong_evidence(
    *,
    index: int = 1,
    doi: str = "",
) -> ReferenceEvidence:
    return ReferenceEvidence(
        reference_index=index,
        raw_reference=(
            "Smith J. Atomic layer deposition of example films. "
            "Example Journal 12, 100-110 (2020)."
        ),
        doi=doi,
        title=(
            "Atomic layer deposition of example films"
        ),
        authors=("J. Smith",),
        year=2020,
        venue="Example Journal",
        volume="12",
        pages="100-110",
    )


def _exact_candidate(
    *,
    doi: str = "10.1000/example",
) -> ResolutionCandidate:
    return ResolutionCandidate(
        source="crossref",
        source_id=doi,
        doi=doi,
        title=(
            "Atomic Layer Deposition of Example Films"
        ),
        authors=("Jane Smith",),
        year=2020,
        venue="Example Journal",
        volume="12",
        pages="100-110",
    )


def _retrieval(
    *,
    index: int = 1,
    existing_doi: str = "",
    existing_candidate=None,
    candidates=(),
) -> CrossrefRetrieval:
    return CrossrefRetrieval(
        reference_index=index,
        raw_reference="Example raw reference.",
        existing_doi=existing_doi,
        existing_doi_checked=bool(existing_doi),
        existing_doi_candidate=existing_candidate,
        bibliographic_search_performed=(
            existing_candidate is None
        ),
        bibliographic_candidates=tuple(candidates),
    )


def test_existing_doi_exactly_returned_by_crossref_is_validated() -> None:
    evidence = _strong_evidence(
        doi="10.1000/example",
    )

    candidate = _exact_candidate()

    result = assess_crossref(
        evidence,
        _retrieval(
            existing_doi="10.1000/example",
            existing_candidate=candidate,
        ),
    )

    assert result.status == (
        CrossrefAssessmentStatus
        .VALIDATED_EXISTING_DOI
    )
    assert result.selected_candidate == candidate
    assert result.needs_core is False


def test_strong_bibliographic_candidate_is_validated() -> None:
    evidence = _strong_evidence()
    candidate = _exact_candidate()

    result = assess_crossref(
        evidence,
        _retrieval(
            candidates=(candidate,),
        ),
    )

    assert result.status == (
        CrossrefAssessmentStatus
        .VALIDATED_SEARCH_MATCH
    )
    assert result.selected_candidate == candidate
    assert result.selected_score is not None
    assert result.selected_score.strong_match is True


def test_author_and_year_only_candidate_falls_through_to_core() -> None:
    evidence = ReferenceEvidence(
        reference_index=1,
        raw_reference="Smith 2020",
        authors=("Smith",),
        year=2020,
    )

    candidate = ResolutionCandidate(
        source="crossref",
        title="Unknown paper",
        authors=("Jane Smith",),
        year=2020,
    )

    result = assess_crossref(
        evidence,
        _retrieval(
            candidates=(candidate,),
        ),
    )

    assert result.status == (
        CrossrefAssessmentStatus.NEEDS_CORE
    )
    assert result.selected_candidate is None


def test_no_crossref_candidates_falls_through_to_core() -> None:
    evidence = _strong_evidence()

    result = assess_crossref(
        evidence,
        _retrieval(candidates=()),
    )

    assert result.status == (
        CrossrefAssessmentStatus.NEEDS_CORE
    )
    assert result.ranked_candidates == ()


def test_close_strong_candidates_are_treated_as_ambiguous() -> None:
    evidence = _strong_evidence()

    best = _exact_candidate(
        doi="10.1000/best",
    )

    close = ResolutionCandidate(
        source="crossref",
        source_id="10.1000/close",
        doi="10.1000/close",
        title=(
            "Atomic Layer Deposition of Example Films"
        ),
        authors=("Jane Smith",),
        year=2020,
        venue="Example Journal",
        volume="99",
        pages="100-110",
    )

    result = assess_crossref(
        evidence,
        _retrieval(
            candidates=(close, best),
        ),
    )

    assert result.status == (
        CrossrefAssessmentStatus.NEEDS_CORE
    )
    assert len(result.ranked_candidates) == 2
    assert (
        result.ranked_candidates[0]
        .candidate.doi
        == "10.1000/best"
    )


def test_clear_top_candidate_is_accepted_over_weaker_runner_up() -> None:
    evidence = _strong_evidence()

    best = _exact_candidate(
        doi="10.1000/best",
    )

    weaker = ResolutionCandidate(
        source="crossref",
        source_id="10.1000/weaker",
        doi="10.1000/weaker",
        title=(
            "Atomic Layer Deposition of Example Films"
        ),
        authors=("Jane Smith",),
        year=2017,
        venue="Example Journal",
        volume="99",
        pages="999-1009",
    )

    result = assess_crossref(
        evidence,
        _retrieval(
            candidates=(weaker, best),
        ),
    )

    assert result.status == (
        CrossrefAssessmentStatus
        .VALIDATED_SEARCH_MATCH
    )
    assert (
        result.selected_candidate.doi
        == "10.1000/best"
    )


def test_rank_candidates_is_deterministic() -> None:
    evidence = _strong_evidence()

    candidate_b = _exact_candidate(
        doi="10.1000/b",
    )
    candidate_a = _exact_candidate(
        doi="10.1000/a",
    )

    ranked = rank_candidates(
        evidence,
        (candidate_b, candidate_a),
    )

    assert [
        item.candidate.doi
        for item in ranked
    ] == [
        "10.1000/a",
        "10.1000/b",
    ]


def test_assess_crossref_retrievals_requires_matching_indices() -> None:
    evidence = _strong_evidence(
        index=1,
    )

    retrieval = _retrieval(
        index=2,
        candidates=(_exact_candidate(),),
    )

    with pytest.raises(
        ValueError,
        match="indices do not match",
    ):
        assess_crossref_retrievals(
            [evidence],
            [retrieval],
        )


def test_container_book_title_is_not_used_as_work_title_evidence() -> None:
    evidence = ReferenceEvidence(
        reference_index=16,
        raw_reference=(
            "N. J. Mason, in Atomic Layer Epitaxy, "
            "edited by T. Suntola and M. Simpson "
            "(Blackie and Son, London, 1990), "
            "pp. 63-109."
        ),
        title="Atomic Layer Epitaxy",
        authors=("N J Mason",),
        year=1990,
        pages="63-109",
    )

    container_book = ResolutionCandidate(
        source="crossref",
        source_id="10.1007/978-94-009-0389-0",
        doi="10.1007/978-94-009-0389-0",
        title="Atomic Layer Epitaxy",
        authors=(),
        year=1990,
    )

    result = assess_crossref(
        evidence,
        _retrieval(
            index=16,
            candidates=(container_book,),
        ),
    )

    assert result.status == CrossrefAssessmentStatus.NEEDS_CORE
    assert result.selected_candidate is None
    assert result.ranked_candidates
    assert (
        "title"
        not in result.ranked_candidates[0].score.comparable_fields
    )


def test_chapter_candidate_can_validate_without_container_title() -> None:
    evidence = ReferenceEvidence(
        reference_index=16,
        raw_reference=(
            "N. J. Mason, in Atomic Layer Epitaxy, "
            "edited by T. Suntola and M. Simpson "
            "(Blackie and Son, London, 1990), "
            "pp. 63-109."
        ),
        title="Atomic Layer Epitaxy",
        authors=("N J Mason",),
        year=1990,
        pages="63-109",
    )

    chapter_candidate = ResolutionCandidate(
        source="crossref",
        source_id="chapter:16",
        title="A chapter-level scholarly contribution",
        authors=("N. J. Mason",),
        year=1990,
        pages="63-109",
    )

    result = assess_crossref(
        evidence,
        _retrieval(
            index=16,
            candidates=(chapter_candidate,),
        ),
    )

    assert result.status == CrossrefAssessmentStatus.VALIDATED_SEARCH_MATCH
    assert result.selected_candidate == chapter_candidate
    assert result.selected_score is not None
    assert result.selected_score.sufficient_evidence is True
    assert result.selected_score.strong_match is True
    assert "title" not in result.selected_score.comparable_fields


def test_explicit_edition_year_conflict_cannot_validate_candidate() -> None:
    evidence = ReferenceEvidence(
        reference_index=1112,
        raw_reference=(
            "H. S. Fogler, Elements of Chemical Reaction Engineering, "
            "2nd ed. (Prentice Hall, Upper Saddle River, "
            "New Jersey, 1992)."
        ),
        title="Elements of Chemical Reaction Engineering",
        authors=("H S Fogler",),
        year=1992,
    )

    wrong_edition_candidate = ResolutionCandidate(
        source="core",
        source_id="89530943",
        title="Elements of Chemical Reaction Engineering",
        authors=("Fogler, H Scott",),
        year=2020,
    )

    result = assess_crossref(
        evidence,
        _retrieval(
            index=1112,
            candidates=(wrong_edition_candidate,),
        ),
    )

    assert result.status == CrossrefAssessmentStatus.NEEDS_CORE
    assert result.selected_candidate is None

    score = result.ranked_candidates[0].score

    assert score.sufficient_evidence is False
    assert score.strong_match is False
    assert dict(score.field_scores)["title"] == 1.0
    assert dict(score.field_scores)["authors"] == 1.0
    assert dict(score.field_scores)["year"] == 0.0


def test_explicit_edition_same_year_candidate_can_validate() -> None:
    evidence = ReferenceEvidence(
        reference_index=1112,
        raw_reference=(
            "H. S. Fogler, Elements of Chemical Reaction Engineering, "
            "2nd ed. (Prentice Hall, Upper Saddle River, "
            "New Jersey, 1992)."
        ),
        title="Elements of Chemical Reaction Engineering",
        authors=("H S Fogler",),
        year=1992,
    )

    candidate = ResolutionCandidate(
        source="crossref",
        source_id="fogler-1992",
        title="Elements of Chemical Reaction Engineering",
        authors=("H. Scott Fogler",),
        year=1992,
    )

    result = assess_crossref(
        evidence,
        _retrieval(
            index=1112,
            candidates=(candidate,),
        ),
    )

    assert result.status == (
        CrossrefAssessmentStatus.VALIDATED_SEARCH_MATCH
    )
    assert result.selected_candidate == candidate
    assert result.selected_score is not None
    assert result.selected_score.sufficient_evidence is True
    assert result.selected_score.strong_match is True
