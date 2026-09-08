from __future__ import annotations

from tabulus.reference_resolution import (
    ReferenceEvidence,
    ReferenceResolution,
    ResolutionCandidate,
    ResolutionStatus,
    normalize_doi,
    score_candidate,
)


def test_reference_resolution_models_are_serializable() -> None:
    resolution = ReferenceResolution(
        reference_index=12,
        raw_reference="Smith J. Example. 2020.",
        status=ResolutionStatus.VALIDATED_WITH_DOI,
        canonical_doi="10.1000/example",
        canonical_title="Example",
        canonical_authors=("Jane Smith",),
        canonical_year=2020,
        canonical_venue="Example Journal",
        source="crossref",
        confidence=0.98,
        reason="Validated against Crossref metadata.",
    )

    payload = resolution.to_dict()

    assert payload["reference_index"] == 12
    assert payload["status"] == "validated_with_doi"
    assert payload["canonical_doi"] == "10.1000/example"
    assert payload["canonical_authors"] == ["Jane Smith"]


def test_normalize_doi_removes_common_prefixes() -> None:
    assert normalize_doi("https://doi.org/10.1000/ABC.") == "10.1000/abc"
    assert normalize_doi("doi:10.1000/abc") == "10.1000/abc"


def test_exact_candidate_scores_as_strong_match() -> None:
    evidence = ReferenceEvidence(
        reference_index=1,
        raw_reference="Smith J. Example paper. Example Journal 12, 100-110 (2020).",
        title="Example paper",
        authors=("J. Smith",),
        year=2020,
        venue="Example Journal",
        volume="12",
        pages="100-110",
    )
    candidate = ResolutionCandidate(
        source="crossref",
        source_id="candidate-1",
        doi="10.1000/example",
        title="Example paper",
        authors=("Jane Smith",),
        year=2020,
        venue="Example Journal",
        volume="12",
        pages="100-110",
    )

    result = score_candidate(evidence, candidate)

    assert result.score == 1.0
    assert result.sufficient_evidence is True
    assert result.strong_match is True


def test_missing_title_does_not_penalize_titleless_reference() -> None:
    evidence = ReferenceEvidence(
        reference_index=2,
        raw_reference="Smith J., Example Journal 12, 100-110 (2020).",
        authors=("J. Smith",),
        year=2020,
        venue="Example Journal",
        volume="12",
        pages="100-110",
    )
    candidate = ResolutionCandidate(
        source="core",
        source_id="candidate-2",
        title="A title absent from the source citation",
        authors=("Jane Smith",),
        year=2020,
        venue="Example Journal",
        volume="12",
        pages="100-110",
    )

    result = score_candidate(evidence, candidate)

    assert "title" not in result.comparable_fields
    assert result.score == 1.0
    assert result.sufficient_evidence is True
    assert result.strong_match is True


def test_author_and_year_alone_are_not_enough_for_automatic_acceptance() -> None:
    evidence = ReferenceEvidence(
        reference_index=3,
        raw_reference="Smith 2020",
        authors=("Smith",),
        year=2020,
    )
    candidate = ResolutionCandidate(
        source="crossref",
        authors=("Jane Smith",),
        year=2020,
        title="An otherwise unknown work",
    )

    result = score_candidate(evidence, candidate)

    assert result.score == 1.0
    assert result.sufficient_evidence is False
    assert result.strong_match is False


def test_title_and_year_can_supply_sufficient_evidence() -> None:
    evidence = ReferenceEvidence(
        reference_index=4,
        raw_reference="Example title. 2021.",
        title="Atomic layer deposition of example films",
        year=2021,
    )
    candidate = ResolutionCandidate(
        source="crossref",
        title="Atomic Layer Deposition of Example Films",
        year=2021,
    )

    result = score_candidate(evidence, candidate)

    assert result.sufficient_evidence is True
    assert result.strong_match is True


def test_conflicting_metadata_reduces_candidate_score() -> None:
    evidence = ReferenceEvidence(
        reference_index=5,
        raw_reference="Smith J. Example paper. 2020. pp. 10-20.",
        title="Example paper",
        authors=("J. Smith",),
        year=2020,
        pages="10-20",
    )
    candidate = ResolutionCandidate(
        source="crossref",
        title="Example paper",
        authors=("Jane Smith",),
        year=2017,
        pages="99-110",
    )

    result = score_candidate(evidence, candidate)

    assert result.score < 0.85
    assert result.strong_match is False


def test_exact_doi_is_sufficient_even_without_other_metadata() -> None:
    evidence = ReferenceEvidence(
        reference_index=6,
        raw_reference="doi:10.1000/example",
        doi="10.1000/example",
    )
    candidate = ResolutionCandidate(
        source="crossref",
        doi="https://doi.org/10.1000/EXAMPLE",
    )

    result = score_candidate(evidence, candidate)

    assert result.score == 1.0
    assert result.sufficient_evidence is True
    assert result.strong_match is True


def test_conflicting_doi_can_never_be_strong_match() -> None:
    evidence = ReferenceEvidence(
        reference_index=7,
        raw_reference="doi:10.1000/source",
        doi="10.1000/source",
        title="Example paper",
        year=2020,
    )
    candidate = ResolutionCandidate(
        source="crossref",
        doi="10.1000/different",
        title="Example paper",
        year=2020,
    )

    result = score_candidate(evidence, candidate)

    assert result.strong_match is False


def test_no_comparable_metadata_is_unscorable() -> None:
    evidence = ReferenceEvidence(
        reference_index=8,
        raw_reference="Unstructured citation",
    )
    candidate = ResolutionCandidate(
        source="core",
        title="Some work",
    )

    result = score_candidate(evidence, candidate)

    assert result.score == 0.0
    assert result.comparable_weight == 0.0
    assert result.comparable_fields == ()
    assert result.sufficient_evidence is False
    assert result.strong_match is False


def test_core_style_surname_first_author_matches() -> None:
    evidence = ReferenceEvidence(
        reference_index=9,
        raw_reference="Stefik M. Example. 2020.",
        authors=("M. Stefik",),
        year=2020,
    )

    candidate = ResolutionCandidate(
        source="core",
        authors=("Stefik, M.",),
        year=2020,
    )

    result = score_candidate(
        evidence,
        candidate,
    )

    assert dict(result.field_scores)["authors"] == 1.0


