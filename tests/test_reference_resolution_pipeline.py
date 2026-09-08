from __future__ import annotations

import json
from pathlib import Path

import pytest

from tabulus.reference_resolution import (
    ResolutionCandidate,
    collect_resolution_targets,
    retrieve_crossref_evidence,
)


def _write_json(
    path: Path,
    payload: dict,
) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _bibliography(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "bibliography.json",
        {
            "bibliography_count": 4,
            "bibliography_source": "grobid",
            "entries": [
                {
                    "index": 1,
                    "raw": "Reference one.",
                    "doi": "10.1000/one",
                    "source": "grobid",
                },
                {
                    "index": 2,
                    "raw": "Reference two.",
                    "doi": "",
                    "source": "grobid",
                },
                {
                    "index": 3,
                    "raw": "Reference three.",
                    "doi": "",
                    "source": "grobid",
                },
                {
                    "index": 4,
                    "raw": "Reference four.",
                    "doi": "",
                    "source": "grobid",
                },
            ],
        },
    )


def _matches(
    path: Path,
    indices_by_row: list[list[int]],
) -> Path:
    return _write_json(
        path,
        {
            "schema_version": 1,
            "matched_tables": [
                {
                    "table_id": 1,
                    "matches": [
                        {
                            "row_index": row_index,
                            "matched_reference_indices": indices,
                        }
                        for row_index, indices
                        in enumerate(indices_by_row)
                    ],
                }
            ],
        },
    )


def test_collect_resolution_targets_only_uses_linked_entries(
    tmp_path,
) -> None:
    bibliography = _bibliography(tmp_path)
    matches = _matches(
        tmp_path / "matches.json",
        [[2], [4]],
    )

    targets = collect_resolution_targets(
        bibliography,
        [matches],
    )

    assert [
        target.reference_index
        for target in targets
    ] == [2, 4]

    assert [
        target.raw_reference
        for target in targets
    ] == [
        "Reference two.",
        "Reference four.",
    ]


def test_collect_resolution_targets_unions_multiple_adapters(
    tmp_path,
) -> None:
    bibliography = _bibliography(tmp_path)

    adapter_a = _matches(
        tmp_path / "adapter-a.json",
        [[1, 2], [2]],
    )
    adapter_b = _matches(
        tmp_path / "adapter-b.json",
        [[2, 3], [1]],
    )

    targets = collect_resolution_targets(
        bibliography,
        [adapter_a, adapter_b],
    )

    assert [
        target.reference_index
        for target in targets
    ] == [1, 2, 3]


def test_collect_resolution_targets_requires_matches_artifact(
    tmp_path,
) -> None:
    bibliography = _bibliography(tmp_path)

    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        collect_resolution_targets(
            bibliography,
            [],
        )


def test_collect_resolution_targets_rejects_unknown_bibliography_index(
    tmp_path,
) -> None:
    bibliography = _bibliography(tmp_path)
    matches = _matches(
        tmp_path / "matches.json",
        [[99]],
    )

    with pytest.raises(
        ValueError,
        match="99",
    ):
        collect_resolution_targets(
            bibliography,
            [matches],
        )



def test_collect_resolution_targets_preserves_structured_metadata(
    tmp_path,
) -> None:
    bibliography = _write_json(
        tmp_path / "bibliography-enriched.json",
        {
            "bibliography_count": 1,
            "bibliography_source": "grobid",
            "entries": [
                {
                    "index": 1,
                    "raw": "Smith J. Example paper.",
                    "doi": "",
                    "source": "grobid",
                    "title": "Example paper",
                    "authors": ["Jane Smith"],
                    "year": 2020,
                    "venue": "Example Journal",
                    "volume": "12",
                    "issue": "3",
                    "pages": "100-110",
                }
            ],
        },
    )

    matches = _matches(
        tmp_path / "matches-enriched.json",
        [[1]],
    )

    targets = collect_resolution_targets(
        bibliography,
        [matches],
    )

    assert len(targets) == 1

    target = targets[0]

    assert target.title == "Example paper"
    assert target.authors == ("Jane Smith",)
    assert target.year == 2020
    assert target.venue == "Example Journal"
    assert target.volume == "12"
    assert target.issue == "3"
    assert target.pages == "100-110"



class _FakeCrossrefClient:
    def __init__(self):
        self.doi_calls: list[str] = []
        self.search_calls: list[str] = []
        self.doi_results: dict[
            str,
            ResolutionCandidate | None,
        ] = {}
        self.search_results: dict[
            str,
            tuple[ResolutionCandidate, ...],
        ] = {}

    def get_by_doi(
        self,
        doi: str,
    ) -> ResolutionCandidate | None:
        self.doi_calls.append(doi)
        return self.doi_results.get(doi)

    def search_bibliographic(
        self,
        reference_text: str,
    ) -> tuple[ResolutionCandidate, ...]:
        self.search_calls.append(reference_text)
        return self.search_results.get(
            reference_text,
            (),
        )


def test_existing_doi_is_checked_before_search(
    tmp_path,
) -> None:
    bibliography = _bibliography(tmp_path)
    matches = _matches(
        tmp_path / "matches.json",
        [[1]],
    )

    targets = collect_resolution_targets(
        bibliography,
        [matches],
    )

    client = _FakeCrossrefClient()
    client.doi_results["10.1000/one"] = (
        ResolutionCandidate(
            source="crossref",
            doi="10.1000/one",
            title="Reference one",
        )
    )

    result = retrieve_crossref_evidence(
        targets,
        client,
    )

    assert client.doi_calls == ["10.1000/one"]
    assert client.search_calls == []
    assert result[0].existing_doi_checked is True
    assert (
        result[0].existing_doi_candidate is not None
    )
    assert (
        result[0].bibliographic_search_performed
        is False
    )


def test_unvalidated_existing_doi_falls_back_to_search(
    tmp_path,
) -> None:
    bibliography = _bibliography(tmp_path)
    matches = _matches(
        tmp_path / "matches.json",
        [[1]],
    )

    targets = collect_resolution_targets(
        bibliography,
        [matches],
    )

    client = _FakeCrossrefClient()
    client.doi_results["10.1000/one"] = None
    client.search_results["Reference one."] = (
        ResolutionCandidate(
            source="crossref",
            doi="10.1000/recovered",
            title="Recovered work",
        ),
    )

    result = retrieve_crossref_evidence(
        targets,
        client,
    )

    assert client.doi_calls == ["10.1000/one"]
    assert client.search_calls == [
        "Reference one."
    ]
    assert (
        result[0].existing_doi_candidate is None
    )
    assert (
        result[0].bibliographic_search_performed
        is True
    )
    assert (
        result[0].bibliographic_candidates[0].doi
        == "10.1000/recovered"
    )


def test_missing_doi_uses_bibliographic_search(
    tmp_path,
) -> None:
    bibliography = _bibliography(tmp_path)
    matches = _matches(
        tmp_path / "matches.json",
        [[2]],
    )

    targets = collect_resolution_targets(
        bibliography,
        [matches],
    )

    client = _FakeCrossrefClient()
    client.search_results["Reference two."] = (
        ResolutionCandidate(
            source="crossref",
            doi="10.1000/two",
        ),
    )

    result = retrieve_crossref_evidence(
        targets,
        client,
    )

    assert client.doi_calls == []
    assert client.search_calls == [
        "Reference two."
    ]
    assert result[0].existing_doi_checked is False
    assert (
        result[0].bibliographic_search_performed
        is True
    )


def test_duplicate_targets_are_retrieved_only_once() -> None:
    from tabulus.reference_resolution import (
        ReferenceEvidence,
    )

    target = ReferenceEvidence(
        reference_index=7,
        raw_reference="Repeated reference.",
    )

    client = _FakeCrossrefClient()

    result = retrieve_crossref_evidence(
        [target, target, target],
        client,
    )

    assert len(result) == 1
    assert client.search_calls == [
        "Repeated reference."
    ]



def test_non_atomic_reference_detection() -> None:
    from tabulus.reference_resolution.pipeline import (
        is_non_atomic_reference,
    )

    compound = (
        "B. Y. Maa and P. D. Dapkus, Appl. Phys. Lett. "
        "58, 1762 (1991). "
        "B. Y. Maa and P. D. Dapkus, Appl. Phys. Lett. "
        "58, 2261 (1991)."
    )

    assert is_non_atomic_reference(
        compound
    ) is True


def test_non_atomic_detection_does_not_reject_multiple_years_alone() -> None:
    from tabulus.reference_resolution.pipeline import (
        is_non_atomic_reference,
    )

    proceedings = (
        "Proceedings of a conference held in 1965, "
        "published in 1967, pp. 149-155."
    )

    assert is_non_atomic_reference(
        proceedings
    ) is False



def test_discover_reference_match_artifacts_for_paper(
    tmp_path,
) -> None:
    from tabulus.reference_resolution.pipeline import (
        discover_reference_match_artifacts,
    )

    expected = []

    for adapter in (
        "adapter-b",
        "adapter-a",
    ):
        path = (
            tmp_path
            / adapter
            / f"run-{adapter}"
            / "reconstruction"
            / "Example Paper"
            / adapter
            / "references"
            / "reference_matches.json"
        )

        path.parent.mkdir(
            parents=True
        )
        path.write_text(
            "{}",
            encoding="utf-8",
        )
        expected.append(path)

    unrelated = (
        tmp_path
        / "adapter-c"
        / "run-c"
        / "reconstruction"
        / "Different Paper"
        / "adapter-c"
        / "references"
        / "reference_matches.json"
    )
    unrelated.parent.mkdir(
        parents=True
    )
    unrelated.write_text(
        "{}",
        encoding="utf-8",
    )

    discovered = (
        discover_reference_match_artifacts(
            tmp_path,
            "Example Paper",
        )
    )

    assert discovered == tuple(
        sorted(expected)
    )


def test_discover_reference_match_artifacts_rejects_duplicate_adapter_runs(
    tmp_path,
) -> None:
    import pytest

    from tabulus.reference_resolution.pipeline import (
        discover_reference_match_artifacts,
    )

    for run in (
        "run-old",
        "run-new",
    ):
        path = (
            tmp_path
            / "adapter-a"
            / run
            / "reconstruction"
            / "Example Paper"
            / "adapter-a"
            / "references"
            / "reference_matches.json"
        )

        path.parent.mkdir(
            parents=True
        )
        path.write_text(
            "{}",
            encoding="utf-8",
        )

    with pytest.raises(
        ValueError,
        match="Multiple Stage 5",
    ):
        discover_reference_match_artifacts(
            tmp_path,
            "Example Paper",
        )


def test_discover_reference_match_artifacts_requires_matching_paper(
    tmp_path,
) -> None:
    import pytest

    from tabulus.reference_resolution.pipeline import (
        discover_reference_match_artifacts,
    )

    with pytest.raises(
        FileNotFoundError,
        match="No Stage 5",
    ):
        discover_reference_match_artifacts(
            tmp_path,
            "Missing Paper",
        )


def test_non_atomic_detection_allows_bracketed_translation_pair() -> None:
    from tabulus.reference_resolution.pipeline import (
        is_non_atomic_reference,
    )

    translated = (
        "S. I. Kol'tsov, V. B. Kopylov, V. M. Smirnov, "
        "and V. B. Aleskovskii, Zh. Prikl. Khim. "
        "\u0351S.-Peterburg\u0352 49, 516 \u03511976\u0352 "
        "\u0353J. Appl. Chem. USSR 49, 525 "
        "\u03511976\u0352\u0354."
    )

    assert is_non_atomic_reference(
        translated
    ) is False


def test_non_atomic_detection_keeps_same_year_separate_works_non_atomic() -> None:
    from tabulus.reference_resolution.pipeline import (
        is_non_atomic_reference,
    )

    compound = (
        "B. Y. Maa and P. D. Dapkus, Appl. Phys. Lett. "
        "58, 1762 (1991). "
        "B. Y. Maa and P. D. Dapkus, Appl. Phys. Lett. "
        "58, 2261 (1991)."
    )

    assert is_non_atomic_reference(
        compound
    ) is True


def test_non_atomic_detection_allows_translation_published_next_year() -> None:
    from tabulus.reference_resolution.pipeline import (
        is_non_atomic_reference,
    )

    translated = (
        "A. A. Malygin, A. N. Volkova, S. I. Kol'tsov, "
        "and A. A. Aleskovskii, Zh. Obshch. Khim. "
        "43, 1436 \u03511972\u0352 "
        "\u0353J. Gen. Chem. USSR 43, 1426 "
        "\u03511973\u0352\u0354."
    )

    assert is_non_atomic_reference(
        translated
    ) is False


def test_translation_pair_year_detection_ignores_four_digit_page_number() -> None:
    from tabulus.reference_resolution.pipeline import (
        is_non_atomic_reference,
    )

    translated = (
        "S. I. Kol'tsov, T. V. Tuz, and A. N. Volkova, "
        "Zh. Prikl. Khim. 52, 2196 \u03511979\u0352 "
        "\u0353J. Appl. Chem. USSR 52, 2074 "
        "\u03511979\u0352\u0354."
    )

    assert is_non_atomic_reference(
        translated
    ) is False
