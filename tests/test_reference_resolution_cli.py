from pathlib import Path

import pytest

import tabulus.cli as cli



def test_stage6_reference_matches_explicit_mode(
    tmp_path,
) -> None:
    first = (
        tmp_path
        / "a"
        / "reference_matches.json"
    )
    second = (
        tmp_path
        / "b"
        / "reference_matches.json"
    )

    result = cli._resolve_reference_match_paths(
        explicit_paths=[
            first,
            second,
        ],
        search_root=None,
        paper_name=None,
    )

    assert result == (
        first,
        second,
    )


def test_stage6_reference_matches_paper_discovery_mode(
    tmp_path,
    monkeypatch,
) -> None:
    discovered = (
        tmp_path
        / "a"
        / "reference_matches.json",
        tmp_path
        / "b"
        / "reference_matches.json",
    )

    calls = []

    def fake_discover(
        root,
        paper_name,
    ):
        calls.append(
            (
                root,
                paper_name,
            )
        )
        return discovered

    monkeypatch.setattr(
        cli,
        "discover_reference_match_artifacts",
        fake_discover,
    )

    result = cli._resolve_reference_match_paths(
        explicit_paths=None,
        search_root=tmp_path,
        paper_name="Example Paper",
    )

    assert result == discovered
    assert calls == [
        (
            tmp_path,
            "Example Paper",
        )
    ]


def test_stage6_reference_matches_rejects_mixed_modes(
    tmp_path,
) -> None:
    with pytest.raises(
        ValueError,
        match="either explicit",
    ):
        cli._resolve_reference_match_paths(
            explicit_paths=[
                tmp_path
                / "reference_matches.json"
            ],
            search_root=tmp_path,
            paper_name="Example Paper",
        )


def test_stage6_reference_matches_requires_complete_discovery_pair(
    tmp_path,
) -> None:
    with pytest.raises(
        ValueError,
        match="requires both",
    ):
        cli._resolve_reference_match_paths(
            explicit_paths=None,
            search_root=tmp_path,
            paper_name=None,
        )

    with pytest.raises(
        ValueError,
        match="requires both",
    ):
        cli._resolve_reference_match_paths(
            explicit_paths=None,
            search_root=None,
            paper_name="Example Paper",
        )


def test_stage6_reference_matches_requires_an_input_mode() -> None:
    with pytest.raises(
        ValueError,
        match="requires either",
    ):
        cli._resolve_reference_match_paths(
            explicit_paths=None,
            search_root=None,
            paper_name=None,
        )
