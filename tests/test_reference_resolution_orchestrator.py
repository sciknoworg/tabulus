from __future__ import annotations

from pathlib import Path

import pytest

import tabulus.reference_resolution.orchestrator as orchestrator_module
from tabulus.reference_resolution import (
    CrossrefAssessment,
    CrossrefAssessmentStatus,
    ReferenceEvidence,
    ReferenceResolution,
    ReferenceResolutionRunResult,
    ReferenceResolutionTrace,
    ResolutionStatus,
    ScholarlyResolution,
    ScholarlyResolutionStatus,
    resolve_reference_artifact,
    resolve_reference_artifact_with_clients,
)


def _evidence(
    index: int = 7,
) -> ReferenceEvidence:
    return ReferenceEvidence(
        reference_index=index,
        raw_reference=f"Reference {index}.",
    )


def _scholarly(
    index: int = 7,
) -> ScholarlyResolution:
    crossref = CrossrefAssessment(
        reference_index=index,
        status=CrossrefAssessmentStatus.NEEDS_CORE,
        selected_candidate=None,
        selected_score=None,
        ranked_candidates=(),
        reason="Test.",
    )

    return ScholarlyResolution(
        reference_index=index,
        raw_reference=f"Reference {index}.",
        status=ScholarlyResolutionStatus.NEEDS_LLM,
        selected_candidate=None,
        selected_score=None,
        crossref_assessment=crossref,
        core_assessment=None,
        reason="Test.",
    )


def _trace(
    index: int = 7,
    status: ResolutionStatus = ResolutionStatus.REJECTED,
) -> ReferenceResolutionTrace:
    resolution = ReferenceResolution(
        reference_index=index,
        raw_reference=f"Reference {index}.",
        status=status,
        canonical_doi=(
            f"10.1000/{index}"
            if status
            == ResolutionStatus.VALIDATED_WITH_DOI
            else ""
        ),
        canonical_title="",
        canonical_authors=(),
        canonical_year=None,
        canonical_venue="",
        source="",
        confidence=None,
        reason="Test final decision.",
    )

    return ReferenceResolutionTrace(
        resolution=resolution,
        initial_scholarly_resolution=_scholarly(
            index
        ),
    )


def test_resolve_reference_artifact_with_clients_runs_full_pipeline(
    monkeypatch,
    tmp_path,
) -> None:
    target = _evidence()

    calls = {}

    def fake_collect(
        bibliography_path,
        reference_matches_paths,
    ):
        calls["bibliography"] = bibliography_path
        calls["matches"] = reference_matches_paths
        return (target,)

    def fake_retrieve(
        targets,
        client,
    ):
        calls["retrieval_targets"] = targets
        calls["crossref_client"] = client
        return ("retrieval",)

    def fake_scholarly(
        targets,
        retrievals,
        client,
    ):
        calls["scholarly_targets"] = targets
        calls["retrievals"] = retrievals
        calls["core_client"] = client
        return (_scholarly(),)

    def fake_finalize(
        evidence,
        scholarly,
        *,
        crossref_client,
        core_client,
        llm_client,
    ):
        calls["final_evidence"] = evidence
        calls["final_scholarly"] = scholarly
        calls["final_crossref"] = crossref_client
        calls["final_core"] = core_client
        calls["llm_client"] = llm_client
        return _trace(
            status=ResolutionStatus.REJECTED
        )

    monkeypatch.setattr(
        orchestrator_module,
        "collect_resolution_targets",
        fake_collect,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "build_reference_resolution_run_fingerprint",
        lambda *args, **kwargs: "test-fingerprint",
    )
    monkeypatch.setattr(
        orchestrator_module,
        "retrieve_crossref_evidence",
        fake_retrieve,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "resolve_crossref_then_core",
        fake_scholarly,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "finalize_reference_resolution",
        fake_finalize,
    )

    crossref = object()
    core = object()
    llm = object()

    result = resolve_reference_artifact_with_clients(
        tmp_path / "references" / "bibliography.json",
        [
            tmp_path / "adapter-a" / "reference_matches.json",
            tmp_path / "adapter-b" / "reference_matches.json",
        ],
        tmp_path,
        crossref_client=crossref,
        core_client=core,
        llm_client=llm,
    )

    assert result.target_count == 1
    assert result.rejected == 1
    assert result.validated_with_doi == 0
    assert result.validated_without_doi == 0

    assert result.output_path == (
        tmp_path
        / "references"
        / "reference_resolution.json"
    )
    assert result.output_path.is_file()

    assert calls["crossref_client"] is crossref
    assert calls["core_client"] is core
    assert calls["llm_client"] is llm


def test_resolve_reference_artifact_constructs_standard_clients(
    monkeypatch,
) -> None:
    calls = {}

    crossref = object()
    core = object()
    llm = object()

    def fake_crossref_client(**kwargs):
        calls["crossref_kwargs"] = kwargs
        return crossref

    def fake_core_client(**kwargs):
        calls["core_kwargs"] = kwargs
        return core

    def fake_llm_client(**kwargs):
        calls["llm_kwargs"] = kwargs
        return llm

    expected = ReferenceResolutionRunResult(
        output_path=Path(
            "out/references/reference_resolution.json"
        ),
        target_count=0,
        validated_with_doi=0,
        validated_without_doi=0,
        rejected=0,
        llm_adjudicated_count=0,
        retry_count=0,
    )

    def fake_run(
        bibliography_path,
        reference_matches_paths,
        artifact_root,
        *,
        crossref_client,
        core_client,
        llm_client,
        resolver_configuration,
    ):
        calls["run"] = {
            "bibliography_path": bibliography_path,
            "reference_matches_paths": tuple(
                reference_matches_paths
            ),
            "artifact_root": artifact_root,
            "crossref_client": crossref_client,
            "core_client": core_client,
            "llm_client": llm_client,
        }
        return expected

    monkeypatch.setattr(
        orchestrator_module,
        "CrossrefClient",
        fake_crossref_client,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "CoreClient",
        fake_core_client,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "OpenAICompatibleLLMClient",
        fake_llm_client,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "resolve_reference_artifact_with_clients",
        fake_run,
    )

    result = resolve_reference_artifact(
        Path("bibliography.json"),
        [
            Path("a/reference_matches.json"),
            Path("b/reference_matches.json"),
        ],
        Path("out"),
        crossref_mailto="researcher@example.org",
        core_api_key="core-secret",
        llm_base_url="https://llm.example/v1",
        llm_api_key="llm-secret",
        llm_model="qwen3.6-35b-a3b",
    )

    assert result is expected

    assert calls["crossref_kwargs"] == {
        "mailto": "researcher@example.org",
    }
    assert calls["core_kwargs"] == {
        "api_key": "core-secret",
    }
    assert calls["llm_kwargs"] == {
        "base_url": "https://llm.example/v1",
        "api_key": "llm-secret",
        "model": "qwen3.6-35b-a3b",
        "enable_thinking": False,
        "provider_name": "kisski",
        "include_chat_template_kwargs": True,
    }


def test_resolve_reference_artifact_does_not_write_partial_on_failure(
    monkeypatch,
    tmp_path,
) -> None:
    def fail_collect(*args, **kwargs):
        raise RuntimeError(
            "simulated operational failure"
        )

    monkeypatch.setattr(
        orchestrator_module,
        "collect_resolution_targets",
        fail_collect,
    )

    with pytest.raises(
        RuntimeError,
        match="operational failure",
    ):
        resolve_reference_artifact_with_clients(
            tmp_path / "bibliography.json",
            [tmp_path / "reference_matches.json"],
            tmp_path,
            crossref_client=object(),
            core_client=object(),
            llm_client=object(),
        )

    assert not (
        tmp_path
        / "references"
        / "reference_resolution.json"
    ).exists()



def test_resolve_with_clients_passes_optional_document_contexts(
    monkeypatch,
    tmp_path,
) -> None:
    from tabulus.reference_resolution.reference_context import (
        ReferenceContext,
    )

    target = _evidence(
        7
    )

    context = ReferenceContext(
        page_index=3,
        block_index=16,
        block_type="paragraph",
        bbox=(10, 20, 30, 40),
        citation_marker="7",
        citation_count=1,
        text="Reference 7 is discussed here.",
    )

    context_path = (
        tmp_path
        / "references"
        / "reference_context.json"
    )

    calls = {}

    monkeypatch.setattr(
        orchestrator_module,
        "load_reference_context_artifact",
        lambda path: {
            7: (
                context,
            ),
            999: (),
        },
    )

    monkeypatch.setattr(
        orchestrator_module,
        "collect_resolution_targets",
        lambda bibliography_path, matches: (
            target,
        ),
    )

    monkeypatch.setattr(
        orchestrator_module,
        "build_reference_resolution_run_fingerprint",
        lambda *args, **kwargs: "test-fingerprint",
    )

    monkeypatch.setattr(
        orchestrator_module,
        "retrieve_crossref_evidence",
        lambda targets, client: (
            "retrieval",
        ),
    )

    monkeypatch.setattr(
        orchestrator_module,
        "resolve_crossref_then_core",
        lambda targets, retrievals, client: (
            _scholarly(
                7
            ),
        ),
    )

    def fake_finalize(
        evidence,
        scholarly,
        *,
        crossref_client,
        core_client,
        llm_client,
        document_contexts,
    ):
        calls[
            "document_contexts"
        ] = document_contexts

        return _trace(
            7,
            ResolutionStatus.REJECTED,
        )

    monkeypatch.setattr(
        orchestrator_module,
        "finalize_reference_resolution",
        fake_finalize,
    )

    result = resolve_reference_artifact_with_clients(
        tmp_path / "bibliography.json",
        [
            tmp_path
            / "reference_matches.json"
        ],
        tmp_path,
        crossref_client=object(),
        core_client=object(),
        llm_client=object(),
        reference_context_path=context_path,
    )

    assert result.target_count == 1

    assert calls[
        "document_contexts"
    ] == (
        context,
    )


def test_resolve_reference_artifact_forwards_optional_context_path(
    monkeypatch,
) -> None:
    context_path = Path(
        "references/reference_context.json"
    )

    crossref = object()
    core = object()
    llm = object()

    monkeypatch.setattr(
        orchestrator_module,
        "CrossrefClient",
        lambda **kwargs: crossref,
    )

    monkeypatch.setattr(
        orchestrator_module,
        "CoreClient",
        lambda **kwargs: core,
    )

    monkeypatch.setattr(
        orchestrator_module,
        "OpenAICompatibleLLMClient",
        lambda **kwargs: llm,
    )

    calls = {}

    expected = ReferenceResolutionRunResult(
        output_path=Path(
            "out/references/reference_resolution.json"
        ),
        target_count=0,
        validated_with_doi=0,
        validated_without_doi=0,
        rejected=0,
        llm_adjudicated_count=0,
        retry_count=0,
    )

    def fake_run(
        bibliography_path,
        reference_matches_paths,
        artifact_root,
        *,
        crossref_client,
        core_client,
        llm_client,
        reference_context_path,
        resolver_configuration,
    ):
        calls[
            "reference_context_path"
        ] = reference_context_path

        return expected

    monkeypatch.setattr(
        orchestrator_module,
        "resolve_reference_artifact_with_clients",
        fake_run,
    )

    result = resolve_reference_artifact(
        Path("bibliography.json"),
        [
            Path(
                "reference_matches.json"
            )
        ],
        Path("out"),
        crossref_mailto="researcher@example.org",
        core_api_key="core-secret",
        llm_base_url="https://llm.example/v1",
        llm_api_key="llm-secret",
        llm_model="qwen3.6-35b-a3b",
        reference_context_path=context_path,
    )

    assert result is expected

    assert calls[
        "reference_context_path"
    ] == context_path



def _checkpoint_entry_for_test(
    reference_index: int,
) -> dict:
    return {
        "resolution": {
            "reference_index": reference_index,
            "raw_reference": (
                f"Reference {reference_index}"
            ),
            "status": "validated_with_doi",
            "canonical_doi": (
                f"10.1000/{reference_index}"
            ),
            "canonical_title": "",
            "canonical_authors": [],
            "canonical_year": None,
            "canonical_venue": "",
            "source": "crossref",
            "confidence": 1.0,
            "reason": "",
        },
        "initial_scholarly_resolution": {},
        "first_llm_response": None,
        "retry_used": False,
        "retry_query": "",
        "retry_crossref_assessment": None,
        "retry_core_assessment": None,
        "second_llm_response": None,
    }


def _fake_trace_for_test(
    reference_index: int,
):
    from types import SimpleNamespace

    entry = _checkpoint_entry_for_test(
        reference_index
    )

    return SimpleNamespace(
        resolution=SimpleNamespace(
            reference_index=reference_index,
            status=SimpleNamespace(
                value="validated_with_doi"
            ),
        ),
        first_llm_response=None,
        second_llm_response=None,
        retry_used=False,
        to_dict=lambda: entry,
    )


def test_orchestrator_resumes_and_skips_completed_reference(
    tmp_path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    targets = (
        SimpleNamespace(
            reference_index=1,
        ),
        SimpleNamespace(
            reference_index=2,
        ),
    )

    crossref_calls = []
    finalized = []
    final_entries = []
    checkpoint_removed = []

    monkeypatch.setattr(
        orchestrator_module,
        "collect_resolution_targets",
        lambda bibliography, matches: targets,
    )

    monkeypatch.setattr(
        orchestrator_module,
        "build_reference_resolution_run_fingerprint",
        lambda *args, **kwargs: "fingerprint",
    )

    monkeypatch.setattr(
        orchestrator_module,
        "load_reference_resolution_checkpoint",
        lambda *args, **kwargs: (
            _checkpoint_entry_for_test(1),
        ),
    )

    def fake_retrieve(
        supplied_targets,
        client,
    ):
        supplied = tuple(
            supplied_targets
        )
        crossref_calls.extend(
            target.reference_index
            for target in supplied
        )
        return (
            SimpleNamespace(),
        )

    monkeypatch.setattr(
        orchestrator_module,
        "retrieve_crossref_evidence",
        fake_retrieve,
    )

    monkeypatch.setattr(
        orchestrator_module,
        "resolve_crossref_then_core",
        lambda supplied_targets, retrievals, client: (
            SimpleNamespace(
                reference_index=tuple(
                    supplied_targets
                )[0].reference_index
            ),
        ),
    )

    def fake_finalize(
        target,
        scholarly,
        **kwargs,
    ):
        finalized.append(
            target.reference_index
        )
        return _fake_trace_for_test(
            target.reference_index
        )

    monkeypatch.setattr(
        orchestrator_module,
        "finalize_reference_resolution",
        fake_finalize,
    )

    monkeypatch.setattr(
        orchestrator_module,
        "write_reference_resolution_checkpoint",
        lambda *args, **kwargs: tmp_path / "checkpoint.json",
    )

    def fake_final_writer(
        entries,
        artifact_root,
    ):
        final_entries.extend(
            entries
        )
        return (
            tmp_path
            / "references"
            / "reference_resolution.json"
        )

    monkeypatch.setattr(
        orchestrator_module,
        "write_reference_resolution_entries_artifact",
        fake_final_writer,
    )

    monkeypatch.setattr(
        orchestrator_module,
        "remove_reference_resolution_checkpoint",
        lambda root: checkpoint_removed.append(
            root
        ),
    )

    monkeypatch.setattr(
        orchestrator_module,
        "build_reference_resolution_payload_from_entries",
        lambda entries: {
            "status_counts": {
                "validated_with_doi": 2,
                "validated_without_doi": 0,
                "rejected": 0,
            },
            "llm_adjudicated_count": 0,
            "retry_count": 0,
        },
    )

    result = (
        orchestrator_module
        .resolve_reference_artifact_with_clients(
            tmp_path / "bibliography.json",
            (
                tmp_path / "matches.json",
            ),
            tmp_path,
            crossref_client=object(),
            core_client=object(),
            llm_client=object(),
        )
    )

    assert crossref_calls == [
        2,
    ]
    assert finalized == [
        2,
    ]

    assert [
        entry["resolution"][
            "reference_index"
        ]
        for entry in final_entries
    ] == [
        1,
        2,
    ]

    assert checkpoint_removed == [
        tmp_path
    ]

    assert result.target_count == 2
    assert result.validated_with_doi == 2


def test_orchestrator_flushes_checkpoint_before_operational_failure(
    tmp_path,
    monkeypatch,
) -> None:
    import pytest
    from types import SimpleNamespace

    targets = (
        SimpleNamespace(
            reference_index=1,
        ),
        SimpleNamespace(
            reference_index=2,
        ),
    )

    checkpoint_snapshots = []
    final_writer_called = []

    monkeypatch.setattr(
        orchestrator_module,
        "collect_resolution_targets",
        lambda bibliography, matches: targets,
    )

    monkeypatch.setattr(
        orchestrator_module,
        "build_reference_resolution_run_fingerprint",
        lambda *args, **kwargs: "fingerprint",
    )

    monkeypatch.setattr(
        orchestrator_module,
        "load_reference_resolution_checkpoint",
        lambda *args, **kwargs: (),
    )

    def fake_retrieve(
        supplied_targets,
        client,
    ):
        target = tuple(
            supplied_targets
        )[0]

        if target.reference_index == 2:
            raise RuntimeError(
                "provider unavailable"
            )

        return (
            SimpleNamespace(),
        )

    monkeypatch.setattr(
        orchestrator_module,
        "retrieve_crossref_evidence",
        fake_retrieve,
    )

    monkeypatch.setattr(
        orchestrator_module,
        "resolve_crossref_then_core",
        lambda supplied_targets, retrievals, client: (
            SimpleNamespace(
                reference_index=tuple(
                    supplied_targets
                )[0].reference_index
            ),
        ),
    )

    monkeypatch.setattr(
        orchestrator_module,
        "finalize_reference_resolution",
        lambda target, scholarly, **kwargs: (
            _fake_trace_for_test(
                target.reference_index
            )
        ),
    )

    def fake_checkpoint_writer(
        entries,
        artifact_root,
        **kwargs,
    ):
        checkpoint_snapshots.append(
            tuple(
                entry["resolution"][
                    "reference_index"
                ]
                for entry in entries
            )
        )
        return tmp_path / "checkpoint.json"

    monkeypatch.setattr(
        orchestrator_module,
        "write_reference_resolution_checkpoint",
        fake_checkpoint_writer,
    )

    monkeypatch.setattr(
        orchestrator_module,
        "write_reference_resolution_entries_artifact",
        lambda *args, **kwargs: (
            final_writer_called.append(
                True
            )
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="provider unavailable",
    ):
        (
            orchestrator_module
            .resolve_reference_artifact_with_clients(
                tmp_path / "bibliography.json",
                (
                    tmp_path / "matches.json",
                ),
                tmp_path,
                crossref_client=object(),
                core_client=object(),
                llm_client=object(),
            )
        )

    assert checkpoint_snapshots == [
        (
            1,
        )
    ]

    assert final_writer_called == []



def test_orchestrator_real_checkpoint_interrupt_and_resume(
    tmp_path,
    monkeypatch,
) -> None:
    import json
    import pytest
    from types import SimpleNamespace

    from tabulus.reference_resolution.artifact import (
        default_reference_resolution_checkpoint_path,
        default_reference_resolution_path,
    )

    bibliography = (
        tmp_path
        / "references"
        / "bibliography.json"
    )
    bibliography.parent.mkdir(
        parents=True
    )

    bibliography.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "index": 1,
                        "raw": "Reference 1",
                        "authors": [],
                        "year": None,
                    },
                    {
                        "index": 2,
                        "raw": "Reference 2",
                        "authors": [],
                        "year": None,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    matches = (
        tmp_path
        / "adapter"
        / "references"
        / "reference_matches.json"
    )
    matches.parent.mkdir(
        parents=True
    )

    matches.write_text(
        json.dumps(
            {
                "matched_tables": [
                    {
                        "matches": [
                            {
                                "matched_reference_indices": [
                                    1,
                                    2,
                                ]
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    calls = []
    fail_reference_two = {
        "enabled": True
    }

    def fake_retrieve(
        supplied_targets,
        client,
    ):
        target = tuple(
            supplied_targets
        )[0]

        calls.append(
            target.reference_index
        )

        if (
            target.reference_index == 2
            and fail_reference_two["enabled"]
        ):
            raise RuntimeError(
                "synthetic provider interruption"
            )

        return (
            SimpleNamespace(),
        )

    monkeypatch.setattr(
        orchestrator_module,
        "retrieve_crossref_evidence",
        fake_retrieve,
    )

    monkeypatch.setattr(
        orchestrator_module,
        "resolve_crossref_then_core",
        lambda supplied_targets, retrievals, client: (
            SimpleNamespace(
                reference_index=tuple(
                    supplied_targets
                )[0].reference_index
            ),
        ),
    )

    monkeypatch.setattr(
        orchestrator_module,
        "finalize_reference_resolution",
        lambda target, scholarly, **kwargs: (
            _fake_trace_for_test(
                target.reference_index
            )
        ),
    )

    # --------------------------------------------------------
    # First invocation: ref 1 completes, ref 2 fails.
    # --------------------------------------------------------

    with pytest.raises(
        RuntimeError,
        match="synthetic provider interruption",
    ):
        (
            orchestrator_module
            .resolve_reference_artifact_with_clients(
                bibliography,
                (
                    matches,
                ),
                tmp_path,
                crossref_client=object(),
                core_client=object(),
                llm_client=object(),
            )
        )

    checkpoint = (
        default_reference_resolution_checkpoint_path(
            tmp_path
        )
    )
    final_artifact = (
        default_reference_resolution_path(
            tmp_path
        )
    )

    assert checkpoint.is_file()
    assert not final_artifact.exists()

    checkpoint_payload = json.loads(
        checkpoint.read_text(
            encoding="utf-8"
        )
    )

    assert checkpoint_payload[
        "completed_count"
    ] == 1

    assert [
        entry["resolution"][
            "reference_index"
        ]
        for entry in checkpoint_payload[
            "entries"
        ]
    ] == [
        1
    ]

    assert calls == [
        1,
        2,
    ]

    # --------------------------------------------------------
    # Second invocation: same inputs, provider recovers.
    # Ref 1 must be loaded from disk and skipped.
    # --------------------------------------------------------

    fail_reference_two[
        "enabled"
    ] = False

    calls.clear()

    result = (
        orchestrator_module
        .resolve_reference_artifact_with_clients(
            bibliography,
            (
                matches,
            ),
            tmp_path,
            crossref_client=object(),
            core_client=object(),
            llm_client=object(),
        )
    )

    # Only the unfinished reference is queried again.
    assert calls == [
        2,
    ]

    assert result.target_count == 2
    assert result.validated_with_doi == 2
    assert result.rejected == 0

    assert final_artifact.is_file()

    final_payload = json.loads(
        final_artifact.read_text(
            encoding="utf-8"
        )
    )

    assert final_payload[
        "resolution_count"
    ] == 2

    assert [
        entry["resolution"][
            "reference_index"
        ]
        for entry in final_payload[
            "entries"
        ]
    ] == [
        1,
        2,
    ]

    # Successful promotion removes the internal work file.
    assert not checkpoint.exists()


def test_standard_orchestrator_builds_primary_fallback_llm(
    monkeypatch,
) -> None:
    import tabulus.reference_resolution.orchestrator as module

    calls = {
        "llm": [],
        "failover": None,
    }

    class DummyLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls["llm"].append(kwargs)

    class DummyFailover:
        def __init__(self, **kwargs):
            calls["failover"] = kwargs

    expected = ReferenceResolutionRunResult(
        output_path=Path(
            "out/references/reference_resolution.json"
        ),
        target_count=0,
        validated_with_doi=0,
        validated_without_doi=0,
        rejected=0,
        llm_adjudicated_count=0,
        retry_count=0,
    )

    monkeypatch.setattr(
        module,
        "CrossrefClient",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        module,
        "CoreClient",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        module,
        "OpenAICompatibleLLMClient",
        DummyLLM,
    )
    monkeypatch.setattr(
        module,
        "FailoverLLMClient",
        DummyFailover,
    )
    monkeypatch.setattr(
        module,
        "resolve_reference_artifact_with_clients",
        lambda *args, **kwargs: expected,
    )

    result = module.resolve_reference_artifact(
        Path("bibliography.json"),
        [Path("reference_matches.json")],
        Path("out"),
        crossref_mailto="researcher@example.org",
        core_api_key="core-secret",
        llm_base_url="https://kisski.example/v1",
        llm_api_key="kisski-secret",
        llm_model="qwen3.6-35b-a3b",
        fallback_llm_base_url=(
            "https://openrouter.ai/api/v1"
        ),
        fallback_llm_api_key="openrouter-secret",
        fallback_llm_model="qwen/qwen3.6-35b-a3b",
    )

    assert result is expected

    assert calls["llm"] == [
        {
            "base_url": "https://kisski.example/v1",
            "api_key": "kisski-secret",
            "model": "qwen3.6-35b-a3b",
            "enable_thinking": False,
            "provider_name": "kisski",
            "include_chat_template_kwargs": True,
        },
        {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "openrouter-secret",
            "model": "qwen/qwen3.6-35b-a3b",
            "enable_thinking": False,
            "provider_name": "openrouter",
            "include_chat_template_kwargs": False,
            "reasoning_effort": "none",
        },
    ]

    assert calls["failover"] is not None
    assert (
        calls["failover"]["primary_provider"]
        == "kisski"
    )
    assert (
        calls["failover"]["fallback_provider"]
        == "openrouter"
    )


def test_standard_orchestrator_rejects_partial_fallback_config() -> None:
    with pytest.raises(
        ValueError,
        match="base URL, API key, and model together",
    ):
        resolve_reference_artifact(
            Path("bibliography.json"),
            [Path("reference_matches.json")],
            Path("out"),
            crossref_mailto="researcher@example.org",
            core_api_key="core-secret",
            llm_base_url="https://kisski.example/v1",
            llm_api_key="kisski-secret",
            llm_model="qwen3.6-35b-a3b",
            fallback_llm_base_url=(
                "https://openrouter.ai/api/v1"
            ),
        )
