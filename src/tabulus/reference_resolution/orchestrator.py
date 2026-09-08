from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tabulus.reference_resolution.artifact import (
    build_reference_resolution_payload_from_entries,
    build_reference_resolution_run_fingerprint,
    load_reference_resolution_checkpoint,
    remove_reference_resolution_checkpoint,
    write_reference_resolution_checkpoint,
    write_reference_resolution_entries_artifact,
)
from tabulus.reference_resolution.core import (
    CoreClient,
)
from tabulus.reference_resolution.crossref import (
    CrossrefClient,
)
from tabulus.reference_resolution.fallback import (
    resolve_crossref_then_core,
)
from tabulus.reference_resolution.finalize import (
    finalize_reference_resolution,
)
from tabulus.reference_resolution.llm_client import (
    FailoverLLMClient,
    OpenAICompatibleLLMClient,
)
from tabulus.reference_resolution.models import (
    ResolutionStatus,
)
from tabulus.reference_resolution.pipeline import (
    collect_resolution_targets,
    retrieve_crossref_evidence,
)
from tabulus.reference_resolution.reference_context import (
    load_reference_context_artifact,
)


LOGGER = logging.getLogger(__name__)

# Rewrite the growing checkpoint periodically during normal execution.
# Any operational exception flushes all newly completed references before
# propagating, so provider failures do not discard successfully finalized
# work. A hard process/machine failure can lose at most this many minus one.
REFERENCE_RESOLUTION_CHECKPOINT_INTERVAL = 10


@dataclass(frozen=True)
class ReferenceResolutionRunResult:
    """Summary of one completed Stage 6 run."""

    output_path: Path
    target_count: int
    validated_with_doi: int
    validated_without_doi: int
    rejected: int
    llm_adjudicated_count: int
    retry_count: int


def resolve_reference_artifact_with_clients(
    bibliography_path: Path,
    reference_matches_paths: Iterable[Path],
    artifact_root: Path,
    *,
    crossref_client,
    core_client,
    llm_client,
    reference_context_path: Path | None = None,
    resolver_configuration: str = "custom-clients",
) -> ReferenceResolutionRunResult:
    """Run Stage 6 using already configured scholarly/LLM clients.

    The Stage 5 match artifacts may come from multiple reconstruction
    adapters. Their linked bibliography indices are unioned before resolution,
    so each unique bibliography entry is resolved exactly once.

    The final artifact is written only after every target has reached a final
    scientific status. Operational failures therefore propagate without
    creating a misleading partial ``reference_resolution.json``.
    """

    match_paths = tuple(
        Path(path)
        for path in reference_matches_paths
    )

    if not match_paths:
        raise ValueError(
            "At least one Stage 5 reference_matches.json "
            "artifact is required."
        )

    document_contexts_by_index = (
        load_reference_context_artifact(
            Path(reference_context_path)
        )
        if reference_context_path is not None
        else {}
    )

    targets = collect_resolution_targets(
        Path(bibliography_path),
        match_paths,
    )

    LOGGER.info(
        "[Stage 6] Collected %d unique bibliography references",
        len(targets),
    )

    target_indices = tuple(
        target.reference_index
        for target in targets
    )

    run_fingerprint = (
        build_reference_resolution_run_fingerprint(
            Path(bibliography_path),
            match_paths,
            reference_context_path=(
                Path(reference_context_path)
                if reference_context_path is not None
                else None
            ),
            resolver_configuration=resolver_configuration,
        )
    )

    checkpoint_entries = (
        load_reference_resolution_checkpoint(
            Path(artifact_root),
            expected_run_fingerprint=run_fingerprint,
            expected_target_indices=target_indices,
        )
    )

    completed_by_index = {
        entry["resolution"]["reference_index"]: entry
        for entry in checkpoint_entries
    }

    if completed_by_index:
        LOGGER.info(
            "[Stage 6] Resuming checkpoint: %d/%d references complete",
            len(completed_by_index),
            len(targets),
        )
    else:
        LOGGER.info(
            "[Stage 6] No matching checkpoint; starting new run"
        )

    newly_completed_since_checkpoint = 0

    try:
        for position, target in enumerate(
            targets,
            start=1,
        ):
            index = target.reference_index

            if index in completed_by_index:
                LOGGER.info(
                    "[Resolve] %d/%d ref=%d -> checkpointed [skip]",
                    position,
                    len(targets),
                    index,
                )
                continue

            LOGGER.info(
                "[Resolve] %d/%d ref=%d",
                position,
                len(targets),
                index,
            )

            # Resolve one bibliography entry end-to-end. This execution
            # model makes completed work independently checkpointable.
            crossref_retrievals = retrieve_crossref_evidence(
                (target,),
                crossref_client,
            )

            scholarly_resolutions = resolve_crossref_then_core(
                (target,),
                crossref_retrievals,
                core_client,
            )

            if len(scholarly_resolutions) != 1:
                raise ValueError(
                    "Per-reference Stage 6 resolution must "
                    "produce exactly one scholarly resolution."
                )

            scholarly = scholarly_resolutions[0]

            if scholarly.reference_index != index:
                raise ValueError(
                    "Per-reference scholarly resolution index "
                    "does not match its Stage 6 target."
                )

            # Preserve the exact no-context execution path when the
            # optional artifact is absent. This remains the ablation
            # baseline.
            if reference_context_path is None:
                trace = finalize_reference_resolution(
                    target,
                    scholarly,
                    crossref_client=crossref_client,
                    core_client=core_client,
                    llm_client=llm_client,
                )
            else:
                trace = finalize_reference_resolution(
                    target,
                    scholarly,
                    crossref_client=crossref_client,
                    core_client=core_client,
                    llm_client=llm_client,
                    document_contexts=(
                        document_contexts_by_index.get(
                            index,
                            (),
                        )
                    ),
                )

            completed_by_index[
                index
            ] = trace.to_dict()

            newly_completed_since_checkpoint += 1

            llm_used = (
                trace.first_llm_response is not None
                or trace.second_llm_response is not None
            )

            suffix = ""

            if llm_used:
                suffix += " [LLM]"

            if trace.retry_used:
                suffix += " [retry]"

            LOGGER.info(
                "[Resolve] %d/%d ref=%d -> %s%s",
                position,
                len(targets),
                index,
                trace.resolution.status.value,
                suffix,
            )

            if (
                newly_completed_since_checkpoint
                >= REFERENCE_RESOLUTION_CHECKPOINT_INTERVAL
            ):
                write_reference_resolution_checkpoint(
                    completed_by_index.values(),
                    Path(artifact_root),
                    run_fingerprint=run_fingerprint,
                    target_indices=target_indices,
                )

                LOGGER.info(
                    "[Checkpoint] saved %d/%d completed references",
                    len(completed_by_index),
                    len(targets),
                )

                newly_completed_since_checkpoint = 0

    except BaseException:
        # Flush every successfully finalized reference before propagating
        # an operational failure or user interruption.
        if newly_completed_since_checkpoint:
            try:
                write_reference_resolution_checkpoint(
                    completed_by_index.values(),
                    Path(artifact_root),
                    run_fingerprint=run_fingerprint,
                    target_indices=target_indices,
                )

                LOGGER.info(
                    "[Checkpoint] saved %d/%d completed references "
                    "before interruption",
                    len(completed_by_index),
                    len(targets),
                )

            except Exception:
                LOGGER.exception(
                    "[Checkpoint] failed to persist completed "
                    "Stage 6 work during interruption"
                )

        raise

    completed_indices = set(
        completed_by_index
    )

    if completed_indices != set(
        target_indices
    ):
        missing = sorted(
            set(target_indices)
            - completed_indices
        )

        raise ValueError(
            "Stage 6 finished execution without final decisions "
            f"for bibliography indices: {missing}"
        )

    final_entries = tuple(
        completed_by_index[index]
        for index in target_indices
    )

    # The canonical artifact remains all-or-nothing. Only after every
    # paper-level target has reached a final scientific state do we promote
    # the accumulated serialized traces.
    output_path = (
        write_reference_resolution_entries_artifact(
            final_entries,
            Path(artifact_root),
        )
    )

    remove_reference_resolution_checkpoint(
        Path(artifact_root)
    )

    LOGGER.info(
        "[Stage 6] Final artifact written; checkpoint removed"
    )

    final_payload = (
        build_reference_resolution_payload_from_entries(
            final_entries
        )
    )

    status_counts = final_payload[
        "status_counts"
    ]

    return ReferenceResolutionRunResult(
        output_path=output_path,
        target_count=len(targets),
        validated_with_doi=status_counts[
            ResolutionStatus.VALIDATED_WITH_DOI.value
        ],
        validated_without_doi=status_counts[
            ResolutionStatus.VALIDATED_WITHOUT_DOI.value
        ],
        rejected=status_counts[
            ResolutionStatus.REJECTED.value
        ],
        llm_adjudicated_count=final_payload[
            "llm_adjudicated_count"
        ],
        retry_count=final_payload[
            "retry_count"
        ],
    )


def resolve_reference_artifact(
    bibliography_path: Path,
    reference_matches_paths: Iterable[Path],
    artifact_root: Path,
    *,
    crossref_mailto: str,
    core_api_key: str,
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
    fallback_llm_base_url: str | None = None,
    fallback_llm_api_key: str | None = None,
    fallback_llm_model: str | None = None,
    reference_context_path: Path | None = None,
) -> ReferenceResolutionRunResult:
    """Run Stage 6 with the standard Tabulus clients."""

    crossref_client = CrossrefClient(
        mailto=crossref_mailto,
    )

    core_client = CoreClient(
        api_key=core_api_key,
    )

    fallback_values = (
        fallback_llm_base_url,
        fallback_llm_api_key,
        fallback_llm_model,
    )

    fallback_configured = all(
        value is not None
        and str(value).strip()
        for value in fallback_values
    )

    if any(
        value is not None
        and str(value).strip()
        for value in fallback_values
    ) and not fallback_configured:
        raise ValueError(
            "Fallback LLM configuration must provide "
            "base URL, API key, and model together."
        )

    primary_llm_client = OpenAICompatibleLLMClient(
        base_url=llm_base_url,
        api_key=llm_api_key,
        model=llm_model,
        enable_thinking=False,
        provider_name="kisski",
        include_chat_template_kwargs=True,
    )

    if fallback_configured:
        fallback_client = OpenAICompatibleLLMClient(
            base_url=str(
                fallback_llm_base_url
            ),
            api_key=str(
                fallback_llm_api_key
            ),
            model=str(
                fallback_llm_model
            ),
            enable_thinking=False,
            provider_name="openrouter",
            include_chat_template_kwargs=False,
            reasoning_effort="none",
        )

        llm_client = FailoverLLMClient(
            primary_client=primary_llm_client,
            fallback_client=fallback_client,
            primary_provider="kisski",
            fallback_provider="openrouter",
        )
    else:
        llm_client = primary_llm_client

    # Include only scientifically relevant public configuration.
    # Credentials and Crossref contact information must never enter
    # the checkpoint fingerprint.
    resolver_configuration = (
        "llm_policy=primary-first-per-adjudication;"
        "primary_provider=kisski;"
        f"primary_base_url={str(llm_base_url).rstrip('/')};"
        f"primary_model={str(llm_model).strip()};"
        "thinking=false"
    )

    if fallback_configured:
        resolver_configuration += (
            ";fallback_provider=openrouter;"
            "fallback_base_url="
            f"{str(fallback_llm_base_url).rstrip('/')};"
            "fallback_model="
            f"{str(fallback_llm_model).strip()}"
        )

    if reference_context_path is None:
        return resolve_reference_artifact_with_clients(
            bibliography_path,
            reference_matches_paths,
            artifact_root,
            crossref_client=crossref_client,
            core_client=core_client,
            llm_client=llm_client,
            resolver_configuration=resolver_configuration,
        )

    return resolve_reference_artifact_with_clients(
        bibliography_path,
        reference_matches_paths,
        artifact_root,
        crossref_client=crossref_client,
        core_client=core_client,
        llm_client=llm_client,
        reference_context_path=reference_context_path,
        resolver_configuration=resolver_configuration,
    )
