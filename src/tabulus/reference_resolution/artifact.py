from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from tabulus.reference_resolution.finalize import (
    ReferenceResolutionTrace,
)
from tabulus.reference_resolution.models import (
    ResolutionStatus,
)


REFERENCE_RESOLUTION_NAME = "reference_resolution.json"
REFERENCE_RESOLUTION_SCHEMA_VERSION = 1

REFERENCE_RESOLUTION_CHECKPOINT_NAME = (
    "reference_resolution.checkpoint.json"
)
REFERENCE_RESOLUTION_CHECKPOINT_SCHEMA_VERSION = 1

_FINAL_STATUSES = (
    ResolutionStatus.VALIDATED_WITH_DOI,
    ResolutionStatus.VALIDATED_WITHOUT_DOI,
    ResolutionStatus.REJECTED,
)


def default_reference_resolution_path(
    artifact_root: Path,
) -> Path:
    """Return the canonical Stage 6 artifact path."""

    return (
        Path(artifact_root).expanduser()
        / "references"
        / REFERENCE_RESOLUTION_NAME
    )


def build_reference_resolution_payload(
    traces: Iterable[ReferenceResolutionTrace],
) -> dict[str, Any]:
    """Build the persisted Stage 6 artifact.

    The artifact contains one final decision per unique bibliography index.
    Only final scientific outcomes are allowed. Operational failures must be
    handled by the caller and must never be converted into ``rejected`` merely
    so that an artifact can be written.
    """

    ordered = sorted(
        tuple(traces),
        key=lambda trace: (
            trace.resolution.reference_index
        ),
    )

    seen: set[int] = set()

    status_counts = {
        status.value: 0
        for status in _FINAL_STATUSES
    }

    llm_adjudicated_count = 0
    retry_count = 0

    for trace in ordered:
        index = trace.resolution.reference_index

        if index in seen:
            raise ValueError(
                "Duplicate final Stage 6 bibliography index: "
                f"{index}"
            )

        seen.add(index)

        status = trace.resolution.status

        if status not in _FINAL_STATUSES:
            raise ValueError(
                "Final Stage 6 artifact cannot contain "
                f"status {status.value!r} for bibliography "
                f"index {index}."
            )

        status_counts[
            status.value
        ] += 1

        if (
            trace.first_llm_response is not None
            or trace.second_llm_response is not None
        ):
            llm_adjudicated_count += 1

        if trace.retry_used:
            retry_count += 1

    return {
        "schema_version": (
            REFERENCE_RESOLUTION_SCHEMA_VERSION
        ),
        "resolution_count": len(ordered),
        "status_counts": status_counts,
        "llm_adjudicated_count": (
            llm_adjudicated_count
        ),
        "retry_count": retry_count,
        "entries": [
            trace.to_dict()
            for trace in ordered
        ],
    }


def write_reference_resolution_json(
    traces: Iterable[ReferenceResolutionTrace],
    output_path: Path,
) -> Path:
    """Atomically write the Stage 6 reference-resolution artifact."""

    output_path = Path(
        output_path
    ).expanduser()

    payload = build_reference_resolution_payload(
        traces
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=(
                f".{output_path.name}."
            ),
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(
                handle.name
            )

            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
            )

            handle.write("\n")
            handle.flush()
            os.fsync(
                handle.fileno()
            )

        os.replace(
            temporary_path,
            output_path,
        )

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()

    return output_path


def write_reference_resolution_artifact(
    traces: Iterable[ReferenceResolutionTrace],
    artifact_root: Path,
) -> Path:
    """Write Stage 6 to its canonical ``references/`` location."""

    return write_reference_resolution_json(
        traces,
        default_reference_resolution_path(
            artifact_root
        ),
    )


def default_reference_resolution_checkpoint_path(
    artifact_root: Path,
) -> Path:
    """Return the internal resumable Stage 6 checkpoint path."""

    return (
        Path(artifact_root).expanduser()
        / "references"
        / REFERENCE_RESOLUTION_CHECKPOINT_NAME
    )


def build_reference_resolution_run_fingerprint(
    bibliography_path: Path,
    reference_matches_paths: Iterable[Path],
    *,
    reference_context_path: Path | None = None,
    resolver_configuration: str = "",
) -> str:
    """Fingerprint immutable file inputs for one Stage 6 run.

    Stage 5 artifact ordering is intentionally ignored. Credentials and
    provider secrets are never included.
    """

    bibliography_path = Path(
        bibliography_path
    ).expanduser()

    match_paths = tuple(
        Path(path).expanduser()
        for path in reference_matches_paths
    )

    if not match_paths:
        raise ValueError(
            "At least one Stage 5 reference_matches.json "
            "artifact is required for a Stage 6 fingerprint."
        )

    digest = hashlib.sha256()

    # A resumable checkpoint must not silently span two different
    # Stage 6 implementations. Hash all Python source files belonging
    # to the reference-resolution package. This is intentionally strict:
    # even a code edit between runs invalidates the old checkpoint.
    source_root = Path(__file__).resolve().parent

    source_files = tuple(
        sorted(
            source_root.glob("*.py"),
            key=lambda path: path.name,
        )
    )

    for source_path in source_files:
        digest.update(
            b"stage6_source\0"
        )
        digest.update(
            source_path.name.encode("utf-8")
        )
        digest.update(
            b"\0"
        )
        digest.update(
            hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest().encode("ascii")
        )
        digest.update(
            b"\0"
        )

    configuration = str(
        resolver_configuration
        or ""
    ).strip()

    digest.update(
        b"resolver_configuration\0"
    )
    digest.update(
        configuration.encode("utf-8")
    )
    digest.update(
        b"\0"
    )

    def file_digest(
        path: Path,
    ) -> str:
        if not path.is_file():
            raise FileNotFoundError(
                "Stage 6 fingerprint input does not exist: "
                f"{path}"
            )

        return hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

    digest.update(
        b"bibliography\0"
    )
    digest.update(
        file_digest(
            bibliography_path
        ).encode("ascii")
    )
    digest.update(
        b"\0"
    )

    # Order-independent Stage 5 input set.
    match_digests = sorted(
        file_digest(path)
        for path in match_paths
    )

    for value in match_digests:
        digest.update(
            b"reference_matches\0"
        )
        digest.update(
            value.encode("ascii")
        )
        digest.update(
            b"\0"
        )

    if reference_context_path is None:
        digest.update(
            b"reference_context\0disabled\0"
        )
    else:
        digest.update(
            b"reference_context\0"
        )
        digest.update(
            file_digest(
                Path(
                    reference_context_path
                ).expanduser()
            ).encode("ascii")
        )
        digest.update(
            b"\0"
        )

    return digest.hexdigest()


def _validated_target_indices(
    target_indices: Iterable[int],
) -> tuple[int, ...]:
    """Normalize and validate the exact Stage 6 target set."""

    values = tuple(
        sorted(
            set(
                target_indices
            )
        )
    )

    if not values:
        raise ValueError(
            "Stage 6 target-index set must not be empty."
        )

    for index in values:
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index <= 0
        ):
            raise ValueError(
                "Stage 6 target indices must be "
                "positive integers."
            )

    return values


def _validate_serialized_trace_entries(
    entries: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Validate already serialized final Stage 6 trace entries."""

    values = tuple(
        entries
    )

    for entry in values:
        if not isinstance(
            entry,
            dict,
        ):
            raise ValueError(
                "Serialized Stage 6 trace entries "
                "must be JSON objects."
            )

    ordered = tuple(
        sorted(
            values,
            key=lambda entry: (
                entry.get(
                    "resolution",
                    {},
                ).get(
                    "reference_index",
                    -1,
                )
            ),
        )
    )

    seen: set[int] = set()
    allowed_statuses = {
        status.value
        for status in _FINAL_STATUSES
    }

    for entry in ordered:
        resolution = entry.get(
            "resolution"
        )

        if not isinstance(
            resolution,
            dict,
        ):
            raise ValueError(
                "Serialized Stage 6 trace entry must "
                "contain a resolution object."
            )

        index = resolution.get(
            "reference_index"
        )

        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index <= 0
        ):
            raise ValueError(
                "Serialized Stage 6 bibliography indices "
                "must be positive integers."
            )

        if index in seen:
            raise ValueError(
                "Duplicate serialized Stage 6 "
                f"bibliography index: {index}"
            )

        seen.add(
            index
        )

        status = resolution.get(
            "status"
        )

        if status not in allowed_statuses:
            raise ValueError(
                "Serialized Stage 6 checkpoint entries "
                "must contain only final scientific "
                f"statuses; index {index} has "
                f"{status!r}."
            )

    return ordered


def build_reference_resolution_payload_from_entries(
    entries: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Build the canonical artifact from serialized final traces."""

    ordered = _validate_serialized_trace_entries(
        entries
    )

    status_counts = {
        status.value: 0
        for status in _FINAL_STATUSES
    }

    llm_adjudicated_count = 0
    retry_count = 0

    for entry in ordered:
        resolution = entry[
            "resolution"
        ]

        status_counts[
            resolution["status"]
        ] += 1

        if (
            entry.get(
                "first_llm_response"
            ) is not None
            or entry.get(
                "second_llm_response"
            ) is not None
        ):
            llm_adjudicated_count += 1

        if bool(
            entry.get(
                "retry_used",
                False,
            )
        ):
            retry_count += 1

    return {
        "schema_version": (
            REFERENCE_RESOLUTION_SCHEMA_VERSION
        ),
        "resolution_count": len(
            ordered
        ),
        "status_counts": status_counts,
        "llm_adjudicated_count": (
            llm_adjudicated_count
        ),
        "retry_count": retry_count,
        "entries": list(
            ordered
        ),
    }


def _write_json_payload_atomically(
    payload: dict[str, Any],
    output_path: Path,
) -> Path:
    """Atomically write one JSON object."""

    output_path = Path(
        output_path
    ).expanduser()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(
                handle.name
            )

            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(
                handle.fileno()
            )

        os.replace(
            temporary_path,
            output_path,
        )

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()

    return output_path


def write_reference_resolution_checkpoint(
    entries: Iterable[dict[str, Any]],
    artifact_root: Path,
    *,
    run_fingerprint: str,
    target_indices: Iterable[int],
) -> Path:
    """Persist completed Stage 6 work for safe resumption."""

    fingerprint = str(
        run_fingerprint
        or ""
    ).strip()

    if not fingerprint:
        raise ValueError(
            "Stage 6 checkpoint run_fingerprint "
            "must be non-empty."
        )

    targets = _validated_target_indices(
        target_indices
    )

    ordered = _validate_serialized_trace_entries(
        entries
    )

    completed_indices = {
        entry["resolution"][
            "reference_index"
        ]
        for entry in ordered
    }

    unknown = (
        completed_indices
        - set(targets)
    )

    if unknown:
        raise ValueError(
            "Stage 6 checkpoint contains bibliography "
            "indices outside the current target set: "
            f"{sorted(unknown)}"
        )

    payload = {
        "schema_version": (
            REFERENCE_RESOLUTION_CHECKPOINT_SCHEMA_VERSION
        ),
        "run_fingerprint": fingerprint,
        "target_count": len(
            targets
        ),
        "target_indices": list(
            targets
        ),
        "completed_count": len(
            ordered
        ),
        "entries": list(
            ordered
        ),
    }

    return _write_json_payload_atomically(
        payload,
        default_reference_resolution_checkpoint_path(
            artifact_root
        ),
    )


def load_reference_resolution_checkpoint(
    artifact_root: Path,
    *,
    expected_run_fingerprint: str,
    expected_target_indices: Iterable[int],
) -> tuple[dict[str, Any], ...]:
    """Load a matching resumable Stage 6 checkpoint.

    A missing checkpoint is represented by an empty tuple.
    """

    path = (
        default_reference_resolution_checkpoint_path(
            artifact_root
        )
    )

    if not path.exists():
        return ()

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "Invalid Stage 6 checkpoint JSON: "
            f"{path}"
        ) from error

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "Stage 6 checkpoint root must be "
            "a JSON object."
        )

    if (
        payload.get(
            "schema_version"
        )
        != REFERENCE_RESOLUTION_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported Stage 6 checkpoint "
            "schema version."
        )

    expected_fingerprint = str(
        expected_run_fingerprint
        or ""
    ).strip()

    if (
        payload.get(
            "run_fingerprint"
        )
        != expected_fingerprint
    ):
        raise ValueError(
            "Stage 6 checkpoint does not match "
            "the current run fingerprint."
        )

    expected_targets = (
        _validated_target_indices(
            expected_target_indices
        )
    )

    if payload.get(
        "target_indices"
    ) != list(
        expected_targets
    ):
        raise ValueError(
            "Stage 6 checkpoint target set does not "
            "match the current Stage 6 target set."
        )

    if payload.get(
        "target_count"
    ) != len(
        expected_targets
    ):
        raise ValueError(
            "Stage 6 checkpoint target_count does "
            "not match its target-index set."
        )

    entries = payload.get(
        "entries"
    )

    if not isinstance(
        entries,
        list,
    ):
        raise ValueError(
            "Stage 6 checkpoint must contain "
            "an entries list."
        )

    ordered = (
        _validate_serialized_trace_entries(
            entries
        )
    )

    if payload.get(
        "completed_count"
    ) != len(
        ordered
    ):
        raise ValueError(
            "Stage 6 checkpoint completed_count "
            "does not match its entries."
        )

    completed_indices = {
        entry["resolution"][
            "reference_index"
        ]
        for entry in ordered
    }

    unknown = (
        completed_indices
        - set(expected_targets)
    )

    if unknown:
        raise ValueError(
            "Stage 6 checkpoint contains bibliography "
            "indices outside the current target set."
        )

    return ordered


def remove_reference_resolution_checkpoint(
    artifact_root: Path,
) -> None:
    """Delete the internal checkpoint after final promotion."""

    path = (
        default_reference_resolution_checkpoint_path(
            artifact_root
        )
    )

    if path.exists():
        path.unlink()


def write_reference_resolution_entries_artifact(
    entries: Iterable[dict[str, Any]],
    artifact_root: Path,
) -> Path:
    """Write canonical Stage 6 from serialized final traces."""

    payload = (
        build_reference_resolution_payload_from_entries(
            entries
        )
    )

    return _write_json_payload_atomically(
        payload,
        default_reference_resolution_path(
            artifact_root
        ),
    )

