from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from tabulus.mineru.backends import (
    DEFAULT_BACKEND,
    HYBRID_BACKEND,
    resolve_backend,
)

from tabulus.mineru.tables import find_content_list


def find_mineru_executable() -> str:
    """Return the MinerU executable available in the current environment."""

    executable = shutil.which("mineru")

    if executable is None:
        raise RuntimeError(
            "MinerU is not installed or `mineru` is not available on PATH."
        )

    return executable


def build_mineru_command(
    pdf_path: Path,
    output_dir: Path,
    *,
    backend: str,
    effort: str = "high",
    method: str = "auto",
    table: bool = True,
    formula: bool = False,
    image_analysis: bool = False,
) -> list[str]:
    """Build a MinerU CLI command without executing it."""

    command = [
        find_mineru_executable(),
        "-p",
        str(Path(pdf_path)),
        "-o",
        str(Path(output_dir)),
        "-b",
        backend,
        "-m",
        method,
        "-t",
        str(table).lower(),
        "-f",
        str(formula).lower(),
        "--image-analysis",
        str(image_analysis).lower(),
    ]

    if backend == HYBRID_BACKEND:
        command.extend(
            [
                "--effort",
                effort,
            ]
        )

    return command


def run_mineru(
    pdf_path: Path,
    output_dir: Path,
    *,
    requested_backend: str = DEFAULT_BACKEND,
    effort: str = "high",
    method: str = "auto",
) -> Path:
    """
    Run MinerU for one PDF.

    Backend resolution is handled before execution. If hybrid-engine is
    requested but the GPU requirements are not met, execution falls back
    to the pipeline backend.

    The function itself is non-interactive. Interactive backend selection
    belongs to the Tabulus CLI.
    """

    pdf_path = Path(pdf_path).resolve()
    output_dir = Path(output_dir).resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"Input PDF not found: {pdf_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    backend, capability = resolve_backend(
        requested_backend
    )

    fallback_reason = None

    if backend != requested_backend:
        fallback_reason = (
            capability.reason
            if capability is not None
            else "Requested backend requirements were not met."
        )

    command = build_mineru_command(
        pdf_path,
        output_dir,
        backend=backend,
        effort=effort,
        method=method,
        table=True,
        formula=False,
        image_analysis=False,
    )

    document_dir = output_dir / pdf_path.stem

    before_content_lists = (
        {
            path.resolve(): path.stat().st_mtime_ns
            for path in document_dir.rglob("*_content_list.json")
        }
        if document_dir.exists()
        else {}
    )

    started = time.perf_counter()

    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=os.environ.copy(),
        shell=False,
    )

    elapsed = time.perf_counter() - started

    if process.returncode == 0:
        after_content_lists = list(
            document_dir.rglob("*_content_list.json")
        )

        changed_content_lists = [
            path
            for path in after_content_lists
            if (
                path.resolve() not in before_content_lists
                or path.stat().st_mtime_ns
                != before_content_lists[path.resolve()]
            )
        ]

        if len(changed_content_lists) == 1:
            run_dir = changed_content_lists[0].parent
        elif len(after_content_lists) == 1:
            run_dir = after_content_lists[0].parent
        else:
            raise RuntimeError(
                "Could not uniquely determine the MinerU-native "
                f"run directory under {document_dir}."
            )
    else:
        # MinerU may fail before creating its native output directory.
        # Store diagnostics at the document level in that case.
        run_dir = document_dir

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stdout_log = run_dir / "mineru_stdout.log"
    stderr_log = run_dir / "mineru_stderr.log"
    run_log = run_dir / "tabulus_run.txt"

    stdout_log.write_text(
        process.stdout or "",
        encoding="utf-8",
    )

    stderr_log.write_text(
        process.stderr or "",
        encoding="utf-8",
    )

    run_log.write_text(
        "\n".join(
            [
                f"requested_backend={requested_backend}",
                f"resolved_backend={backend}",
                f"fallback_reason={fallback_reason or ''}",
                f"method={method}",
                f"effort={effort if backend == HYBRID_BACKEND else ''}",
                f"duration_seconds={elapsed:.3f}",
                f"return_code={process.returncode}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    if process.returncode != 0:
        raise RuntimeError(
            "MinerU failed with exit code "
            f"{process.returncode}. "
            f"See {stderr_log}."
        )

    return run_dir
