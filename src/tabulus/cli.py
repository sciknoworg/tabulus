from __future__ import annotations

import argparse
from pathlib import Path

from tabulus import __version__
from tabulus.mineru.backends import (
    HYBRID_BACKEND,
    PIPELINE_BACKEND,
    resolve_backend,
)
from tabulus.mineru.runner import run_mineru


def prompt_for_backend() -> str:
    """Prompt the user to choose a MinerU profiling backend."""

    print("Select PDF profiling backend:")
    print()
    print("  1. pipeline       CPU-compatible [default]")
    print("  2. hybrid-engine  GPU-accelerated")
    print()

    choice = input("Backend [1]: ").strip().lower()

    if choice in {"", "1", "pipeline"}:
        return PIPELINE_BACKEND

    if choice in {"2", "hybrid-engine", "hybrid"}:
        return HYBRID_BACKEND

    raise ValueError(
        f"Invalid backend selection: {choice!r}. "
        "Choose 1/pipeline or 2/hybrid-engine."
    )


def select_backend(requested_backend: str | None = None) -> str:
    """
    Select and validate the profiling backend.

    If hybrid-engine is requested but the GPU requirements are not met,
    explain the reason and fall back to pipeline.
    """

    requested = requested_backend or prompt_for_backend()

    resolved, capability = resolve_backend(requested)

    if requested == HYBRID_BACKEND and resolved == PIPELINE_BACKEND:
        reason = (
            capability.reason
            if capability is not None
            else "GPU requirements were not met."
        )

        print()
        print("hybrid-engine is unavailable:")
        print(f"  {reason}")
        print()
        print("Falling back to pipeline.")

    elif requested == HYBRID_BACKEND and capability is not None:
        print()
        print("GPU backend available:")
        print(f"  Device: {capability.device_name}")
        print(f"  VRAM: {capability.vram_gb:.1f} GB")
        print(
            "  Compute capability: "
            f"{capability.compute_capability[0]}."
            f"{capability.compute_capability[1]}"
        )

    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tabulus",
        description="Scientific PDF table extraction and enrichment pipeline.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
    )

    profile = subparsers.add_parser(
        "profile",
        help="Profile a scientific PDF using MinerU.",
    )

    profile.add_argument(
        "--pdf",
        required=True,
        type=Path,
        help="Input PDF file.",
    )

    profile.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory.",
    )

    profile.add_argument(
        "--backend",
        choices=[
            PIPELINE_BACKEND,
            HYBRID_BACKEND,
        ],
        default=None,
        help=(
            "MinerU backend. If omitted, Tabulus prompts interactively. "
            "pipeline is CPU-compatible; hybrid-engine requires a suitable GPU."
        ),
    )

    profile.add_argument(
        "--method",
        choices=["auto", "txt", "ocr"],
        default="auto",
        help="MinerU parsing method.",
    )

    profile.add_argument(
        "--effort",
        choices=["medium", "high"],
        default="high",
        help="Processing effort for hybrid-engine.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "profile":
        backend = select_backend(args.backend)

        print()
        print("PDF profiling configuration:")
        print(f"  PDF: {args.pdf}")
        print(f"  Output: {args.out}")
        print(f"  Backend: {backend}")
        print(f"  Method: {args.method}")

        run_mineru(
            pdf_path=args.pdf,
            output_dir=args.out,
            requested_backend=backend,
            effort=args.effort,
            method=args.method,
        )

        print()
        print(f"MinerU profiling completed: {args.out}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
