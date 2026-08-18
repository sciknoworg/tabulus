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
from tabulus.table_crops import export_mineru_table_crops

DEFAULT_PROFILER = "mineru"
SUPPORTED_PROFILERS = (DEFAULT_PROFILER,)

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

def default_profile_output_root(
    pdf_path: Path,
    *,
    profiler: str,
    backend: str,
) -> Path:
    """Return the default output root for a profiling backend."""

    pdf_path = Path(pdf_path).resolve()

    return (
        pdf_path.parent
        / "tabulus-output"
        / profiler
        / backend
    )

def default_table_crops_output_root(pdf_path: Path) -> Path:
    """Return the default normalized table-crop handoff directory."""

    pdf_path = Path(pdf_path).resolve()

    return (
        pdf_path.parent
        / "tabulus-output"
        / "table-crops"
        / pdf_path.stem
    )


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
        help="Profile a scientific PDF.",
    )

    profile.add_argument(
        "--pdf",
        required=True,
        type=Path,
        help="Input PDF file.",
    )

    profile.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Profiling output root. If omitted, Tabulus writes to "
            "<PDF directory>/tabulus-output/<profiler>/<backend>/."
        ),
    )

    profile.add_argument(
        "--profiler",
        choices=SUPPORTED_PROFILERS,
        default=DEFAULT_PROFILER,
        help="PDF profiling tool. Currently supported: mineru.",
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

    profile.add_argument(
        "--table-crops-out",
        type=Path,
        default=None,
        help=(
            "Normalized table-crop handoff directory. If omitted, Tabulus "
            "writes to <PDF directory>/tabulus-output/table-crops/<PDF stem>/."
        ),
    )

    profile.add_argument(
        "--no-export-table-crops",
        dest="export_table_crops",
        action="store_false",
        help=(
            "Skip automatic export of MinerU-detected tables into the "
            "normalized Tabulus table-crop handoff."
        ),
    )

    profile.set_defaults(export_table_crops=True)

    export_table_crops = subparsers.add_parser(
        "export-table-crops",
        help="Export MinerU table crops into a normalized handoff directory.",
    )

    export_table_crops.add_argument(
        "--mineru-root",
        required=True,
        type=Path,
        help="Existing MinerU output directory.",
    )

    export_table_crops.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Table-crop handoff output directory.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "profile":
        backend = select_backend(args.backend)

        output_root = (
            args.out
            if args.out is not None
            else default_profile_output_root(
                args.pdf,
                profiler=args.profiler,
                backend=backend,
            )
        )

        table_crops_output = (
            args.table_crops_out
            if args.table_crops_out is not None
            else default_table_crops_output_root(args.pdf)
        )

        print()
        print("PDF profiling configuration:")
        print(f"  PDF: {args.pdf}")
        print(f"  Profiler: {args.profiler}")
        print(f"  Backend: {backend}")
        print(f"  Method: {args.method}")
        print(f"  Output root: {output_root}")
        print(
            "  Export table crops: "
            f"{'yes' if args.export_table_crops else 'no'}"
        )
        if args.export_table_crops:
            print(f"  Table-crops output: {table_crops_output}")

        if args.profiler == "mineru":
            run_dir = run_mineru(
                pdf_path=args.pdf,
                output_dir=output_root,
                requested_backend=backend,
                effort=args.effort,
                method=args.method,
            )
        else:
            raise ValueError(
                f"Unsupported profiler: {args.profiler}"
            )

        print()
        print(f"PDF profiling completed: {run_dir}")

        if args.export_table_crops:
            crop_result = export_mineru_table_crops(
                mineru_output_dir=run_dir,
                output_dir=table_crops_output,
            )

            print()
            print("Canonical table-crop export completed:")
            print(f"  Tables found: {crop_result.tables_found}")
            print(f"  Crops saved: {crop_result.crops_saved}")
            print(f"  Index: {crop_result.index_path}")

        return

    if args.command == "export-table-crops":
        result = export_mineru_table_crops(
            mineru_output_dir=args.mineru_root,
            output_dir=args.out,
        )

        print()
        print("Table-crop export completed:")
        print(f"  Tables found: {result.tables_found}")
        print(f"  Crops saved: {result.crops_saved}")
        print(f"  Index: {result.index_path}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
