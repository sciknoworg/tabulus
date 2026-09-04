from __future__ import annotations

import argparse
from pathlib import Path

from tabulus import __version__
from tabulus.evaluation import (
    DEFAULT_NUMBER_THRESHOLD,
    DEFAULT_TEXT_THRESHOLD,
    SUPPORTED_TABLE_RECONSTRUCTION_METRICS,
    evaluate_table_reconstruction,
)
from tabulus.mineru.backends import (
    HYBRID_BACKEND,
    PIPELINE_BACKEND,
    resolve_backend,
)
from tabulus.mineru.runner import run_mineru
from tabulus.crop_inputs import resolve_crop_inputs
from tabulus.pdf_inputs import resolve_pdf_inputs
from tabulus.reconstruction_inputs import resolve_reconstruction_inputs
from tabulus.reference_tables import (
    REFERENCE_TABLE_CLASSIFICATION_NAME,
    classify_reconstruction_tables,
)
from tabulus.table_crops import export_mineru_table_crops
from tabulus.table_ocr import (
    create_table_ocr_adapter,
    list_table_ocr_adapters,
    run_table_ocr_batch,
)

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


def default_table_reconstruction_output_root(
    crop_root: Path,
    *,
    adapter_name: str,
) -> Path:
    """Return the default reconstruction output directory for one adapter."""

    return Path(crop_root) / "reconstructions" / adapter_name


def default_reference_table_classification_output(
    reconstruction_dir: Path,
) -> Path:
    """Return the default reference-table classification manifest path."""

    return (
        Path(reconstruction_dir)
        / REFERENCE_TABLE_CLASSIFICATION_NAME
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

    profile_input = profile.add_mutually_exclusive_group(required=True)

    profile_input.add_argument(
        "--pdf",
        type=Path,
        help="Process one PDF file.",
    )

    profile_input.add_argument(
        "--folder",
        type=Path,
        help=(
            "Process all PDF files directly inside this folder. "
            "Discovery is non-recursive and sorted by filename."
        ),
    )

    profile_input.add_argument(
        "--pdf-list",
        dest="pdf_list",
        type=Path,
        help=(
            "Process PDF paths listed in a UTF-8 text file, one path per "
            "line. Blank lines and lines beginning with # are ignored; "
            "relative paths are resolved relative to the list file."
        ),
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
            "Normalized table-crop handoff directory. For one PDF this is "
            "the exact output directory. For multiple PDFs it is treated as "
            "a parent directory and each paper writes to <out>/<PDF stem>/. "
            "If omitted, Tabulus uses the default per-paper output."
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

    adapter_names = tuple(
        spec.name
        for spec in list_table_ocr_adapters()
    )

    reconstruct_tables = subparsers.add_parser(
        "reconstruct-tables",
        help="Reconstruct all canonical table crops through one OCR adapter.",
    )

    reconstruct_input = reconstruct_tables.add_mutually_exclusive_group(
        required=True,
    )

    reconstruct_input.add_argument(
        "--crops",
        type=Path,
        help="Process one canonical table-crop directory.",
    )

    reconstruct_input.add_argument(
        "--crops-folder",
        dest="crops_folder",
        type=Path,
        help=(
            "Process all immediate child directories containing "
            "tables_index.json, sorted by directory name."
        ),
    )

    reconstruct_input.add_argument(
        "--crops-list",
        dest="crops_list",
        type=Path,
        help=(
            "Process canonical table-crop directories listed in a UTF-8 "
            "text file, one path per line. Blank lines and lines beginning "
            "with # are ignored; relative paths are resolved relative to "
            "the list file."
        ),
    )

    reconstruct_tables.add_argument(
        "--adapter",
        choices=adapter_names,
        default="paddleocr-vl",
        help="Table reconstruction adapter.",
    )

    reconstruct_tables.add_argument(
        "--device",
        default="cpu",
        help="Execution device passed to the adapter, for example cpu or gpu:0.",
    )

    reconstruct_tables.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Reconstruction output directory. For one crop root this is the "
            "exact output directory. For multiple crop roots it is treated "
            "as a parent and Tabulus writes to "
            "<out>/<crop-root-name>/<adapter>/. If omitted, each paper uses "
            "<crops>/reconstructions/<adapter>/."
        ),
    )

    evaluate_table_reconstruction_parser = subparsers.add_parser(
        "evaluate-table-reconstruction",
        help=(
            "Evaluate one reconstructed table prediction against a "
            "gold-standard CSV."
        ),
    )

    evaluate_table_reconstruction_parser.add_argument(
        "--gold",
        required=True,
        type=Path,
        help="Gold-standard table CSV.",
    )
    evaluate_table_reconstruction_parser.add_argument(
        "--prediction",
        required=True,
        type=Path,
        help="Reconstructed table prediction CSV.",
    )
    evaluate_table_reconstruction_parser.add_argument(
        "--metric",
        choices=SUPPORTED_TABLE_RECONSTRUCTION_METRICS,
        default="rms",
        help="Table-reconstruction evaluation metric. Default: rms.",
    )
    evaluate_table_reconstruction_parser.add_argument(
        "--text-threshold",
        type=float,
        default=DEFAULT_TEXT_THRESHOLD,
        help="RMS text similarity threshold. Default: 0.5.",
    )
    evaluate_table_reconstruction_parser.add_argument(
        "--number-threshold",
        type=float,
        default=DEFAULT_NUMBER_THRESHOLD,
        help="RMS numeric relative-error threshold. Default: 0.1.",
    )
    evaluate_table_reconstruction_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Optional evaluation JSON output path. If omitted, no "
            "evaluation artifact is written."
        ),
    )

    classify_reference_tables = subparsers.add_parser(
        "classify-reference-tables",
        help="Classify reconstructed tables for reference-like content.",
    )

    classify_input = (
        classify_reference_tables.add_mutually_exclusive_group(
            required=True,
        )
    )

    classify_input.add_argument(
        "--reconstruction",
        type=Path,
        help=(
            "Adapter reconstruction directory containing "
            "batch_summary.json and parsed/."
        ),
    )

    classify_input.add_argument(
        "--crops-folder",
        dest="crops_folder",
        type=Path,
        help=(
            "Process reconstructions for all canonical crop roots directly "
            "inside this folder, using --adapter to select the "
            "reconstruction directory."
        ),
    )

    classify_input.add_argument(
        "--reconstruction-list",
        dest="reconstruction_list",
        type=Path,
        help=(
            "Process reconstruction directories listed in a UTF-8 text "
            "file, one path per line. Blank lines and lines beginning with "
            "# are ignored; relative paths are resolved relative to the "
            "list file."
        ),
    )

    classify_reference_tables.add_argument(
        "--adapter",
        choices=adapter_names,
        default="paddleocr-vl",
        help=(
            "Reconstruction adapter to select when --crops-folder is used."
        ),
    )

    classify_reference_tables.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Classification manifest path. If omitted, Tabulus writes "
            "reference_table_classification.json inside the reconstruction "
            "directory."
        ),
    )

    return parser


def _profile_one_pdf(
    args: argparse.Namespace,
    *,
    pdf_path: Path,
    backend: str,
    batch_size: int,
) -> None:
    output_root = (
        args.out
        if args.out is not None
        else default_profile_output_root(
            pdf_path,
            profiler=args.profiler,
            backend=backend,
        )
    )

    if args.table_crops_out is None:
        table_crops_output = default_table_crops_output_root(pdf_path)
    elif batch_size == 1:
        table_crops_output = args.table_crops_out
    else:
        table_crops_output = (
            Path(args.table_crops_out) / pdf_path.stem
        )

    print()
    print("PDF profiling configuration:")
    print(f"  PDF: {pdf_path}")
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
            pdf_path=pdf_path,
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "profile":
        backend = select_backend(args.backend)
        pdf_paths = resolve_pdf_inputs(
            pdf=args.pdf,
            folder=args.folder,
            pdf_list=args.pdf_list,
        )

        print()
        print("PDF input selection:")
        print(f"  PDFs detected: {len(pdf_paths)}")
        for index, pdf_path in enumerate(pdf_paths, start=1):
            print(f"  {index}. {pdf_path}")

        for index, pdf_path in enumerate(pdf_paths, start=1):
            print()
            print(
                "============================================================"
            )
            print(
                f"Processing PDF {index}/{len(pdf_paths)}: "
                f"{pdf_path.name}"
            )
            print(
                "============================================================"
            )
            _profile_one_pdf(
                args,
                pdf_path=pdf_path,
                backend=backend,
                batch_size=len(pdf_paths),
            )

        print()
        print("PDF profiling batch completed:")
        print(f"  Papers processed: {len(pdf_paths)}")

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

    if args.command == "reconstruct-tables":
        crop_roots = resolve_crop_inputs(
            crops=args.crops,
            crops_folder=args.crops_folder,
            crops_list=args.crops_list,
        )

        adapter = create_table_ocr_adapter(
            args.adapter,
            device=args.device,
        )

        if not adapter.capabilities.supports_device(args.device):
            raise ValueError(
                f"Adapter {args.adapter!r} does not support device "
                f"{args.device!r}."
            )

        print()
        print("Table reconstruction input selection:")
        print(f"  Papers detected: {len(crop_roots)}")
        for index, crop_root in enumerate(crop_roots, start=1):
            print(f"  {index}. {crop_root}")

        totals = {
            "tables_requested": 0,
            "tables_ok": 0,
            "tables_empty": 0,
            "tables_error": 0,
            "prediction_csvs": 0,
        }

        for index, crop_root in enumerate(crop_roots, start=1):
            if args.out is None:
                output_dir = default_table_reconstruction_output_root(
                    crop_root,
                    adapter_name=args.adapter,
                )
            elif len(crop_roots) == 1:
                output_dir = args.out
            else:
                output_dir = (
                    Path(args.out)
                    / crop_root.name
                    / args.adapter
                )

            print()
            print(
                "============================================================"
            )
            print(
                f"Reconstructing paper {index}/{len(crop_roots)}: "
                f"{crop_root.name}"
            )
            print(
                "============================================================"
            )
            print()
            print("Table reconstruction configuration:")
            print(f"  Crops: {crop_root}")
            print(f"  Adapter: {args.adapter}")
            print(f"  Device: {args.device}")
            print(f"  Output: {output_dir}")

            result = run_table_ocr_batch(
                crop_root=crop_root,
                output_dir=output_dir,
                adapter=adapter,
            )

            print()
            print("Table reconstruction completed:")
            print(f"  Tables requested: {result.tables_requested}")
            print(f"  Tables ok: {result.tables_ok}")
            print(f"  Tables empty: {result.tables_empty}")
            print(f"  Tables error: {result.tables_error}")
            print(f"  Prediction CSVs: {result.prediction_csvs}")
            print(f"  Summary: {result.summary_path}")

            for key in totals:
                totals[key] += getattr(result, key)

        print()
        print("Table reconstruction batch completed:")
        print(f"  Papers processed: {len(crop_roots)}")
        print(f"  Adapter: {args.adapter}")
        print(f"  Device: {args.device}")
        print(f"  Tables requested: {totals['tables_requested']}")
        print(f"  Tables ok: {totals['tables_ok']}")
        print(f"  Tables empty: {totals['tables_empty']}")
        print(f"  Tables error: {totals['tables_error']}")
        print(f"  Prediction CSVs: {totals['prediction_csvs']}")
        return

    if args.command == "evaluate-table-reconstruction":
        result = evaluate_table_reconstruction(
            args.gold,
            args.prediction,
            metric=args.metric,
            text_threshold=args.text_threshold,
            number_threshold=args.number_threshold,
        )

        output_path = None
        if args.out is not None:
            output_path = result.write_json(args.out)

        print()
        print("Table reconstruction evaluation completed:")
        print(f"  Gold: {result.gold_csv}")
        print(f"  Prediction: {result.prediction_csv}")
        print(
            "  Metric: "
            f"{result.metric_name} ({result.metric_short_name})"
        )
        print(f"  Score scale: {result.score_scale}")
        print(f"  RMS precision: {result.precision:.6f}")
        print(f"  RMS recall: {result.recall:.6f}")
        print(f"  RMS F1: {result.f1:.6f}")
        if output_path is not None:
            print(f"  Evaluation JSON: {output_path}")

        return

    if args.command == "classify-reference-tables":
        reconstruction_dirs = resolve_reconstruction_inputs(
            reconstruction=args.reconstruction,
            crops_folder=args.crops_folder,
            reconstruction_list=args.reconstruction_list,
            adapter_name=args.adapter,
        )

        if args.out is not None and len(reconstruction_dirs) != 1:
            raise ValueError(
                "--out can only be used when exactly one reconstruction "
                "directory is selected. Multi-paper classification writes "
                "the default manifest inside each reconstruction directory."
            )

        print()
        print("Reference-table classification input selection:")
        print(f"  Reconstructions detected: {len(reconstruction_dirs)}")
        for index, reconstruction_dir in enumerate(
            reconstruction_dirs,
            start=1,
        ):
            print(f"  {index}. {reconstruction_dir}")

        total_tables = 0
        total_reference_tables = 0

        for index, reconstruction_dir in enumerate(
            reconstruction_dirs,
            start=1,
        ):
            output_path = (
                args.out
                if args.out is not None
                else default_reference_table_classification_output(
                    reconstruction_dir
                )
            )

            print()
            print("============================================================")
            print(
                "Classifying reconstruction "
                f"{index}/{len(reconstruction_dirs)}: "
                f"{reconstruction_dir}"
            )
            print("============================================================")
            print()
            print("Reference-table classification configuration:")
            print(f"  Reconstruction: {reconstruction_dir}")
            print(f"  Output: {output_path}")

            result = classify_reconstruction_tables(
                reconstruction_dir,
                output_path=output_path,
            )

            print()
            print("Reference-table classification completed:")
            print(f"  Tables considered: {result.tables_considered}")
            print(
                "  Reference tables found: "
                f"{result.reference_tables_found}"
            )
            print(f"  Manifest: {result.output_path}")

            total_tables += result.tables_considered
            total_reference_tables += result.reference_tables_found

        print()
        print("Reference-table classification batch completed:")
        print(
            f"  Reconstructions processed: {len(reconstruction_dirs)}"
        )
        print(f"  Tables considered: {total_tables}")
        print(f"  Reference tables found: {total_reference_tables}")
        print(
            "  Non-reference tables: "
            f"{total_tables - total_reference_tables}"
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
