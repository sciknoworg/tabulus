from pathlib import Path
import tabula


def run(pdf_path: Path, out_dir: Path):
    """
    Run table extraction on a single PDF using Tabula and store results
    for later qualitative and quantitative evaluation.
    """
    # Directory for storing extracted table CSV files
    tables_dir = out_dir / "extracted_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Notes file records extraction outcomes per mode
    notes = [
        "- Library: Tabula",
        f"- PDF: {pdf_path.name}",
        "- Attempts:"
    ]

    # Tabula supports two main parsing strategies:
    # - stream: whitespace-based parsing
    # - lattice: line-based parsing (for ruled tables)
    modes = {
        "stream": {"stream": True},
        "lattice": {"lattice": True},
    }

    total_saved = 0

    for mode, opts in modes.items():
        # Console output helps track progress during batch execution
        print(f"[Tabula] Trying mode={mode} on {pdf_path.name} ...")

        try:
            # force_subprocess=True is important on Windows to avoid JPype JVM issues
            tables = tabula.read_pdf(
                str(pdf_path),
                pages="all",
                multiple_tables=True,
                force_subprocess=True,
                **opts
            )

            extracted = 0 if tables is None else len(tables)
            saved = 0

            for i, df in enumerate(tables or [], start=1):
                # Skip empty or invalid tables
                if df is None or df.empty:
                    continue

                # Save each extracted table for later inspection
                out_file = tables_dir / f"{mode}_table_{i:03d}.csv"
                df.to_csv(out_file, index=False)
                saved += 1

            # Record per-mode extraction statistics
            notes.append(f"  - {mode}: extracted={extracted}, saved_nonempty={saved}")
            print(f"[Tabula] mode={mode} extracted={extracted}, saved_nonempty={saved}")
            total_saved += saved

        except Exception as e:
            # Capture failures without stopping the overall evaluation
            notes.append(f"  - {mode}: FAILED ({type(e).__name__}: {e})")
            print(f"[Tabula] mode={mode} FAILED: {type(e).__name__}: {e}")

    # Persist notes for reproducibility and later comparison across libraries
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"[Tabula] Done. Total saved tables: {total_saved}")
