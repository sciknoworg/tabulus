from pathlib import Path
import camelot


def run(pdf_path: Path, out_dir: Path):
    """
    Run table extraction on a single PDF using Camelot and store results
    for later comparison and evaluation.
    """
    # Directory where extracted tables will be stored
    tables_dir = out_dir / "extracted_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Test both Camelot parsing strategies:
    # - stream: spacing-based parsing (better for text-based PDFs)
    # - lattice: line-based parsing (better for ruled tables)
    flavors = ["stream", "lattice"]

    # Notes file collects a short summary of what worked and what failed
    notes = ["- Library: Camelot", f"- PDF: {pdf_path.name}", "- Attempts:"]

    for flavor in flavors:
        try:
            # Run Camelot on all pages of the PDF using the selected flavor
            tables = camelot.read_pdf(
                str(pdf_path),
                pages="all",
                flavor=flavor
            )

            saved = 0
            for i, table in enumerate(tables, start=1):
                df = table.df

                # Skip empty or invalid tables
                if df is None or df.empty:
                    continue

                # Save each extracted table as CSV for manual inspection
                df.to_csv(
                    tables_dir / f"{flavor}_table_{i:03d}.csv",
                    index=False
                )
                saved += 1

            # Record how many tables were detected and saved for this flavor
            notes.append(f"  - {flavor}: extracted={len(tables)}, saved={saved}")

        except Exception as e:
            # Capture failures to keep the evaluation script robust
            notes.append(f"  - {flavor}: FAILED ({e})")

    # Write a short evaluation summary for this PDF and library
    (out_dir / "notes.md").write_text("\n".join(notes), encoding="utf-8")
