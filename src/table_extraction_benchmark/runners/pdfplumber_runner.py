from pathlib import Path
import pdfplumber


def run(pdf_path: Path, out_dir: Path):
    """
    Evaluate pdfplumber's table extraction on a PDF and store outputs for comparison.
    This runner tests two table-detection strategies ("lines" and "text") because
    pdfplumber does not provide 'stream'/'lattice' modes like Camelot/Tabula.
    """
    # Output directory for extracted tables
    tables_dir = out_dir / "extracted_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Two common pdfplumber strategies for table extraction:
    # - "lines": relies on detected ruling lines / edges
    # - "text" : infers structure from text alignment/positions
    strategies = {
        "lines": {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
        },
        "text": {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
        },
    }

    # notes.md summary for reproducibility and later ranking
    notes = [
        "- Library: pdfplumber",
        f"- PDF: {pdf_path.name}",
        "- Attempts:",
    ]

    # Open PDF once and iterate all pages (bulk-style evaluation)
    with pdfplumber.open(pdf_path) as pdf:
        for name, settings in strategies.items():
            print(f"[pdfplumber] strategy={name}")
            extracted = 0
            saved = 0

            for page_idx, page in enumerate(pdf.pages, start=1):
                # extract_tables returns a list of tables (each table is a list of rows)
                tables = page.extract_tables(table_settings=settings)
                if not tables:
                    continue

                extracted += len(tables)

                for t_idx, table in enumerate(tables, start=1):
                    if not table:
                        continue

                    # Save as CSV-like text (simple, dependency-free output)
                    out_file = tables_dir / f"{name}_p{page_idx:03d}_t{t_idx:02d}.csv"
                    with out_file.open("w", encoding="utf-8") as f:
                        for row in table:
                            # Replace None cells with empty strings to keep CSV rectangular
                            f.write(",".join(cell or "" for cell in row) + "\n")
                    saved += 1

            notes.append(f"  - {name}: extracted={extracted}, saved={saved}")

    (out_dir / "notes.md").write_text("\n".join(notes), encoding="utf-8")
