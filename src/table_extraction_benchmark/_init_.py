from pathlib import Path
import tabula

def _save_tables(tables, tables_dir: Path):
    count = 0
    for i, df in enumerate(tables or [], start=1):
        if df is None or df.empty:
            continue
        out_file = tables_dir / f"table_{i:03d}.csv"
        df.to_csv(out_file, index=False)
        count += 1
    return count

def run(pdf_path: Path, out_dir: Path):
    tables_dir = out_dir / "extracted_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # P2 known table pages (Tabula uses 1-based page numbers)
    pages_to_try = "4,10"

    tries = [
        ("stream", dict(stream=True)),
        ("lattice", dict(lattice=True)),
        ("default", dict()),
    ]

    total_saved = 0
    notes_lines = [
        "- Library: Tabula",
        f"- PDF: {pdf_path.name}",
        f"- Pages tried: {pages_to_try}",
        "- Attempts:"
    ]

    for name, opts in tries:
        tables = tabula.read_pdf(
            str(pdf_path),
            pages=pages_to_try,
            multiple_tables=True,
            force_subprocess=True,   # avoids JPype issues on your setup
            **opts
        )
        saved = _save_tables(tables, tables_dir)
        notes_lines.append(f"  - {name}: extracted={0 if tables is None else len(tables)}, saved_nonempty={saved}")
        total_saved += saved

        # If we got something usable, stop early
        if saved > 0:
            break

    (out_dir / "notes.md").write_text("\n".join(notes_lines) + "\n", encoding="utf-8")
    print(f"[Tabula] Saved {total_saved} non-empty tables to {tables_dir}")
