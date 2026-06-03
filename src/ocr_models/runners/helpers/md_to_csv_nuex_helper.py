from pathlib import Path
import csv
from bs4 import BeautifulSoup


BASE_DIR = Path(r"C:\Users\vruml\OneDrive\Рабочий стол\Neuer Ordner (4)")

DATASET_ROOT = BASE_DIR / "dataset"
OUTPUT_ROOT = BASE_DIR / "output"

TARGET_FOLDER_NAME = "NuExtract3_prediction"


def clean_cell(cell):
    # Replace <br> with space
    for br in cell.find_all("br"):
        br.replace_with(" ")

    text = cell.get_text(" ", strip=True)
    text = " ".join(text.split())
    return text


def html_table_to_matrix(table):
    rows = []
    rowspan_map = {}

    for r_idx, tr in enumerate(table.find_all("tr")):
        row = []
        col_idx = 0

        cells = tr.find_all(["td", "th"])

        for cell in cells:
            while (r_idx, col_idx) in rowspan_map:
                row.append(rowspan_map[(r_idx, col_idx)])
                col_idx += 1

            text = clean_cell(cell)

            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))

            for c in range(colspan):
                row.append(text if c == 0 else "")

                if rowspan > 1:
                    for rr in range(1, rowspan):
                        rowspan_map[(r_idx + rr, col_idx + c)] = text if c == 0 else ""

            col_idx += colspan

        while (r_idx, col_idx) in rowspan_map:
            row.append(rowspan_map[(r_idx, col_idx)])
            col_idx += 1

        rows.append(row)

    max_cols = max((len(r) for r in rows), default=0)

    # Normalize all rows to same length
    normalized = []
    for row in rows:
        normalized.append(row + [""] * (max_cols - len(row)))

    return normalized


def find_dataset_px_folder(px_name):
    matches = [
        p for p in DATASET_ROOT.rglob(px_name)
        if p.is_dir() and p.name == px_name
    ]
    return matches


def convert_md_file(md_file, target_dir):
    content = md_file.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(content, "html.parser")

    tables = soup.find_all("table")

    if not tables:
        print(f"No HTML table found: {md_file}")
        return 0

    created = 0

    for i, table in enumerate(tables, start=1):
        matrix = html_table_to_matrix(table)

        if not matrix:
            continue

        if len(tables) == 1:
            csv_name = md_file.stem + ".csv"
        else:
            csv_name = f"{md_file.stem}_table_{i:03d}.csv"

        csv_path = target_dir / csv_name

        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(matrix)

        created += 1

    return created


def main():
    total_created = 0

    for output_px in OUTPUT_ROOT.iterdir():
        if not output_px.is_dir():
            continue

        px_name = output_px.name  # P51, P52, ...

        dataset_matches = find_dataset_px_folder(px_name)

        if not dataset_matches:
            print(f"Dataset folder not found for: {px_name}")
            continue

        md_files = list(output_px.rglob("*.md"))

        if not md_files:
            print(f"No .md files found in: {output_px}")
            continue

        for dataset_px in dataset_matches:
            target_dir = dataset_px / "Ref_Tables" / TARGET_FOLDER_NAME
            target_dir.mkdir(parents=True, exist_ok=True)

            px_created = 0

            for md_file in md_files:
                px_created += convert_md_file(md_file, target_dir)

            total_created += px_created

            print(f"{px_name}: created {px_created} CSV files -> {target_dir}")

    print(f"\nFinished. Total CSV files created: {total_created}")


if __name__ == "__main__":
    main()