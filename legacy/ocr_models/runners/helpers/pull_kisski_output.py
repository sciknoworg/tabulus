from pathlib import Path
import shutil

BASE_DIR = Path(r"C:\Users\vruml\OneDrive\Рабочий стол\Neuer Ordner (4)")

DATASET_ROOT = BASE_DIR / "dataset"
OUTPUT_ROOT = BASE_DIR / "output"

for output_px in OUTPUT_ROOT.iterdir():
    if not output_px.is_dir():
        continue

    px_name = output_px.name  # e.g. P51

    matches = [
        p for p in DATASET_ROOT.rglob(px_name)
        if p.is_dir() and p.name == px_name
    ]

    if not matches:
        print(f"Not found in dataset: {px_name}")
        continue

    for dataset_px in matches:
        target_dir = dataset_px / "Ref_Tables" / "NuExtract3_prediction"
        target_dir.mkdir(parents=True, exist_ok=True)

        csv_files = list(output_px.glob("*.csv"))

        for csv_file in csv_files:
            shutil.copy2(csv_file, target_dir / csv_file.name)

        print(f"Copied {len(csv_files)} CSV files: {px_name} -> {target_dir}")

print("Finished.")