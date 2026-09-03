from pathlib import Path

DATASET_ROOT = Path(
    r"C:\Users\vruml\OneDrive\Рабочий стол\Neuer Ordner (4)\dataset"
)

count = 0

for folder in DATASET_ROOT.rglob("NuExtract3_prediction"):
    count += len(list(folder.glob("*.csv")))

print(f"NuExtract3 CSV files: {count}")