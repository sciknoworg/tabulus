import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def get_runner(lib: str):
    if lib == "tabula":
        from table_extraction_benchmark.runners.tabula_runner import run
        return run
    if lib == "camelot":
        from table_extraction_benchmark.runners.camelot_runner import run
        return run
    if lib == "pdfplumber":
        from table_extraction_benchmark.runners.pdfplumber_runner import run
        return run
    if lib == "img2table":
        from table_extraction_benchmark.runners.img2table_runner import run
        return run
    if lib == "surya":
        from table_extraction_benchmark.runners.surya_runner import run
        return run
    if lib == "paddle":
        from table_extraction_benchmark.runners.paddle_runner import run
        return run

    raise ValueError(f"Unknown lib: {lib}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument(
        "--lib",
        required=True,
        choices=["tabula", "camelot", "pdfplumber","img2table","surya","paddle"]
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    pdf_name = pdf_path.stem

    out_dir = ROOT / "outputs" / pdf_name / args.lib
    runner = get_runner(args.lib)
    runner(pdf_path, out_dir)

if __name__ == "__main__":
    main()
