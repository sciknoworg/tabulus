#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 /path/to/input.pdf /path/to/output.txt <start-page>"
  exit 2
fi

IN_PDF="$1"
OUT_TXT="$2"
START_PAGE="$3"

if [[ ! -f "$IN_PDF" ]]; then
  echo "ERROR: input PDF not found: $IN_PDF"
  exit 1
fi

if ! [[ "$START_PAGE" =~ ^[0-9]+$ ]] || [[ "$START_PAGE" -lt 1 ]]; then
  echo "ERROR: start-page must be an integer >= 1"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHANDRA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_PATH="$CHANDRA_DIR/env"

mkdir -p "$CHANDRA_DIR/logs" "$CHANDRA_DIR/work/tmp" "$(dirname "$OUT_TXT")"

JOBSCRIPT="$(mktemp "$CHANDRA_DIR/work/tmp/pdf_to_txt_XXXXXX.sh")"

cat > "$JOBSCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=pdf_txt
#SBATCH --partition=kisski-h100
#SBATCH -G H100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH -C inet
#SBATCH --output=$CHANDRA_DIR/logs/pdf_txt_%j.log
#SBATCH --error=$CHANDRA_DIR/logs/pdf_txt_%j.err

set -euo pipefail

IN_PDF="$IN_PDF"
OUT_TXT="$OUT_TXT"
START_PAGE="$START_PAGE"
ENV_PATH="$ENV_PATH"

echo "Input: \$IN_PDF"
echo "Output: \$OUT_TXT"
echo "Start page: \$START_PAGE"
echo "Env: \$ENV_PATH"

module purge
module load miniforge3
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "\$ENV_PATH"

python -c "import pypdf" || { echo "ERROR: pypdf is not installed in \$ENV_PATH"; exit 1; }

python - <<PY
from pathlib import Path
import sys
from pypdf import PdfReader

in_pdf = Path("$IN_PDF")
out_txt = Path("$OUT_TXT")
start_page = int("$START_PAGE")

if not in_pdf.exists():
    print(f"ERROR: input PDF not found inside job: {in_pdf}", file=sys.stderr)
    sys.exit(1)

reader = PdfReader(str(in_pdf))
num_pages = len(reader.pages)

if start_page > num_pages:
    print(f"ERROR: start page {start_page} is larger than total pages {num_pages}", file=sys.stderr)
    sys.exit(1)

parts = []
for i in range(start_page - 1, num_pages):
    try:
        txt = reader.pages[i].extract_text() or ""
    except Exception as e:
        txt = f"\\n[ERROR extracting page {i+1}: {e}]\\n"
    parts.append(f"\\n\\n===== PAGE {i+1} =====\\n\\n{txt}")

out_txt.parent.mkdir(parents=True, exist_ok=True)
out_txt.write_text("".join(parts), encoding="utf-8")

print(f"TXT created at: {out_txt}")
print(f"Pages extracted: {start_page}-{num_pages}")
PY

if [[ ! -f "\$OUT_TXT" ]]; then
  echo "ERROR: TXT not created: \$OUT_TXT"
  exit 1
fi

echo "TXT created at: \$OUT_TXT"
EOF

chmod +x "$JOBSCRIPT"
sbatch "$JOBSCRIPT"