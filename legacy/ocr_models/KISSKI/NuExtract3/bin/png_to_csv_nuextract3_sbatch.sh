#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 /path/to/input.png /path/to/output.csv"
  exit 2
fi

IN_PNG="$(readlink -f "$1")"
OUT_CSV="$(readlink -m "$2")"

BASE_DIR="/user/v.rumleanschi/u25807/NuExtract3"
ENV_PATH="$BASE_DIR/env"

mkdir -p "$BASE_DIR/logs"
mkdir -p "$(dirname "$OUT_CSV")"

JOBSCRIPT="$(mktemp "$BASE_DIR/tmp_nuextract3_XXXXXX.sh")"

cat > "$JOBSCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=nuextract3_csv
#SBATCH --partition=kisski-h100
#SBATCH -G H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -C inet
#SBATCH --output=$BASE_DIR/logs/nuextract3_%j.log
#SBATCH --error=$BASE_DIR/logs/nuextract3_%j.err

set -euo pipefail

module purge
module load miniforge3

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

export HF_HOME="$BASE_DIR/cache/huggingface"
export TRANSFORMERS_CACHE="\$HF_HOME"
export TORCH_HOME="$BASE_DIR/cache/torch"

export HF_TOKEN="HF_TOKEN"
export HUGGING_FACE_HUB_TOKEN="\$HF_TOKEN"

mkdir -p "\$HF_HOME"
mkdir -p "\$TORCH_HOME"

python "$BASE_DIR/run_nuextract3_png_to_csv.py" \
  --png "$IN_PNG" \
  --out "$OUT_CSV"
EOF

chmod +x "$JOBSCRIPT"
sbatch "$JOBSCRIPT"