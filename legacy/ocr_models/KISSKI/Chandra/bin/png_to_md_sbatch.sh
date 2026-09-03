#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 /path/to/input.png /path/to/output.md"
  exit 2
fi

IN_PNG="$(readlink -f "$1")"
OUT_MD="$(readlink -m "$2")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHANDRA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_PATH="$CHANDRA_DIR/env"
RUNS_DIR="$CHANDRA_DIR/work/output/_runs"

mkdir -p "$RUNS_DIR" "$(dirname "$OUT_MD")" "$CHANDRA_DIR/logs" "$CHANDRA_DIR/work/tmp"

JOBSCRIPT="$(mktemp "$CHANDRA_DIR/work/tmp/png_to_md_XXXXXX.sh")"

cat > "$JOBSCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=chandra_md
#SBATCH --partition=kisski-h100
#SBATCH -G H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -C inet
#SBATCH --output=$CHANDRA_DIR/logs/chandra_md_%j.log
#SBATCH --error=$CHANDRA_DIR/logs/chandra_md_%j.err

set -euo pipefail

module purge
module load miniforge3
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

export PYTHONNOUSERSITE=1
export HF_HOME="$CHANDRA_DIR/cache/huggingface"
export TRANSFORMERS_CACHE="\$HF_HOME"
export TORCH_HOME="$CHANDRA_DIR/cache/torch"

mkdir -p "\$HF_HOME" "\$TORCH_HOME"

BASENAME="\$(basename "$IN_PNG")"
NAME="\${BASENAME%.*}"

echo "Running Chandra on GPU..."

chandra "$IN_PNG" "$RUNS_DIR" \
  --method hf \
  --batch-size 1 \
  --include-images \
  --max-output-tokens 4096

MD="$RUNS_DIR/\$NAME/\$NAME.md"

if [[ ! -f "\$MD" ]]; then
  echo "ERROR: MD not created: \$MD"
  exit 1
fi

cp "\$MD" "$OUT_MD"
echo "MD copied to: $OUT_MD"
EOF

chmod +x "$JOBSCRIPT"
sbatch "$JOBSCRIPT"