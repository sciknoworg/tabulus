#!/bin/bash
#SBATCH --job-name=tabulus-grid-eval
#SBATCH --array=0-15
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=tabulus-grid-%A_%a.log
set -euo pipefail

: "${BENCH_ROOT:?Set BENCH_ROOT to the TabulusBench checkout}"
: "${EVAL_OUT:?Set EVAL_OUT to a new evaluation directory}"
cd "${SLURM_SUBMIT_DIR}"
adapters=(
  chandra granite-vision-table dots-mocr nuextract3 trivia glm-ocr
  paddleocr-vl dolphin-v2 monkeyocrv2-b-parsing hunyuanocr-1-5
  deepseek-ocr-2 nanonets-ocr-s nemotron-parse-v1-2 internvl3-5-8b
  rapidocr-tableformer tesseract-tatr
)
task_out="${EVAL_OUT}/task-${SLURM_ARRAY_TASK_ID}"
resume=()
if [[ -f "${task_out}/run.json" ]]; then
  resume=(--resume)
fi
python -u scripts/evaluation/step2/evaluate_exact_match.py \
  --root "${BENCH_ROOT}" --out "${task_out}" \
  --adapters "${adapters[SLURM_ARRAY_TASK_ID]}" --no-native \
  --metrics rms exact-mapping exact --workers "${SLURM_CPUS_PER_TASK}" \
  --run-id "${EVAL_RUN_ID:-run_01}" --subset "${EVAL_SUBSET:-all}" \
  "${resume[@]}"
