#!/bin/bash
#SBATCH --job-name=tabulus-grid-aggregate
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=tabulus-grid-aggregate-%j.log
set -euo pipefail
: "${EVAL_OUT:?Set EVAL_OUT to the evaluation directory used by the array}"
cd "${SLURM_SUBMIT_DIR}"
python -m tabulus.cli aggregate-table-evaluations \
  --runs "${EVAL_OUT}"/task-* --out "${EVAL_OUT}/aggregate"
