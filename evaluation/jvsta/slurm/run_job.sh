#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: run_job.sh <conda-env> <command> [args ...]" >&2
  exit 2
fi

env_name="$1"
shift

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

conda_bin="${CONDA_EXE:-}"
if [[ -z "$conda_bin" || ! -x "$conda_bin" ]]; then
  conda_bin="$(command -v conda || true)"
fi
if [[ -z "$conda_bin" || ! -x "$conda_bin" ]]; then
  for candidate in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do
    if [[ -x "$candidate" ]]; then
      conda_bin="$candidate"
      break
    fi
  done
fi
if [[ -z "$conda_bin" || ! -x "$conda_bin" ]]; then
  echo "Could not locate a conda executable." >&2
  exit 2
fi

exec "$conda_bin" run --no-capture-output -n "$env_name" "$@"
