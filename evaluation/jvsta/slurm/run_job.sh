#!/bin/bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: run_job.sh <repo-root> <conda-env> <command> [args ...]" >&2
  exit 2
fi

repo_root="$1"
env_name="$2"
shift 2

if [[ ! -d "$repo_root" ]]; then
  echo "Tabulus repository root does not exist: $repo_root" >&2
  exit 2
fi
if [[ ! -f "$repo_root/evaluation/__init__.py" ]]; then
  echo "Tabulus repository root does not contain evaluation package: $repo_root" >&2
  exit 2
fi
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
