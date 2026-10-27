#!/usr/bin/env bash
set -euo pipefail

# Standalone TTVS launcher for the Qwen2.5-Instruct math experiment.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "${TTVS_LOAD_MODULES:-1}" == "1" ]] && command -v module >/dev/null 2>&1; then
  module purge
  module load "${TTVS_ANACONDA_MODULE:-Anaconda3/2023.09-0}"
fi

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${TTVS_CONDA_ENV:-ttrl}"
fi

export RAY_TMPDIR="${RAY_TMPDIR:-$REPO_ROOT/.ray_tmp}"
mkdir -p "$RAY_TMPDIR"
ray stop --force >/dev/null 2>&1 || true
ray start --head --temp-dir="$RAY_TMPDIR"
export RAY_ADDRESS="auto"

bash examples/ttrl/Qwen2.5-Instruct/math.sh "$@"
