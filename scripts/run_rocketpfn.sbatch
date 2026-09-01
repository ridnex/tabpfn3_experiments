#!/bin/bash
#SBATCH --job-name=rocketpfn
# No --partition: Ibex forbids naming gpu4/gpu24 directly and routes GPU jobs
# from --gres + --time.
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/ibex/user/zharkyy/TabPFN/logs/%x-%A_%a.out
#SBATCH --error=/ibex/user/zharkyy/TabPFN/logs/%x-%A_%a.out

# One resample per array task. Submit the validation run as
#   sbatch --array=0 scripts/run_rocketpfn.sbatch
# and the full sweep as
#   sbatch --array=0-29 scripts/run_rocketpfn.sbatch
# Results are append-only and keyed by (dataset, resample), so a failed element
# is re-runnable on its own.
set -euo pipefail
source /ibex/user/zharkyy/TabPFN/scripts/env.sh

RESAMPLE="${SLURM_ARRAY_TASK_ID:-0}"
EXTRA=("$@")

echo "=== node: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK}  resample: ${RESAMPLE}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"$PY" "$PROJ/src/tsc/run.py" \
    --resample "$RESAMPLE" \
    --device cuda \
    --tag "a100-reference" \
    "${EXTRA[@]}"
