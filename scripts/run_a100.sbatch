#!/bin/bash
#SBATCH --job-name=tabpfn-bench
# NOTE: do NOT set --partition. Ibex forbids naming gpu4/gpu24/... directly and
# routes GPU jobs automatically from --gres + --time (4h here lands on gpu4).
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Keep this TIGHT. Slurm backfills short jobs into schedule gaps, so an honest
# estimate gets an A100 in seconds while an inflated one waits on priority.
# Measured: the 1k rung took 4m13s end to end. Raise this for the big rungs.
#SBATCH --time=00:30:00
#SBATCH --output=/ibex/user/zharkyy/TabPFN/logs/%x-%j.out
#SBATCH --error=/ibex/user/zharkyy/TabPFN/logs/%x-%j.out

# Measurement run. Only numbers produced here are quotable - the A100 is the
# reference GPU and --cpus-per-task=8 is the reference CPU budget for GBDT.
set -euo pipefail
source /ibex/user/zharkyy/TabPFN/scripts/env.sh

DATASET="${1:-concrete}"

echo "=== node: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"$PY" "$PROJ/src/benchmark.py" \
    --dataset "$DATASET" \
    --models all \
    --gbdt-device cpu \
    --tabpfn-device cuda \
    --tag "a100-reference"
