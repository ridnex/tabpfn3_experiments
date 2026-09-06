#!/bin/bash
#SBATCH --job-name=rocketpfn-92
# a100 (sm_80) is a requirement, not a preference: torch 2.11.0+cu128 ships no
# sm_70 kernels, so a V100 fails instantly with
#   CUDA error: no kernel image is available for execution on the device
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
# 128G, double the 20-dataset run. ElectricDevices holds 8926 training series in
# TabPFN's context - 5x the largest we have measured - and host-side Rocket
# features for it are ~140M floats before anything reaches the GPU.
#SBATCH --mem=128G
# ONE allocation for all 92 datasets x 30 resamples.
#
# On this cluster the scarce resource is ACCESS, not GPU time: the 30-task array
# trickled through 6 tasks while 144 a100 jobs sat pending. Paying the queue
# wait once is worth a long walltime.
#
# Sizing: 29 GPU-h predicted for the 72 new datasets, from a linear fit on the
# 20 already measured (R^2 = 0.99), plus ~7.5h if the finished ones are re-run.
# 48h is deliberately generous. The `gpu` partition allows up to 14 days.
#SBATCH --time=48:00:00
#SBATCH --output=/ibex/user/zharkyy/TabPFN/logs/%x-%j.out
#SBATCH --error=/ibex/user/zharkyy/TabPFN/logs/%x-%j.out

# Usage: sbatch scripts/run_rocketpfn_full.sbatch [START] [END]   (inclusive)
set -euo pipefail
source /ibex/user/zharkyy/TabPFN/scripts/env.sh

START="${1:-0}"
END="${2:-29}"

echo "=== node: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK}  resamples ${START}..${END}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Liveness check before committing 48h to the loop. Job 51210098 burned a
# whole allocation in 43 seconds because run.py had lost its
# `if __name__ == "__main__"` block in a refactor: it imported, did nothing,
# and exited 0, so every resample "completed" instantly. Syntax checks do not
# catch that - only running it does. Two minutes here is cheap insurance
# against another three-day queue wait.
SMOKE="$(mktemp -d)/smoke.csv"
"$PY" "$PROJ/src/tsc/run.py" --datasets "$PROJ/configs/ucr92.json" \
    --resample 0 --device cuda --only Chinatown --out "$SMOKE" --tag smoke
if [ ! -s "$SMOKE" ] || [ "$(wc -l < "$SMOKE")" -lt 2 ]; then
    echo "FATAL: smoke run produced no rows - aborting before wasting the allocation"
    exit 1
fi
echo "=== smoke ok, starting sweep"

for R in $(seq "$START" "$END"); do
    echo ""
    echo "===== resample $R  ($(date -u +%H:%M:%S) UTC) ====="
    # run.py resumes: it skips every dataset already in resample_NN.csv, so the
    # 20 x 30 already measured cost nothing here and a walltime kill loses only
    # the dataset in flight. It also catches per-dataset failures rather than
    # aborting - configs/ucr92.json is ordered so the three datasets with an
    # unproven context size (ElectricDevices, FordB, FordA) run last, and one of
    # them OOMing costs itself and nothing else.
    "$PY" "$PROJ/src/tsc/run.py" \
        --datasets "$PROJ/configs/ucr92.json" \
        --resample "$R" \
        --device cuda \
        --tag "a100-reference"
done

echo ""
echo "=== all resamples ${START}..${END} complete"
