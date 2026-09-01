#!/bin/bash
#SBATCH --job-name=rocketpfn-seq
# a100 (sm_80) is not a preference, it is a requirement: torch 2.11.0+cu128
# ships no sm_70 kernels, so a V100 fails immediately with
#   CUDA error: no kernel image is available for execution on the device
# Measured 2026-09-01 by racing an identical v100 array against an a100 one.
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# One GPU, all remaining resamples back to back.
#
# The array form (run_rocketpfn.sbatch) backfills better per task, but on this
# cluster the scarce resource is ACCESS, not GPU time: 144 a100 jobs were
# pending and the array trickled through 6 of 30 tasks. One long allocation
# pays that wait once instead of once per task.
#
# Sizing: measured 15.7 min per resample over resamples 0-5 (range 14.8-16.6),
# so 24 resamples is ~6.3h. 9h is ~43% margin - deliberately generous, because
# a timeout here costs the queue wait this script exists to avoid.
#SBATCH --time=09:00:00
#SBATCH --output=/ibex/user/zharkyy/TabPFN/logs/%x-%j.out
#SBATCH --error=/ibex/user/zharkyy/TabPFN/logs/%x-%j.out

# Usage: sbatch scripts/run_rocketpfn_seq.sbatch [START] [END]   (inclusive)
set -euo pipefail
source /ibex/user/zharkyy/TabPFN/scripts/env.sh

START="${1:-6}"
END="${2:-29}"

echo "=== node: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK}  resamples ${START}..${END}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

for R in $(seq "$START" "$END"); do
    echo ""
    echo "===== resample $R  ($(date -u +%H:%M:%S) UTC) ====="
    # Each resample writes its own CSV and run.py skips datasets already in it,
    # so a walltime kill loses at most the resample in flight - resubmitting
    # this same script continues rather than restarting.
    "$PY" "$PROJ/src/tsc/run.py" \
        --resample "$R" \
        --device cuda \
        --tag "a100-reference"
done

echo ""
echo "=== all resamples ${START}..${END} complete"
