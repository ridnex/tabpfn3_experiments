#!/bin/bash
#SBATCH --job-name=kernel-probe
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Deliberately tiny: 3 kernel settings x 5 datasets x 5 resamples = 75 runs of
# ~10s. A short, small request backfills into a schedule gap in seconds, where
# our big jobs waited hours on priority - and here the queue wait dominates the
# compute by an order of magnitude.
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/user/zharkyy/TabPFN/logs/%x-%j.out
#SBATCH --error=/ibex/user/zharkyy/TabPFN/logs/%x-%j.out

# Does the feature count need to scale with dataset size?
#
# RocketPFN fixes 1000 kernels per group (2000 features) for EVERY dataset -
# confirmed against the paper, which ablated G and feature width but never the
# kernel count. On these five datasets that is 2000 features for 17-40 training
# rows, a 50-100:1 ratio. This sweeps down to 20 features and asks whether less
# is more when data is scarce.
#
# The k=1000 control is already on disk from the 92x30 sweep at these same
# resamples, so nothing is re-measured and the comparison is paired.
set -euo pipefail
source /ibex/user/zharkyy/TabPFN/scripts/env.sh

echo "=== node: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK}"
nvidia-smi --query-gpu=name --format=csv,noheader

for K in 10 50 100; do
    for R in 0 1 2 3 4; do
        echo ""
        echo "===== kernels=$K resample=$R  ($(date -u +%H:%M:%S) UTC) ====="
        "$PY" "$PROJ/src/tsc/run.py" \
            --datasets "$PROJ/configs/kernel_probe.json" \
            --resample "$R" --kernels "$K" --device cuda \
            --out "$PROJ/results/kernel_probe/k${K}_resample_$(printf %02d $R).csv" \
            --tag "kernel-probe k=$K"
    done
done

echo ""
echo "=== kernel probe complete"
