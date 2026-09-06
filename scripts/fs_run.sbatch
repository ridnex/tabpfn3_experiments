#!/bin/bash
#SBATCH --job-name=fs-mrpca
# a100 (sm_80) required: torch 2.11.0+cu128 has no sm_70 kernels.
#SBATCH --gres=gpu:a100:1
# CPU-heavy, GPU-light: TabPFN sees 100 features (~1-2s), while MultiRocket's
# transform and the PCA over ~50k features run on the CPU.
#SBATCH --cpus-per-task=16
# 96G: MultiRocket makes 49728 float64 columns, so ECG5000's 4500 test series
# alone is ~1.8GB before StandardScaler copies it.
#SBATCH --mem=96G
# Measured on CPU: ~50s per dataset, of which TabPFN is most of it and moves to
# the GPU here. 20 datasets x 30 resamples at ~40s is ~7h; 14h is real margin.
#SBATCH --time=14:00:00
#SBATCH --output=/ibex/user/zharkyy/TabPFN/logs/%x-%j.out
#SBATCH --error=/ibex/user/zharkyy/TabPFN/logs/%x-%j.out

# Usage: sbatch scripts/fs_run.sbatch [START] [END]   (inclusive resamples)
set -euo pipefail
source /ibex/user/zharkyy/TabPFN/scripts/env.sh

START="${1:-0}"
END="${2:-29}"

echo "=== node: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK}  resamples ${START}..${END}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Liveness check before committing the allocation. Job 51210098 burned a whole
# 48h allocation in 43 seconds because a refactor had dropped run.py's
# `if __name__ == "__main__"` block: it imported, did nothing, exited 0. Only
# running the file catches that.
SMOKE="$(mktemp -d)/smoke.csv"
"$PY" "$PROJ/src/tsc/fs_run.py" --resample 0 --device cuda \
    --only ItalyPowerDemand --out "$SMOKE" --tag smoke
if [ ! -s "$SMOKE" ] || [ "$(wc -l < "$SMOKE")" -lt 2 ]; then
    echo "FATAL: smoke run produced no rows - aborting"
    exit 1
fi
echo "=== smoke ok, starting sweep"

for R in $(seq "$START" "$END"); do
    echo ""
    echo "===== resample $R  ($(date -u +%H:%M:%S) UTC) ====="
    # fs_run.py resumes per resample and catches per-dataset failures, so a
    # walltime kill loses at most the dataset in flight.
    "$PY" "$PROJ/src/tsc/fs_run.py" \
        --resample "$R" \
        --transformer MultiRocket --selector pca --top-n 100 \
        --device cuda \
        --tag "a100-reference"
done

echo ""
echo "=== all resamples ${START}..${END} complete"
