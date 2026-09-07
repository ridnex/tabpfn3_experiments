#!/bin/bash
#SBATCH --job-name=stage1
# a100 (sm_80) required: torch 2.11.0+cu128 ships no sm_70 kernels.
#SBATCH --gres=gpu:a100:1
# CPU-bound, not GPU-bound: TabPFN sees 100 features (~1.5s per arm) while the
# MultiRocket transform and the MI scoring over ~50k features run on CPU.
#SBATCH --cpus-per-task=16
# MultiRocket emits 49728 float64 columns; Wafer's 7164 series is ~2.8GB before
# StandardScaler copies it for PCA.
#SBATCH --mem=96G
# 20 datasets x 10 resamples x 4 arms. Transform and MI are computed once per
# dataset-resample and shared across arms, so ~90s each -> ~5h. 10h is margin.
#SBATCH --time=10:00:00
#SBATCH --output=/ibex/user/zharkyy/TabPFN/logs/%x-%j.out
#SBATCH --error=/ibex/user/zharkyy/TabPFN/logs/%x-%j.out

set -euo pipefail
source /ibex/user/zharkyy/TabPFN/scripts/env.sh

START="${1:-0}"
END="${2:-9}"

echo "=== node: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK}  resamples ${START}..${END}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Liveness check before committing the allocation. Job 51210098 burned a 48h
# allocation in 43 seconds because a refactor had dropped run.py's
# `if __name__ == "__main__"` block: it imported, did nothing, exited 0. Only
# executing the file catches that, and this script has never run on a GPU.
SMOKE="$(mktemp -d)/smoke.csv"
"$PY" "$PROJ/src/tsc/stage1.py" --resample 0 --device cuda \
    --only Fish --out "$SMOKE" --tag smoke
if [ "$(wc -l < "$SMOKE")" -lt 5 ]; then
    echo "FATAL: smoke produced fewer than 4 arm rows - aborting"
    exit 1
fi
echo "=== smoke ok (4 arms wrote rows), starting sweep"

for R in $(seq "$START" "$END"); do
    echo ""
    echo "===== resample $R  ($(date -u +%H:%M:%S) UTC) ====="
    # stage1.py resumes at dataset granularity - a dataset counts as done only
    # when all four arms are on disk, so a kill never leaves a half-measured
    # factorial cell.
    "$PY" "$PROJ/src/tsc/stage1.py" \
        --resample "$R" \
        --transformer MultiRocket --top-n 100 \
        --device cuda \
        --tag "a100-reference"
done

echo ""
echo "=== stage 1 resamples ${START}..${END} complete"
