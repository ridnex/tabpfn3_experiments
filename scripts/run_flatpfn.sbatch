#!/bin/bash
#SBATCH --job-name=flatpfn
# Plain TabPFN v3 on raw series (src/tsc/flatpfn.py), all 92 UCR datasets.
# a100 only: torch 2.11.0+cu128 has no sm_70 kernels, a V100 fails on first forward.
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
# One TabPFN call per dataset (vs RocketPFN's 10 groups of 2000 features), so
# resample 0 should be a fraction of RocketPFN's ~1.1 GPU-h. run.py resumes, so a
# walltime kill costs only the dataset in flight - resubmit.
#SBATCH --time=06:00:00
#SBATCH --output=/ibex/user/zharkyy/TabPFN/logs/%x-%j.out
#SBATCH --error=/ibex/user/zharkyy/TabPFN/logs/%x-%j.out

# Usage: sbatch scripts/run_flatpfn.sbatch [START] [END]   (inclusive, default 0 0)
set -euo pipefail
source /ibex/user/zharkyy/TabPFN/scripts/env.sh
# env.sh sets PROJ to the main checkout, which must be on the flatpfn branch.
DATA="$PROJ/data/ucr"

START="${1:-0}"
END="${2:-0}"

echo "=== node: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK}  resamples ${START}..${END}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Liveness check before the sweep: a run.py that imports and exits 0 once
# burned a whole allocation. Only running it catches that.
SMOKE="$(mktemp -d)/smoke.csv"
"$PY" "$PROJ/src/tsc/run.py" --method flat --datasets "$PROJ/configs/ucr92.json" \
    --data-dir "$DATA" --resample 0 --device cuda --only Chinatown --out "$SMOKE" --tag smoke
if [ ! -s "$SMOKE" ] || [ "$(wc -l < "$SMOKE")" -lt 2 ]; then
    echo "FATAL: smoke run produced no rows - aborting before wasting the allocation"
    exit 1
fi
echo "=== smoke ok, starting sweep"

for R in $(seq "$START" "$END"); do
    echo ""
    echo "===== resample $R  ($(date -u +%H:%M:%S) UTC) ====="
    "$PY" "$PROJ/src/tsc/run.py" \
        --method flat \
        --datasets "$PROJ/configs/ucr92.json" \
        --data-dir "$DATA" \
        --resample "$R" \
        --device cuda \
        --tag "a100-flat"
done

echo ""
echo "=== flatpfn resamples ${START}..${END} complete"
