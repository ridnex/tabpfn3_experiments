#!/bin/bash
#SBATCH --job-name=fs-probe
# a100 (sm_80) required: torch 2.11.0+cu128 has no sm_70 kernels, so V100 fails
# on the first forward pass.
#SBATCH --gres=gpu:a100:1
# 16 CPUs, not the usual 8. This job is CPU-bound, not GPU-bound: TabPFN sees
# only 100 features (one cheap forward pass), while mutual_info_classif runs
# per-feature over pools of 10k-50k on the CPU. n_jobs is set from this.
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# 6 runs on one dataset. Measured on CPU: MiniRocket transform 26s, PCA 2.5s.
# MI is the unmeasured term and the reason this job exists - 1h is generous
# even if it is 10x my estimate, and a short request backfills quickly.
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/user/zharkyy/TabPFN/logs/%x-%j.out
#SBATCH --error=/ibex/user/zharkyy/TabPFN/logs/%x-%j.out

set -euo pipefail
source /ibex/user/zharkyy/TabPFN/scripts/env.sh

DATASET="${1:-Fish}"

echo "=== node: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"$PY" "$PROJ/src/tsc/fs_probe.py" \
    --dataset "$DATASET" \
    --top-n 100 \
    --device cuda \
    --tag "a100-reference"
