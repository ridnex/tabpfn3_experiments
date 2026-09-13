#!/bin/bash
#SBATCH --job-name=feature-order
# Do NOT set --partition; Ibex routes GPU jobs from --gres + --time.
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/user/zharkyy/TabPFN/logs/%x-%j.out
#SBATCH --error=/ibex/user/zharkyy/TabPFN/logs/%x-%j.out

# Feature-order probe: 4 datasets x 2 settings x 6 column orders, TabPFN seed 42.
# Data must already be cached in data/ (run load() on the login node first).
set -euo pipefail
source /ibex/user/zharkyy/TabPFN/scripts/env.sh

echo "=== node: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"$PY" "$PROJ/src/feature_order.py" "$@"
