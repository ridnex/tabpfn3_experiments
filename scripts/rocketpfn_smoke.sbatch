#!/bin/bash
#SBATCH --job-name=rpfn-smoke
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=/ibex/user/zharkyy/TabPFN/logs/%x-%j.out
#SBATCH --error=/ibex/user/zharkyy/TabPFN/logs/%x-%j.out
set -euo pipefail
source /ibex/user/zharkyy/TabPFN/scripts/env.sh
echo "=== node: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
"$PY" "$PROJ/scripts/rocketpfn_smoke.py"
