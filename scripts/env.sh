# Shared environment setup. Source this from any run script.
export PROJ=/ibex/user/zharkyy/TabPFN
export PY=/ibex/user/zharkyy/conda-environments/tabpfn-bench/bin/python

# TabPFN-3 weights are license-gated; the token lives in a mode-600 file that is
# gitignored, so it never lands in a script, a log, or job output.
if [ -f "$PROJ/.tabpfn_token" ]; then
    . "$PROJ/.tabpfn_token"
fi

# Keep the TabPFN checkpoint and HF cache off the home quota.
export TABPFN_MODEL_CACHE_DIR=/ibex/user/zharkyy/tabpfn_models
export HF_HOME=/ibex/user/zharkyy/hf_home
export XDG_CACHE_HOME=/ibex/user/zharkyy/xdg_cache

# Pin BLAS/OpenMP threads to the cores Slurm actually gave us, so CPU timings
# are reproducible instead of depending on whatever the node happened to expose.
NCPU="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$NCPU"
export MKL_NUM_THREADS="$NCPU"
export OPENBLAS_NUM_THREADS="$NCPU"
export NUMEXPR_NUM_THREADS="$NCPU"
