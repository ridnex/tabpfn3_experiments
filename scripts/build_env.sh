#!/bin/bash
# Build the tabpfn-bench conda environment. Run on a login node (needs internet).
set -euo pipefail

export PIP_CACHE_DIR=/ibex/user/zharkyy/pip_cache
ENV_PREFIX=/ibex/user/zharkyy/conda-environments/tabpfn-bench
MAMBA=/ibex/user/zharkyy/miniforge/condabin/mamba

$MAMBA create -y -p "$ENV_PREFIX" python=3.11 pip

"$ENV_PREFIX/bin/pip" install --no-input \
    torch \
    tabpfn \
    xgboost \
    lightgbm \
    scikit-learn \
    pandas \
    pyarrow \
    requests

echo "=== versions ==="
"$ENV_PREFIX/bin/python" - <<'PY'
import torch, xgboost, lightgbm, sklearn, pandas, pyarrow, tabpfn
print("python  ", __import__("sys").version.split()[0])
print("torch   ", torch.__version__)
print("tabpfn  ", tabpfn.__version__ if hasattr(tabpfn, "__version__") else "?")
print("xgboost ", xgboost.__version__)
print("lightgbm", lightgbm.__version__)
print("sklearn ", sklearn.__version__)
print("pandas  ", pandas.__version__)
PY
