"""Pre-flight check: CUDA works, TabPFN loads, and the real API matches our assumptions."""
import inspect
import sys

import numpy as np
import torch

print("torch", torch.__version__, "| cuda avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    x = torch.randn(1000, 1000, device="cuda") @ torch.randn(1000, 1000, device="cuda")
    torch.cuda.synchronize()
    print("matmul ok, sum =", float(x.sum()))
else:
    print("!! CUDA NOT AVAILABLE")
    sys.exit(1)

from tabpfn import TabPFNRegressor

sig = inspect.signature(TabPFNRegressor.__init__)
print("\nTabPFNRegressor params:")
for name, prm in sig.parameters.items():
    if name != "self":
        print(f"   {name} = {prm.default}")

rng = np.random.RandomState(0)
X, y = rng.rand(200, 5), rng.rand(200)
m = TabPFNRegressor(device="cuda", random_state=0)
m.fit(X, y)
p = m.predict(X[:20])
print("\ntabpfn ok, pred shape", np.asarray(p).shape)

import xgboost, lightgbm
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

XGBRegressor(n_estimators=5, device="cpu").fit(X, y)
XGBRegressor(n_estimators=5, device="cuda").fit(X, y)
LGBMRegressor(n_estimators=5, verbose=-1).fit(X, y)
print("xgboost", xgboost.__version__, "cpu+gpu ok | lightgbm", lightgbm.__version__, "ok")
print("\nALL OK")
