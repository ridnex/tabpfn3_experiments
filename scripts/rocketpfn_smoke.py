"""GPU smoke test for RocketPFN, run before spending array-job time.

Three things a wrong implementation would fail:
  1. accuracy on GunPoint - the paper's territory is ~0.99, a broken feature
     pipeline lands near the 0.5 default rate
  2. determinism - same random_state twice must be bit-identical, or nothing
     downstream is reproducible
  3. G=1 vs G=10 - more groups must not be worse, which is the whole argument
     for averaging over groups
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aeon.datasets import load_classification  # noqa: E402

from tsc.rocketpfn import RocketPFNClassifier  # noqa: E402


def run(name, groups, seed=0):
    Xtr, ytr = load_classification(name, split="train")
    Xte, yte = load_classification(name, split="test")
    m = RocketPFNClassifier(n_groups=groups, device="cuda", random_state=seed, n_jobs=8)
    t = time.perf_counter()
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)
    acc = (m.classes_[p.argmax(1)] == yte).mean()
    return acc, p, time.perf_counter() - t, m


acc1, _, t1, m1 = run("GunPoint", 1)
print(f"GunPoint G=1   acc={acc1:.4f}  {t1:.1f}s  est={m1.n_estimators_}", flush=True)

acc10, p10, t10, m10 = run("GunPoint", 10)
print(f"GunPoint G=10  acc={acc10:.4f}  {t10:.1f}s  est={m10.n_estimators_}", flush=True)

_, p10b, _, _ = run("GunPoint", 10)
same = np.array_equal(p10, p10b)
print(f"determinism (bit-identical probabilities): {same}", flush=True)

print()
ok = True
if acc10 < 0.90:
    print(f"FAIL: G=10 accuracy {acc10:.4f} < 0.90 - paper territory is ~0.99")
    ok = False
if acc10 < acc1 - 1e-9:
    print(f"FAIL: G=10 ({acc10:.4f}) worse than G=1 ({acc1:.4f})")
    ok = False
if not same:
    print("FAIL: not reproducible under a fixed random_state")
    ok = False
print("SMOKE OK" if ok else "SMOKE FAILED")
sys.exit(0 if ok else 1)
