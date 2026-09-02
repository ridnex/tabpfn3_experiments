"""Pick the 20-dataset UCR subset for the RocketPFN reproduction.

The selection rule is fixed here and the output committed BEFORE any accuracy
is measured. That ordering is the whole point: a subset chosen after seeing
results is indistinguishable from a cherry-pick, so the rule has to be
auditable and the artefact has to predate the numbers.

The candidate pool is DERIVED, not transcribed from the paper:

    aeon's univariate_equal_length (112)  x  UCR "Class" <= 10   ->  92

which reproduces the paper's 92 exactly. Deriving it means a typo in a
hand-copied exclusion list cannot silently shift the pool, and it records the
reason each dataset is out (TabPFN's class cap) rather than just the fact.

Allocation across UCR problem types uses largest-remainder (Hare quota) rather
than a plain round, so the parts sum to exactly 20 without a fudge, and ties
break on type name so the result does not depend on dict ordering.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "data" / "published" / "UCR_DataSummary.csv"
OUT = ROOT / "configs" / "ucr20.json"

# TabPFN's class cap is what removed 20 datasets from the 112 for the paper.
# Kept as a named constant because step 0 measures v3's actual cap: if v3 lifts
# it, this is the single number to change to widen the pool beyond the paper.
MAX_CLASSES = 10
# Largest training set we have measured end to end (PhalangesOutlinesCorrect).
# Used only to order the full-pool run so unproven context sizes go last.
PROVEN_CONTEXT_ROWS = 1800
N_SELECT = 20
SEED = 42

# Frozen here, not at the analysis step. Average rank is defined relative to the
# comparison set, so adding a classifier later silently changes every rank we
# report. Changing this list means rerunning the analysis and saying so.
CLASSIFIERS = ["HC2", "InceptionTime", "ROCKET", "DrCIF", "TS-CHIEF"]


def load_pool() -> pd.DataFrame:
    """The 92 candidates, with problem type and class count attached."""
    from aeon.datasets.tsc_datasets import univariate_equal_length

    # encoding="utf-8-sig": the UCR file ships with a BOM, which otherwise ends
    # up glued to the first column name as "﻿ID".
    d = pd.read_csv(SUMMARY, encoding="utf-8-sig")
    d.columns = [c.strip() for c in d.columns]
    d["Name"] = d["Name"].str.strip()

    equal_length = set(univariate_equal_length)
    missing = equal_length - set(d["Name"])
    if missing:
        raise SystemExit(
            f"UCR summary is missing {len(missing)} aeon datasets: {sorted(missing)}\n"
            "The summary and aeon's list must agree, or the pool is not the paper's."
        )
    pool = d[d["Name"].isin(equal_length) & (d["Class"] <= MAX_CLASSES)].copy()
    # Length arrives as text because variable-length datasets carry "Vary" in
    # that column - hence the coercion only after filtering to the equal-length
    # set, where every value really is a number. It matters: (Train+Test)*Length
    # on strings does list repetition, not multiplication, and silently produced
    # a nonsense cost ordering before this line existed.
    for c in ("Train", "Test", "Length"):
        pool[c] = pd.to_numeric(pool[c].astype(str).str.strip(), errors="raise")
    return pool.sort_values("Name").reset_index(drop=True)


def hare_quota(counts: pd.Series, total: int) -> dict[str, int]:
    """Allocate `total` slots across strata proportionally, summing exactly.

    Plain rounding of proportional shares does not sum to the target - here it
    would give 21. Largest-remainder assigns the floors first, then hands the
    leftovers to the largest fractional parts. Ties break on stratum name so the
    allocation is a pure function of the counts.
    """
    quota = counts / counts.sum() * total
    alloc = np.floor(quota).astype(int)
    remainder = total - int(alloc.sum())
    order = sorted(counts.index, key=lambda k: (-(quota[k] - np.floor(quota[k])), k))
    for name in order[:remainder]:
        alloc[name] += 1
    return {k: int(v) for k, v in alloc.items()}


def select(pool: pd.DataFrame, n: int = N_SELECT, seed: int = SEED) -> list[str]:
    counts = pool["Type"].value_counts().sort_index()
    alloc = hare_quota(counts, n)
    rng = np.random.default_rng(seed)
    chosen: list[str] = []
    # Iterate types in sorted order, not value_counts order, so the stream of
    # draws from `rng` is reproducible independently of the counts.
    for t in sorted(alloc):
        k = alloc[t]
        if k == 0:
            continue
        names = sorted(pool.loc[pool["Type"] == t, "Name"])
        chosen += list(rng.choice(names, size=k, replace=False))
    return sorted(chosen)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=N_SELECT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", default=str(OUT))
    # --all emits the whole 92 pool instead of a sample. The 20-dataset config
    # stays exactly as committed: this writes a separate file, and the 20 are a
    # subset of the 92, so results already measured carry over unchanged.
    p.add_argument("--all", action="store_true",
                   help="write the full pool rather than a stratified sample")
    args = p.parse_args()

    pool = load_pool()
    if args.all:
        args.n = len(pool)
    print(f"[pool] {len(pool)} datasets (univariate_equal_length x Class<={MAX_CLASSES})")
    counts = pool["Type"].value_counts().sort_index()
    alloc = hare_quota(counts, args.n)
    for t in sorted(counts.index):
        print(f"  {t:11s} {counts[t]:3d} -> {alloc[t]}")

    if args.all:
        # Cheapest first. Every dataset costs one TabPFN forward pass per test
        # series over a context of every training series, so (train + test) *
        # length tracks runtime closely (R^2 = 0.99 against our measured 20).
        # Ordering by it puts the four datasets big enough to risk an OOM -
        # Crop, ElectricDevices, StarLightCurves, Wafer - at the very end, so a
        # failure there costs only itself and the ~88 results before it are
        # already on disk.
        cost = (pool["Train"] + pool["Test"]) * pool["Length"]
        # Two different things are being deferred, and they are not the same
        # column. Runtime scales with (train+test)*length. MEMORY scales with
        # the training set alone, because TabPFN holds every training series in
        # context: 1800 rows is the largest we have actually run, so anything
        # above that is unproven and sorts after everything proven, however
        # cheap it looks. ElectricDevices (8926 train) is mid-pack on cost and
        # is exactly the dataset that must not run early.
        risky = (pool["Train"] > PROVEN_CONTEXT_ROWS).astype(int)
        chosen = list(
            pool.assign(_r=risky, _c=cost).sort_values(["_r", "_c", "Name"])["Name"]
        )
    else:
        chosen = select(pool, args.n, args.seed)
    if len(chosen) != args.n:
        raise SystemExit(f"selected {len(chosen)}, expected {args.n}")

    meta = pool.set_index("Name")
    payload = {
        "seed": args.seed,
        "n": args.n,
        "pool_size": int(len(pool)),
        "max_classes": MAX_CLASSES,
        "classifiers": CLASSIFIERS,
        "rule": (
            "aeon univariate_equal_length (112) filtered to UCR Class<=%d (92), "
            "all of them, ordered by (train+test)*length ascending"
            % MAX_CLASSES
        ) if args.all else (
            "aeon univariate_equal_length (112) filtered to UCR Class<=%d (92), "
            "stratified by UCR problem type with largest-remainder allocation, "
            "sampled with numpy default_rng(seed)" % MAX_CLASSES
        ),
        "datasets": [
            {
                "name": n,
                "type": meta.loc[n, "Type"],
                "n_classes": int(meta.loc[n, "Class"]),
                "length": int(meta.loc[n, "Length"]),
                "n_train": int(meta.loc[n, "Train"]),
                "n_test": int(meta.loc[n, "Test"]),
            }
            for n in chosen
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n[done] {len(chosen)} datasets -> {out}")
    for d in payload["datasets"]:
        print(f"  {d['name']:26s} {d['type']:10s} C={d['n_classes']:2d} "
              f"L={d['length']:5d} train={d['n_train']:5d} test={d['n_test']:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
