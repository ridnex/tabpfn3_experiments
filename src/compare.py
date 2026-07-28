"""Paired per-fold comparison against a reference model.

Comparing mean +/- std across folds is misleading here: most of that spread is
*fold difficulty*, which every model sees identically. Because all models run on
the exact same folds, the right analysis is paired - look at the per-fold
difference, whose variance excludes the shared fold effect.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(__file__).resolve().parent.parent / "results" / "results.csv"

# Metrics where a SMALLER value is better. The diff is sign-flipped for these so
# that "positive = reference is better" holds for every metric - otherwise the
# wins column and the legend silently invert and an rmse table reads backwards.
LOWER_IS_BETTER = {"rmse", "mae", "log_loss", "fit_time", "predict_time", "total_time"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(RESULTS))
    p.add_argument("--dataset", default="concrete")
    p.add_argument("--ref", default="tabpfn_v3")
    p.add_argument("--metric", default="r2")
    args = p.parse_args()

    df = pd.read_csv(args.results)
    df = df[df.dataset == args.dataset]
    piv = df.pivot_table(index="fold", columns="model", values=args.metric)
    if args.ref not in piv.columns:
        raise SystemExit(f"{args.ref} not in {list(piv.columns)}")

    ref = piv[args.ref]
    sign = -1.0 if args.metric in LOWER_IS_BETTER else 1.0
    direction = "lower is better" if sign < 0 else "higher is better"
    print(f"\n=== paired per-fold {args.metric} ({direction}): {args.ref} vs others  ({args.dataset}, {len(piv)} folds)")
    print(f"{'model':17s} {'mean diff':>11s} {'sd of diff':>11s} {'wins':>6s} {'p (wilcoxon)':>13s}")
    for m in piv.columns:
        if m == args.ref:
            continue
        d = (sign * (ref - piv[m])).dropna()
        wins = int((d > 0).sum())
        try:
            p_val = stats.wilcoxon(d).pvalue
        except Exception:
            p_val = float("nan")
        print(f"{m:17s} {d.mean():+11.4f} {d.std():11.4f} {wins:3d}/{len(d):<2d} {p_val:13.4f}")

    print("\n  mean diff > 0  =>  reference is better on that metric")
    print("  sd of diff is the number that matters, NOT the sd of raw scores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
