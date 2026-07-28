"""Summarise results.csv into a per-(dataset, model) table.

R2 is the headline metric; RMSE is kept for per-dataset interpretation. Every
cell is mean +/- std across folds, because a gap smaller than the spread is not
a result.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results" / "results.csv"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(RESULTS))
    p.add_argument("--dataset", default=None)
    args = p.parse_args()

    df = pd.read_csv(args.results)
    if args.dataset:
        df = df[df.dataset == args.dataset]
    if df.empty:
        print("no rows")
        return 1

    # Task is detected from the columns present rather than passed in, so the
    # same command works for both results files.
    clf = "roc_auc" in df.columns
    # (headline, spread-partner, plain) - headline drives the sort order.
    if clf:
        headline, second, third = "roc_auc", "log_loss", "balanced_accuracy"
        extra, fmt = "accuracy", 4
    else:
        headline, second, third = "r2", "rmse", "mae"
        # RMSE is in dataset units and can be ~44,000 dollars - 3dp keeps the
        # regression tables reading exactly as they did before this refactor.
        extra, fmt = None, 3

    # Group by gpu_name too: `device` is "cuda" for BOTH an A100 and a V100, so
    # grouping on device alone would silently average two different GPUs into one
    # timing row.
    # NB: bracket access below, not attribute access - a column named "head"
    # would resolve to pandas' Series.head method and format as garbage.
    agg = dict(
        folds=("fold", "count"),
        head=(headline, "mean"), head_sd=(headline, "std"),
        sec=(second, "mean"), sec_sd=(second, "std"),
        third=(third, "mean"),
        fit_s=("fit_time", "mean"),
        pred_s=("predict_time", "mean"),
        total_s=("total_time", "mean"), total_sd=("total_time", "std"),
    )
    if extra:
        agg["extra"] = (extra, "mean")
    g = df.groupby(["dataset", "n_rows", "model", "device", "gpu_name"], as_index=False).agg(**agg)
    g = g.sort_values(["dataset", "head"], ascending=[True, False])

    for (dset, n_rows, gpu), sub in g.groupby(["dataset", "n_rows", "gpu_name"]):
        cpus = df[df.dataset == dset].n_cpus.iloc[0]
        print(f"\n=== {dset}  (n={n_rows})   GPU={gpu}  CPU cores={cpus}")
        cols = (
            f"{'model':17s} {'dev':5s} {headline.upper():>18s} {second.upper():>18s} "
            f"{third[:9].upper():>9s} "
        )
        if extra:
            cols += f"{extra[:8].upper():>9s} "
        print(cols + f"{'fit s':>9s} {'pred s':>9s} {'total s':>18s}")
        for _, r in sub.sort_values("head", ascending=False).iterrows():
            line = (
                f"{r['model']:17s} {r['device']:5s} "
                f"{r['head']:8.4f}+/-{r['head_sd']:<7.4f} "
                f"{r['sec']:8.{fmt}f}+/-{r['sec_sd']:<7.{fmt}f} "
                f"{r['third']:9.{fmt}f} "
            )
            if extra:
                line += f"{r['extra']:9.4f} "
            print(line + f"{r['fit_s']:9.3f} {r['pred_s']:9.3f} "
                  f"{r['total_s']:8.3f}+/-{r['total_sd']:<7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
