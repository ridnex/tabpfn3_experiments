"""Break a single test block into sub-periods, from saved predictions.

A holdout run produces one number per model, which cannot be told apart from
noise. Slicing the same predictions by calendar period costs no extra compute and
turns that one number into several paired observations - and shows whether a model
degrades as the test period advances, which an aggregate hides completely.

Sub-periods are NOT independent samples: they are autocorrelated and usually sit
on a trend. So this prints spread and win counts, deliberately not a p-value.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import datasets as ds_mod  # noqa: E402
from benchmark import score_regression  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _timestamps(name: str) -> pd.Series:
    """Rebuild the time index that `row` in the predictions file points into.

    Deliberately routed through datasets.load()'s own cache and ordering rules
    rather than re-reading the source: if the two ever disagreed, every period
    boundary here would be silently wrong.
    """
    ds = ds_mod.REGISTRY[name]
    if not ds.time_col:
        raise SystemExit(f"{name} has no time_col - nothing to group by")
    df = ds_mod._read_local(ds, ds_mod.DATA_DIR) if ds.local_file else pd.read_parquet(
        ds_mod.DATA_DIR / f"{ds.name}_{ds.openml_id}.pq"
    )
    if ds.dedup:
        df = df.drop_duplicates()
    return df.sort_values(ds.time_col, kind="mergesort").reset_index(drop=True)[ds.time_col]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--preds", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--freq", default="M", help="pandas period alias: M, W, Q, D")
    p.add_argument("--metric", default="rmse", choices=["rmse", "mae", "r2"])
    p.add_argument("--ref", default="tabpfn_v3")
    p.add_argument("--baseline", default="mean_baseline",
                   help="model to express skill against; '' to disable")
    p.add_argument("--min-rows", type=int, default=500,
                   help="drop periods with fewer rows than this (and say so)")
    args = p.parse_args()

    pr = pd.read_parquet(args.preds)
    pr["period"] = _timestamps(args.dataset).iloc[pr["row"].to_numpy()].dt.to_period(
        args.freq
    ).to_numpy()

    sizes = pr.groupby("period")["row"].nunique()
    keep = sizes[sizes >= args.min_rows].index
    dropped = sizes[sizes < args.min_rows]
    if len(dropped):
        # Stated, never silent: a quietly discarded period reads as a period that
        # never existed, and the reader cannot audit what the table covers.
        print(f"[dropped] below --min-rows {args.min_rows}: "
              + ", ".join(f"{p} ({n} rows)" for p, n in dropped.items()))
    if not len(keep):
        raise SystemExit("no period met --min-rows")

    rec = []
    for (model, period), g in pr[pr.period.isin(keep)].groupby(["model", "period"]):
        rec.append({"model": model, "period": str(period), "n": len(g),
                    **score_regression(g.y_true, g.y_pred)})
    tab = pd.DataFrame(rec).pivot(index="period", columns="model", values=args.metric)

    models = [c for c in tab.columns if c != args.baseline]
    order = tab[models].mean().sort_values(ascending=args.metric != "r2").index.tolist()

    print(f"\n=== {args.dataset}: {args.metric} by {args.freq}  "
          f"({len(keep)} periods, {sizes[keep].sum():,} rows) ===")
    head = "".join(f"{m[:13]:>14s}" for m in order)
    print(f"{'period':9s} {'n':>6s}" + head)
    for per in tab.index:
        n = int(sizes[[i for i in sizes.index if str(i) == per][0]])
        print(f"{per:9s} {n:>6d}" + "".join(f"{tab.loc[per, m]:14.3f}" for m in order))
    print(f"{'MEAN':9s} {'':>6s}" + "".join(f"{tab[m].mean():14.3f}" for m in order))
    print(f"{'SD':9s} {'':>6s}" + "".join(f"{tab[m].std():14.3f}" for m in order))

    if args.baseline and args.baseline in tab.columns:
        print(f"\n--- {args.metric} as % of {args.baseline} (100% = no skill) ---")
        print(f"{'period':9s}" + "".join(f"{m[:13]:>14s}" for m in order))
        sk = tab[order].div(tab[args.baseline], axis=0) * 100
        for per in sk.index:
            print(f"{per:9s}" + "".join(f"{sk.loc[per, m]:13.1f}%" for m in order))
        print(f"{'MEAN':9s}" + "".join(f"{sk[m].mean():13.1f}%" for m in order))

    if args.ref in tab.columns:
        better = -1 if args.metric == "r2" else 1  # sign making "ref wins" positive
        print(f"\n--- periods won vs {args.ref} (of {len(tab)}) ---")
        for m in order:
            if m == args.ref:
                continue
            d = better * (tab[m] - tab[args.ref])
            print(f"{m:16s} {args.ref} wins {int((d > 0).sum())}/{len(d)}   "
                  f"mean diff {d.mean():+10.3f}   sd {d.std():9.3f}")
        print("\n  Sub-periods are autocorrelated and trending, so these are NOT")
        print("  independent samples - no p-value is reported for that reason.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
