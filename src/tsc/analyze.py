"""Compare RocketPFN against published UCR results.

Average rank is the headline, not mean accuracy. Mean accuracy over datasets of
very different difficulty is dominated by whichever ones happen to be hard;
rank is what the TSC literature reports and what a critical-difference diagram
is defined on.

Nothing is rerun here. HC2 and the other comparators come from the published
per-resample results the archive hosts, so our column is dropped into their
matrix rather than re-measured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load_ours(path: Path, datasets: list[str], n_resamples: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df.dataset.isin(datasets) & (df.resample < n_resamples)]
    # Guard against a half-finished sweep being averaged as if complete: a
    # dataset with 3 of 30 resamples would otherwise silently get a noisier
    # mean than its neighbours and still be ranked against them.
    counts = df.groupby("dataset").resample.nunique()
    incomplete = counts[counts < n_resamples]
    if len(incomplete):
        print(f"[warn] {len(incomplete)} datasets have < {n_resamples} resamples:")
        for name, c in incomplete.items():
            print(f"       {name:28s} {c}/{n_resamples}")
    return df


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(ROOT / "results" / "rocketpfn_ucr.csv"))
    p.add_argument("--config", default=str(ROOT / "configs" / "ucr20.json"))
    p.add_argument("--resamples", type=int, default=30)
    p.add_argument("--report", default=str(ROOT / "reports" / "rocketpfn.md"))
    p.add_argument("--cd-plot", default=str(ROOT / "reports" / "rocketpfn_cd.png"))
    args = p.parse_args()

    from aeon.benchmarking.results_loaders import get_estimator_results_as_array
    from aeon.benchmarking.stats import wilcoxon_test

    cfg = json.loads(Path(args.config).read_text())
    datasets = [d["name"] for d in cfg["datasets"]]
    comparators = cfg["classifiers"]

    ours = load_ours(Path(args.results), datasets, args.resamples)
    mine = ours.groupby("dataset").accuracy.agg(["mean", "std", "count"])

    published, names = get_estimator_results_as_array(
        estimators=comparators, datasets=datasets,
        num_resamples=args.resamples, include_missing=True,
    )
    published = pd.DataFrame(np.asarray(published, dtype=float),
                             index=names, columns=comparators)
    if published.isna().any().any():
        raise SystemExit("published results have missing cells; ranking would be invalid")

    # Restrict to datasets we actually finished, so ranks are computed on a
    # complete matrix rather than a ragged one.
    have = [d for d in datasets if d in mine.index]
    table = published.loc[have].copy()
    table["RocketPFN"] = mine.loc[have, "mean"]

    # rank(ascending=False) -> rank 1 is the most accurate. Ties share the mean
    # rank, which is the convention the CD diagram assumes.
    ranks = table.rank(axis=1, ascending=False)
    avg_rank = ranks.mean().sort_values()

    lines: list[str] = []
    add = lines.append
    add("# RocketPFN on UCR — reproduction\n")
    add(f"- Datasets: **{len(have)}** of {cfg['n']} "
        f"(stratified from a pool of {cfg['pool_size']}, seed {cfg['seed']})")
    add(f"- Resamples: **{args.resamples}** (resample 0 is the archive default split)")
    add(f"- Comparators: {', '.join(comparators)} — published results, not rerun")
    add("- **Deviation:** TabPFN v3 here; the paper used v2.5\n")

    add("## Average rank (lower is better)\n")
    add("| method | avg rank | mean accuracy |")
    add("|---|---|---|")
    for m, r in avg_rank.items():
        add(f"| {'**' + m + '**' if m == 'RocketPFN' else m} | {r:.2f} | "
            f"{table[m].mean():.4f} |")

    add("\n## RocketPFN vs HC2\n")
    if "HC2" in table:
        diff = table["RocketPFN"] - table["HC2"]
        w = wilcoxon_test(table[["RocketPFN", "HC2"]].to_numpy(),
                          ["RocketPFN", "HC2"], lower_better=False)
        add(f"- mean accuracy: RocketPFN {table['RocketPFN'].mean():.4f} "
            f"vs HC2 {table['HC2'].mean():.4f}")
        add(f"- RocketPFN wins on {int((diff > 0).sum())}, loses {int((diff < 0).sum())}, "
            f"ties {int((diff == 0).sum())}")
        add(f"- Wilcoxon signed-rank p = {float(np.asarray(w)[0, 1]):.4f}")

    add("\n## Per-dataset accuracy\n")
    cols = ["RocketPFN"] + comparators
    add("| dataset | type | " + " | ".join(cols) + " | n_resamples |")
    add("|---" * (len(cols) + 3) + "|")
    types = {d["name"]: d["type"] for d in cfg["datasets"]}
    for d in have:
        cells = " | ".join(
            (f"**{table.loc[d, c]:.4f}**"
             if table.loc[d, c] == table.loc[d, cols].max() else f"{table.loc[d, c]:.4f}")
            for c in cols
        )
        add(f"| {d} | {types[d]} | {cells} | {int(mine.loc[d, 'count'])} |")

    add("\n## Cost\n")
    add(f"- median total per dataset per resample: {ours.total_time.median():.1f}s")
    add(f"- median Rocket feature time: {ours.feature_time.median():.1f}s")
    add(f"- median TabPFN time: {ours.tabpfn_time.median():.1f}s")
    add(f"- total GPU time measured: {ours.total_time.sum() / 3600:.2f}h")

    try:
        from aeon.visualisation import plot_critical_difference

        fig, _ = plot_critical_difference(table.to_numpy(), list(table.columns))
        Path(args.cd_plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.cd_plot, dpi=150, bbox_inches="tight")
        add(f"\n![critical difference]({Path(args.cd_plot).name})")
        print(f"[plot] -> {args.cd_plot}")
    except Exception as exc:
        # A missing plot must not destroy the numeric report, which is the part
        # that actually carries the result.
        add(f"\n_(CD diagram not generated: {type(exc).__name__}: {exc})_")
        print(f"[warn] CD plot failed: {exc}")

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    print(f"\n[done] -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
