"""Charts for the feature-order probe (results/feature_order.csv).

  * one PNG per dataset: baseline + 5 shuffles, default vs n_estimators=1, with
    the exact value on every bar. Regression gets R2 and RMSE side by side (two
    panels, never two y-scales on one axis); classification gets accuracy.
  * one overall PNG: per dataset, baseline and shuffle mean for each setting.

Y-axes are zoomed to the data range, because the differences are ~0.001 and
bars from zero would all look identical. Printed values keep the zoom honest.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "feature_order"

BLUE, BLUE_LIGHT = "#2a78d6", "#86b6ef"
ORANGE, ORANGE_LIGHT = "#eb6834", "#f5b08f"
INK, INK_MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
SETTING_COLOR = {"default": BLUE, "n_est1": ORANGE}
SETTING_LABEL = {"default": "default (8 estimators)", "n_est1": "N=1 (n_estimators=1)"}
DISPLAY_NAME = {
    "ke02_shuffle": "KE02 well - shuffled 80/20 split (gap filling)",
    "ke02_2025": "KE02 well - forecast 2025 (train < 2025-01-01)",
}
DISPLAY_SHORT = {"ke02_shuffle": "KE02 shuffled", "ke02_2025": "KE02 forecast 2025"}
METRIC_LABEL = {"accuracy": "Accuracy", "r2": "R2", "rmse": "RMSE (target units)"}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def _style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


FULL_SCALE = False  # --full-scale: fixed 0-1 axis, main metric only (no RMSE)


def _zoom(ax, values, higher_better=True):
    if FULL_SCALE:
        # 0-1 ticks; headroom above 1 only so the rotated value labels fit.
        ax.set_ylim(0, 1.16)
        ax.set_yticks(np.linspace(0, 1, 6))
        return
    lo, hi = float(np.min(values)), float(np.max(values))
    pad = max(hi - lo, abs(hi) * 1e-3) * 0.6
    ax.set_ylim(lo - pad, hi + pad * 1.4)  # extra headroom for value labels


def _fmt(v, metric):
    return f"{v:.3f}" if metric == "rmse" else f"{v:.4f}"


def dataset_chart(g: pd.DataFrame, name: str) -> Path:
    task = g.task.iloc[0]
    metrics = ["r2", "rmse"] if task == "regression" and not FULL_SCALE else [g.main_metric.iloc[0]]
    settings = [s for s in ("default", "n_est1") if s in set(g.setting)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(max(7.5, 6.2 * len(metrics)), 4.4),
                             squeeze=False)
    x = np.arange(6)
    w = 0.38
    for ax, metric in zip(axes[0], metrics):
        for k, s in enumerate(settings):
            gs = g[g.setting == s].sort_values("perm_seed")
            vals = gs[metric].to_numpy()
            pos = x + (k - (len(settings) - 1) / 2) * w
            ax.bar(pos, vals, width=w * 0.92, color=SETTING_COLOR[s], label=SETTING_LABEL[s])
            for p, v in zip(pos, vals):
                ax.text(p, v, _fmt(v, metric), ha="center", va="top" if v < 0 else "bottom",
                        rotation=90, fontsize=8, color=INK, clip_on=False)
        allvals = g[metric].to_numpy()
        if (allvals < 0).all():
            # Negative scores (KE02 forecast R2): bars must hang from 0, so the
            # axis spans [min, 0] instead of zooming, or they'd hang from the top.
            ax.set_ylim(allvals.min() * 1.3, 0)  # room below for the value labels
            ax.axhline(0, color=INK_MUTED, linewidth=1)
        else:
            _zoom(ax, allvals)
        # Baseline reference lines so every shuffle reads against its own baseline.
        for s in settings:
            b = g[(g.setting == s) & (g.perm_seed == 0)][metric].iloc[0]
            ax.axhline(b, color=SETTING_COLOR[s], linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xticks(x, ["baseline\n(original)"] + [f"shuffle {i}" for i in range(1, 6)])
        better = "lower is better" if metric == "rmse" else "higher is better"
        ax.set_ylabel(f"{METRIC_LABEL[metric]}  ({better})")
        _style(ax)
    r = g.iloc[0]
    fig.suptitle(f"{DISPLAY_NAME.get(name, name)}\n"
                 f"{task}, {r.n_features} features, {r.n_train} train / {r.n_test} test",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    axes[0][0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
                      fontsize=9)
    scale_note = "Y-axis 0-1" if FULL_SCALE else "Y-axis zoomed"
    note = (f"Dashed line = that setting's baseline. {scale_note}; values on bars are exact.\n"
            "TabPFN seed 42 and train/test split fixed in all runs.")
    bottom = 0.06
    # Few features: spell out every column order so each bar can be traced back.
    if "column_order" in g.columns and r.n_features <= 10:
        orders = g[g.setting == settings[0]].sort_values("perm_seed")
        lines = [("baseline" if s == 0 else f"shuffle {s}") + f":  {o}"
                 for s, o in zip(orders.perm_seed, orders.column_order)]
        note = "Column order per run:\n" + "\n".join(lines) + "\n\n" + note
        fig.set_figheight(fig.get_figheight() + 1.4)
        bottom = 0.27
    fig.text(0.01, 0.005, note, fontsize=8, color=INK_MUTED, family="monospace")
    fig.tight_layout(rect=(0, bottom, 1, 0.95))
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def split_settings_chart(g: pd.DataFrame, name: str) -> Path:
    """Default on the left, N=1 on the right, same y-range so the panels compare.

    Baseline bar is the setting's dark hue, shuffles its light tint.
    """
    # Accuracy or R2 only - this view is about the score people quote. The KE02
    # forecast's main metric is RMSE, but its R2 is shown here (negative).
    metric = "accuracy" if g.task.iloc[0] == "classification" else "r2"
    settings = [s for s in ("default", "n_est1") if s in set(g.setting)]
    dark = {"default": BLUE, "n_est1": ORANGE}
    light = {"default": BLUE_LIGHT, "n_est1": ORANGE_LIGHT}
    fig, axes = plt.subplots(1, len(settings), figsize=(6.2 * len(settings), 4.4),
                             squeeze=False, sharey=True)
    x = np.arange(6)
    for ax, s in zip(axes[0], settings):
        gs = g[g.setting == s].sort_values("perm_seed")
        vals = gs[metric].to_numpy()
        colors = [dark[s]] + [light[s]] * 5
        ax.bar(x, vals, width=0.62, color=colors)
        for p, v in zip(x, vals):
            ax.text(p, v, _fmt(v, metric), ha="center", va="top" if v < 0 else "bottom",
                    fontsize=8.5, color=INK)
        ax.axhline(vals[0], color=dark[s], linestyle="--", linewidth=1, alpha=0.7)
        sh = vals[1:]
        ax.set_title(f"{SETTING_LABEL[s]}\nshuffles: mean {_fmt(sh.mean(), metric)}, "
                     f"std {sh.std(ddof=1):.4f}, max change {np.max(np.abs(sh - vals[0])):.4f}",
                     loc="left", fontsize=9.5, color=INK)
        ax.set_xticks(x, ["baseline\n(original)"] + [f"shuffle {i}" for i in range(1, 6)])
        _style(ax)
    allvals = g[metric].to_numpy()
    if (allvals < 0).all():
        # Negative R2: bars hang from 0, axis spans [min, 0] - see dataset_chart.
        axes[0][0].set_ylim(allvals.min() * 1.3, 0)
        for ax in axes[0]:
            ax.axhline(0, color=INK_MUTED, linewidth=1)
    else:
        _zoom(axes[0][0], allvals)
    axes[0][0].set_ylabel(f"{METRIC_LABEL[metric]}  (higher is better)")
    r = g.iloc[0]
    fig.suptitle(f"{DISPLAY_NAME.get(name, name)}\n"
                 f"{r.task}, {r.n_features} features, {r.n_train} train / {r.n_test} test",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.005, "Dark bar = original column order, light bars = 5 shuffled orders; "
             "dashed line = baseline.\nBoth panels share the y-axis. TabPFN seed 42 and "
             "train/test split fixed in all runs.", fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def split_overall_chart(df: pd.DataFrame) -> Path:
    """All datasets: default panel left, N=1 panel right, shared y-axis.

    Per dataset: dark bar = baseline, light bar = mean of 5 shuffles (+/- std).
    Datasets with a negative score (KE02 forecast R2 ~ -2.7) are left out: on a
    shared axis they would flatten every other bar to nothing.
    """
    score = {"classification": "accuracy", "regression": "r2"}
    names = [n for n in dict.fromkeys(df.dataset)
             if (df[df.dataset == n][score[df[df.dataset == n].task.iloc[0]]] > 0).all()]
    dropped = [n for n in dict.fromkeys(df.dataset) if n not in names]
    settings = [s for s in ("default", "n_est1") if s in set(df.setting)]
    dark = {"default": BLUE, "n_est1": ORANGE}
    light = {"default": BLUE_LIGHT, "n_est1": ORANGE_LIGHT}
    fig, axes = plt.subplots(1, len(settings), figsize=(7.5 * len(settings), 5.0),
                             squeeze=False, sharey=True)
    x = np.arange(len(names))
    w = 0.38
    allv = []
    for ax, s in zip(axes[0], settings):
        for n_i, n in enumerate(names):
            g = df[(df.dataset == n) & (df.setting == s)]
            if g.empty:  # this dataset wasn't run with this setting
                continue
            m = score[g.task.iloc[0]]
            base = g[g.perm_seed == 0][m].iloc[0]
            sh = g[g.perm_seed > 0][m]
            mean, std = sh.mean(), sh.std(ddof=1)
            for k, (v, e, c) in enumerate([(base, 0, dark[s]), (mean, std, light[s])]):
                p = x[n_i] + (k - 0.5) * w
                ax.bar(p, v, width=w * 0.92, color=c, yerr=e if k else None,
                       error_kw={"ecolor": INK_MUTED, "elinewidth": 1, "capsize": 3})
                ax.text(p, v + e, f"{v:.4f}", ha="center", va="bottom", rotation=90, fontsize=8)
            allv += [base, mean - std, mean + std]
        ax.set_xticks(x, [f"{DISPLAY_SHORT.get(n, n)}\n({score[df[df.dataset == n].task.iloc[0]]})"
                          for n in names], fontsize=8.5)
        ax.set_title(SETTING_LABEL[s], loc="left", fontsize=10.5, color=INK)
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=dark[s]),
                           plt.Rectangle((0, 0), 1, 1, color=light[s])],
                  labels=["baseline (original order)", "mean of 5 shuffles (+/- std)"],
                  frameon=False, loc="upper right", fontsize=8.5, ncol=2)
        _style(ax)
    lo, hi = min(allv), max(allv)
    axes[0][0].set_ylim(lo - 0.03, hi + 0.045)
    axes[0][0].set_ylabel("Score (accuracy or R2, higher is better)")
    fig.suptitle("Feature order vs TabPFN-3 score, all datasets", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    note = "Both panels share the y-axis (zoomed); values on bars are exact."
    if dropped:
        note += (f"\nNot shown: {', '.join(DISPLAY_SHORT.get(n, n) for n in dropped)} "
                 "(negative R2 would flatten the shared axis) - see its own chart.")
    fig.text(0.01, 0.005, note, fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    path = OUT / "overall.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def overall_chart(df: pd.DataFrame) -> Path:
    names = list(dict.fromkeys(df.dataset))
    bars = [  # (setting, which, color, label)
        ("default", "baseline", BLUE, "default: baseline"),
        ("default", "shuffle", BLUE_LIGHT, "default: mean of 5 shuffles"),
        ("n_est1", "baseline", ORANGE, "N=1: baseline"),
        ("n_est1", "shuffle", ORANGE_LIGHT, "N=1: mean of 5 shuffles"),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(names))
    w = 0.2
    allv = []
    for k, (s, which, color, label) in enumerate(bars):
        vals, errs = [], []
        for n in names:
            g = df[(df.dataset == n) & (df.setting == s)]
            metric = g.main_metric.iloc[0]
            if which == "baseline":
                vals.append(g[g.perm_seed == 0][metric].iloc[0]); errs.append(0)
            else:
                sh = g[g.perm_seed > 0][metric]
                vals.append(sh.mean()); errs.append(sh.std(ddof=1))
        pos = x + (k - 1.5) * w
        ax.bar(pos, vals, width=w * 0.92, color=color, label=label,
               yerr=errs if which == "shuffle" else None,
               error_kw={"ecolor": INK_MUTED, "elinewidth": 1, "capsize": 3})
        for p, v, e in zip(pos, vals, errs):
            ax.text(p, v + e, f"{v:.4f}", ha="center", va="bottom", rotation=90, fontsize=8)
        allv += [v - e for v, e in zip(vals, errs)] + [v + e for v, e in zip(vals, errs)]
    if FULL_SCALE:
        _zoom(ax, allv)
    else:
        lo, hi = min(allv), max(allv)
        ax.set_ylim(lo - 0.02, hi + 0.035)
    metric_of = {n: METRIC_LABEL[df[df.dataset == n].main_metric.iloc[0]] for n in names}
    ax.set_xticks(x, [f"{n}\n({metric_of[n]})" for n in names])
    ax.set_ylabel("Score (accuracy or R2, higher is better)")
    _style(ax)
    ax.legend(frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    fig.suptitle("Feature order vs TabPFN-3 score, all datasets", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    scale_note = "Y-axis 0-1" if FULL_SCALE else "Y-axis zoomed"
    fig.text(0.01, 0.005, f"Error bars = std over the 5 shuffles. {scale_note}; values on bars are exact.",
             fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    path = OUT / "overall.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> int:
    global OUT, FULL_SCALE
    p = argparse.ArgumentParser()
    p.add_argument("--full-scale", action="store_true")
    p.add_argument("--out-dir", default=str(OUT))
    p.add_argument("--csv", default=str(ROOT / "results" / "feature_order.csv"))
    # The overall chart puts every dataset on one score axis; skip it when the
    # datasets' main metrics don't share a scale (KE02: R2 vs RMSE).
    p.add_argument("--no-overall", action="store_true")
    # One PNG per dataset with default and N=1 in separate side-by-side panels.
    p.add_argument("--split-settings", action="store_true")
    args = p.parse_args()
    FULL_SCALE, OUT = args.full_scale, Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    # Comma-separated: several result files are stacked into one frame.
    df = pd.concat([pd.read_csv(f) for f in args.csv.split(",")], ignore_index=True)
    chart = split_settings_chart if args.split_settings else dataset_chart
    for name, g in df.groupby("dataset", sort=False):
        print(chart(g, name))
    if not args.no_overall:
        print((split_overall_chart if args.split_settings else overall_chart)(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
