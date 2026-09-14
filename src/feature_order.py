"""Does the ORDER of input columns change TabPFN-3's score?

Everything is held fixed except the column permutation:
  * one 80/20 split (split seed 0, stratified for classification), built once
    per dataset and reused by every run;
  * TabPFN random_state = 42 on every run;
  * perm_seed 0 = original order (baseline), perm_seed 1..5 = random orders.

TabPFN's default ensemble (8 members) already shuffles features internally per
member, which could average an order effect away. Every run is therefore done
twice: default settings and n_estimators=1.

Verdict per dataset x setting: "meaningful" if any shuffle's main score (accuracy
or R2) differs from baseline by more than 1% of the baseline score.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets as ds_mod  # noqa: E402
from benchmark import RESULTS_DIR, gpu_name, score_regression  # noqa: E402

DATASETS = ["spambase", "mfeat_fourier", "superconductivity", "scm20d"]
SETTINGS = {"default": {}, "n_est1": {"n_estimators": 1}}
TABPFN_SEED = 42
SPLIT_SEED = 0
PERM_SEEDS = [0, 1, 2, 3, 4, 5]  # 0 = original order
REL_THRESHOLD = 0.01


def score_clf(y_true, proba) -> dict:
    """Binary or multiclass. benchmark.score_classification is binary-only."""
    proba = np.asarray(proba, dtype="float64")
    proba = proba / proba.sum(axis=1, keepdims=True)
    labels = list(range(proba.shape[1]))
    if proba.shape[1] == 2:
        auc = roc_auc_score(y_true, proba[:, 1])
    else:
        auc = roc_auc_score(y_true, proba, multi_class="ovr", labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, proba.argmax(axis=1))),
        "roc_auc": float(auc),
        "log_loss": float(log_loss(y_true, proba, labels=labels)),
    }


def permutation(perm_seed: int, n: int) -> np.ndarray:
    return np.arange(n) if perm_seed == 0 else np.random.default_rng(perm_seed).permutation(n)


def run_one(task, X_tr, y_tr, X_te, params):
    import torch
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    Est = TabPFNClassifier if task == "classification" else TabPFNRegressor
    model = Est(device="cuda", random_state=TABPFN_SEED, **params)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    out = model.predict_proba(X_te) if task == "classification" else model.predict(X_te)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return np.asarray(out, dtype="float64"), t1 - t0, t2 - t1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--out", default=str(RESULTS_DIR / "feature_order.csv"))
    p.add_argument("--settings", default=",".join(SETTINGS))
    args = p.parse_args()
    settings = {s: SETTINGS[s] for s in args.settings.split(",")}

    gpu = gpu_name()
    rows = []
    for name in args.datasets.split(","):
        X, y, meta = ds_mod.load(name)
        columns = list(X.columns)
        X = X.to_numpy(dtype="float64")
        y = y.to_numpy()
        task = meta.task
        clf = task == "classification"
        split_seed = SPLIT_SEED
        main_metric = "accuracy" if clf else "r2"
        if meta.cv == "holdout":
            # Forecast: rows are time-sorted by load(); train strictly before split_at.
            n_tr = meta.holdout_n_train
            X_tr, X_te, y_tr, y_te = X[:n_tr], X[n_tr:], y[:n_tr], y[n_tr:]
            split_seed = -1
            # R2 is negative on this split, so a % of it is meaningless; use RMSE.
            main_metric = "rmse"
        elif meta.cv == "shuffle":
            # Same split as benchmark.py (index split, seed 42), so baselines match
            # results/results_ke02_shuffle.csv.
            split_seed = 42
            tr, te = train_test_split(np.arange(len(X)), test_size=0.2, shuffle=True,
                                      random_state=split_seed)
            X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=SPLIT_SEED, stratify=y if clf else None
            )
        n_feat = X.shape[1]
        perms = {s: permutation(s, n_feat) for s in PERM_SEEDS}
        shuffled = [tuple(perms[s]) for s in PERM_SEEDS[1:]]
        assert len(set(shuffled)) == len(shuffled), "permutations not distinct"
        assert tuple(range(n_feat)) not in shuffled, "a shuffle is the identity"
        print(f"\n=== {name}: {task}, train {X_tr.shape}, test {X_te.shape}", flush=True)

        for setting, params in settings.items():
            base_out = base_score = None
            for s in PERM_SEEDS:
                perm = perms[s]
                out, t_fit, t_pred = run_one(task, X_tr[:, perm], y_tr, X_te[:, perm], params)
                metrics = score_clf(y_te, out) if clf else score_regression(y_te, out)
                if s == 0:
                    base_out, base_score = out, metrics[main_metric]
                row = {
                    "dataset": name, "task": task, "n_train": len(X_tr), "n_test": len(X_te),
                    "n_features": n_feat, "setting": setting, "perm_seed": s,
                    "tabpfn_seed": TABPFN_SEED, "split_seed": split_seed,
                    "main_metric": main_metric, "score": metrics[main_metric],
                    "abs_diff": abs(metrics[main_metric] - base_score),
                    "rel_diff": abs(metrics[main_metric] - base_score) / abs(base_score),
                    **metrics,
                }
                d = np.abs(out - base_out)
                if clf:
                    row["max_abs_dproba"] = float(d.max())
                    row["label_flip_frac"] = float(np.mean(out.argmax(1) != base_out.argmax(1)))
                else:
                    row["max_abs_dpred"] = float(d.max())
                    row["mean_abs_dpred"] = float(d.mean())
                row.update(fit_time=t_fit, predict_time=t_pred, gpu=gpu,
                           column_order=" | ".join(columns[i] for i in perm),
                           perm_index=" ".join(map(str, perm)))
                rows.append(row)
                print(f"  {setting:8s} perm {s}: {main_metric}={row['score']:.5f} "
                      f"abs_diff={row['abs_diff']:.5f} fit={t_fit:.1f}s pred={t_pred:.1f}s",
                      flush=True)

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    summ = []
    for (name, setting), g in df.groupby(["dataset", "setting"], sort=False):
        sh = g[g.perm_seed > 0]
        summ.append({
            "dataset": name, "task": g.task.iloc[0], "setting": setting,
            "main_metric": g.main_metric.iloc[0],
            "baseline": g[g.perm_seed == 0].score.iloc[0],
            "shuffle_mean": sh.score.mean(), "shuffle_std": sh.score.std(ddof=1),
            "max_abs_diff": sh.abs_diff.max(), "max_rel_diff": sh.rel_diff.max(),
            "meaningful": bool(sh.rel_diff.max() > REL_THRESHOLD),
        })
    summ = pd.DataFrame(summ)
    summ.to_csv(out_path.with_name(out_path.stem + "_summary.csv"), index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print("\n", df[["dataset", "setting", "perm_seed", "main_metric", "score",
                        "abs_diff", "rel_diff"]].to_string(index=False))
        print("\n", summ.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
