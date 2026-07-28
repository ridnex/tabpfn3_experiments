"""Benchmark runner: TabPFN-3 vs XGBoost vs LightGBM on OpenML-CTR23 regression.

Protocol (all decided before any code was written):
  * 10-fold CV, deterministic KFold(shuffle=True, random_state=SEED).
    NOTE: OpenML's predefined task splits were unreachable (API 504) when this
    was built, so folds are generated locally. Reproducible here, but not
    bit-identical to published CTR23 numbers.
  * Metrics per fold: RMSE, MAE, R2. R2 is the headline (scale-free, so it
    aggregates across datasets with different units).
  * Timing per fold: fit / predict / total, with checkpoint download and CUDA
    context creation pre-warmed OUT of the measurement.
  * Results are append-only, keyed by (dataset, model, fold, seed, hardware),
    so adding a dataset later never invalidates earlier rows.
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

import datasets as ds_mod  # noqa: E402
import models as models_mod  # noqa: E402

SEED = 42
N_FOLDS = 10
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

_COMMON_HEAD = [
    "timestamp", "dataset", "openml_id", "n_rows", "n_features", "model", "device",
    # n_train/n_test are per-fold. Under plain KFold they are constant and dull,
    # but forward-chaining grows the train set ~5x across folds and TabPFN's
    # predict time scales with it (~n^2) - a timing column without the train size
    # it was measured at is uninterpretable.
    "fold", "n_train", "n_test", "seed",
]
_COMMON_TAIL = [
    # fit_time == search_time + refit_time. Untuned models have search_time 0 and
    # refit_time == fit_time, so the three columns are always consistent.
    "search_time", "refit_time", "fit_time", "predict_time", "total_time",
    "n_cpus", "gpu_name", "host", "torch_version", "notes",
]
FIELDS = _COMMON_HEAD + ["rmse", "mae", "r2"] + _COMMON_TAIL
CLF_FIELDS = _COMMON_HEAD + ["accuracy", "balanced_accuracy", "roc_auc", "log_loss"] + _COMMON_TAIL


def score_regression(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def score_classification(y_true, proba) -> dict:
    """Binary metrics from a single predict_proba result.

    roc_auc and log_loss read the POSITIVE-class column; passing labels or the
    wrong column yields a plausible ~0.5 rather than an error, so the column
    index is pinned here in one place rather than at each call site.
    """
    proba = np.asarray(proba, dtype="float64")
    if proba.ndim != 2 or proba.shape[1] != 2:
        raise ValueError(f"expected binary proba of shape (n, 2), got {proba.shape}")
    # XGBoost returns float32 probabilities whose rows miss 1.0 by ~3e-8, while
    # LightGBM's are exact. sklearn renormalises internally and warns, so the
    # numbers were never wrong - but only XGBoost was being renormalised, which
    # is an asymmetry between models we are comparing head-to-head. Normalise
    # here so every model's log-loss is computed on identical footing.
    proba = proba / proba.sum(axis=1, keepdims=True)
    y_hat = proba.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_hat)),
        "roc_auc": float(roc_auc_score(y_true, proba[:, 1])),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
    }


def gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return "none"


def warmup(
    names: list[str], tabpfn_device: str, gbdt_device: str, task: str = "regression"
) -> None:
    """Pay one-off costs BEFORE timing: checkpoint download, CUDA context, JIT.

    Without this, whichever model runs first absorbs several seconds of setup
    that has nothing to do with its actual speed.

    Only warms the models actually selected - warming TabPFN for a GBDT-only run
    would need the license token for no reason.
    """
    rng = np.random.RandomState(0)
    X = rng.rand(64, 4)
    # Both classes present, or the classifiers refuse to fit.
    y = np.tile([0, 1], 32) if task == "classification" else rng.rand(64)
    clf = task == "classification"
    devices = {tabpfn_device if n == "tabpfn_v3" else gbdt_device for n in names}
    if any(n.endswith("_gpu") for n in names):
        devices.add("cuda")
    print(f"[warmup] starting for {names}...", flush=True)

    if "cuda" in devices:
        import torch

        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
            print(f"[warmup] cuda ok: {torch.cuda.get_device_name(0)}", flush=True)
        else:
            raise SystemExit(
                "cuda requested but torch.cuda.is_available() is False - refusing to "
                "silently fall back to CPU and report it as GPU timing"
            )

    if "tabpfn_v3" in names:
        if clf:
            from tabpfn import TabPFNClassifier as TabPFN
        else:
            from tabpfn import TabPFNRegressor as TabPFN

        m = TabPFN(device=tabpfn_device, random_state=0)
        m.fit(X, y)
        m.predict(X)
        print("[warmup] tabpfn ok", flush=True)

    if any(n.startswith("xgb") for n in names):
        import xgboost

        Est = xgboost.XGBClassifier if clf else xgboost.XGBRegressor
        for dev in devices:
            Est(n_estimators=5, device=dev).fit(X, y)
    if any(n.startswith("lgbm") for n in names):
        import lightgbm

        Est = lightgbm.LGBMClassifier if clf else lightgbm.LGBMRegressor
        Est(n_estimators=5, verbose=-1).fit(X, y)
    print("[warmup] gbdt ok", flush=True)


def append_rows(path: Path, rows: list[dict], fields: list[str] = FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    if not new:
        # DictWriter writes values in `fields` order and never re-reads the
        # header, so appending under a different schema silently produces rows
        # whose columns are shifted relative to everything above them. That is
        # unrecoverable after the fact - the old rows and new rows look equally
        # valid. Refuse instead.
        with path.open(newline="") as f:
            existing = next(csv.reader(f), [])
        if existing != fields:
            raise SystemExit(
                f"{path} has header {existing}\n"
                f"but this run writes  {fields}\n"
                "Refusing to append - it would misalign every row. Use --out to "
                "write a new file."
            )
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="concrete")
    p.add_argument("--models", default="all")
    p.add_argument("--folds", type=int, default=N_FOLDS)
    p.add_argument(
        "--gap", type=int, default=0,
        help="timeseries CV only: rows dropped between train end and test start. "
             "On autocorrelated data the last training row would otherwise be a "
             "near-twin of the first test row.",
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--gbdt-device", default="cpu")
    p.add_argument("--tabpfn-device", default="cuda")
    p.add_argument("--out", default=None)
    p.add_argument(
        "--save-preds", default=None,
        help="parquet path for per-row test predictions. Lets a single test block "
             "be sliced after the fact (by month, by regime) without refitting.",
    )
    p.add_argument("--tag", default="", help="free-text note recorded on every row")
    args = p.parse_args()

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

    X, y, meta = ds_mod.load(args.dataset)
    task = meta.task
    clf = task == "classification"
    fields = CLF_FIELDS if clf else FIELDS
    if meta.cv != "kfold":
        # Its own file: these rows are not comparable with the random-KFold rungs
        # and must not be averaged in with them by a later summarize call.
        # Keyed off "not kfold" rather than a list of time-based modes, so a new
        # mode cannot forget to opt in and end up aimed at results.csv.
        default_out = f"results_{meta.name}.csv"
    else:
        default_out = "results_clf.csv" if clf else "results.csv"
    out_path = Path(args.out) if args.out else RESULTS_DIR / default_out
    print(f"[data] {meta.name}: X={X.shape} target={meta.target!r} task={task}", flush=True)

    registry = models_mod.build_registry(args.gbdt_device, args.tabpfn_device, task, meta.cv)
    names = list(registry) if args.models == "all" else args.models.split(",")
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise SystemExit(f"unknown models {unknown}; known: {sorted(registry)}")

    warmup(names, args.tabpfn_device, args.gbdt_device, task)

    import torch

    torch_v = torch.__version__
    gpu = gpu_name()
    host = socket.gethostname()

    Xv = X.to_numpy(dtype="float64")
    yv = y.to_numpy(dtype="int64" if clf else "float64")
    if meta.cv == "timeseries":
        # Forward chaining: train on the past, test on the future, expanding the
        # train window each fold. Relies on datasets.load() having sorted by
        # time_col, which is why that is a documented post-condition there.
        #
        # Shuffling here would be actively wrong, not merely different: on
        # autocorrelated data a random split puts each test row's own neighbour
        # in train, and every model scores ~0.99 for copying it.
        kf = TimeSeriesSplit(n_splits=args.folds, gap=args.gap)
        folds = list(kf.split(Xv))
        print(f"[cv] TimeSeriesSplit(n_splits={args.folds}, gap={args.gap})", flush=True)
        for k, (tr, te) in enumerate(folds):
            print(f"  fold {k}: train[0:{tr[-1] + 1}] ({len(tr)})  "
                  f"test[{te[0]}:{te[-1] + 1}] ({len(te)})  "
                  f"gap={te[0] - tr[-1] - 1}", flush=True)
    elif meta.cv == "shuffle":
        # One random 80/20 split. Correct for gap-filling, wrong for forecasting -
        # see the ke02_shuffle registry comment.
        from sklearn.model_selection import train_test_split

        tr, te = train_test_split(
            np.arange(len(Xv)), test_size=0.2, shuffle=True, random_state=args.seed
        )
        folds = [(np.sort(tr), np.sort(te))]
        print(f"[cv] shuffled 80/20  seed={args.seed}  "
              f"train={len(tr)}  test={len(te)}", flush=True)
    elif meta.cv == "holdout":
        # One split at a fixed date, resolved by datasets.load() into a row index.
        n_tr = meta.holdout_n_train
        if n_tr <= args.gap:
            raise SystemExit(f"--gap {args.gap} would empty the {n_tr}-row train side")
        folds = [(np.arange(0, n_tr - args.gap), np.arange(n_tr, len(Xv)))]
        tr, te = folds[0]
        print(f"[cv] holdout at {meta.split_at}  gap={args.gap}", flush=True)
        print(f"  train[0:{len(tr)}] ({len(tr)})  test[{te[0]}:{len(Xv)}] ({len(te)})",
              flush=True)
    else:
        # Stratified for classification so every fold keeps the class ratio; same
        # seed and shuffle as every regression rung, so folds stay comparable.
        splitter = StratifiedKFold if clf else KFold
        kf = splitter(n_splits=args.folds, shuffle=True, random_state=args.seed)
        folds = list(kf.split(Xv, yv))

    rows: list[dict] = []
    preds: list[dict] = []
    for name in names:
        runner = registry[name]
        per_fold = []
        for k, (tr, te) in enumerate(folds):
            y_out, t_fit, t_pred, info = runner.run(
                Xv[tr], yv[tr], Xv[te], args.seed, n_cpus
            )
            metrics = (score_classification if clf else score_regression)(yv[te], y_out)
            if args.save_preds and not clf:
                # `row` is the positional index into the time-sorted matrix, which
                # is what makes a later join back to the timestamps unambiguous.
                preds.append({
                    "model": name, "fold": k, "row": te,
                    "y_true": yv[te], "y_pred": np.asarray(y_out, dtype="float64"),
                })
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "dataset": meta.name, "openml_id": meta.openml_id,
                "n_rows": len(Xv), "n_features": Xv.shape[1],
                "model": name, "device": runner.device, "fold": k,
                "n_train": len(tr), "n_test": len(te), "seed": args.seed,
                **metrics,
                "search_time": info.get("search_time", 0.0),
                "refit_time": info.get("refit_time", t_fit),
                "fit_time": t_fit, "predict_time": t_pred, "total_time": t_fit + t_pred,
                "n_cpus": n_cpus, "gpu_name": gpu, "host": host,
                "torch_version": torch_v,
                "notes": args.tag or (str(info) if info else ""),
            }
            rows.append(row)
            per_fold.append(row)
            headline = (
                f"auc={row['roc_auc']:.4f}  acc={row['accuracy']:.4f}"
                if clf
                else f"r2={row['r2']:.4f}  rmse={row['rmse']:8.3f}"
            )
            print(
                f"  {name:16s} fold {k:2d}  n_tr={len(tr):7d}  {headline}  "
                f"fit={t_fit:7.3f}s  pred={t_pred:7.3f}s",
                flush=True,
            )
        # Flush after each model rather than once at the end: a run that hits its
        # Slurm time limit then keeps everything finished so far instead of
        # losing all of it.
        append_rows(out_path, per_fold, fields)

        key = "roc_auc" if clf else "r2"
        r2s = np.array([r[key] for r in per_fold])
        tot = np.array([r["total_time"] for r in per_fold])
        print(
            f"[{name}] {key} = {r2s.mean():.4f} +/- {r2s.std():.4f}   "
            f"total = {tot.mean():.3f}s +/- {tot.std():.3f}s   "
            f"[{len(per_fold)} rows saved]\n",
            flush=True,
        )

    print(f"[done] appended {len(rows)} rows -> {out_path}", flush=True)

    if preds:
        import pandas as pd

        pred_path = Path(args.save_preds)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        # Written once at the end rather than incrementally: parquet has no append,
        # and a partial file here costs nothing since the CSV is already flushed
        # per model and holds every metric.
        pd.concat([pd.DataFrame(d) for d in preds], ignore_index=True).to_parquet(
            pred_path, index=False
        )
        print(f"[done] wrote predictions -> {pred_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
