"""Rocket features -> feature selection -> TabPFN, as a probe on one dataset.

RocketPFN handles TabPFN's per-member feature budget by GROUPING: 10 passes over
2000 features each. This asks whether SELECTING instead - keep the best 100 of
10,000 - does as well for a tenth of the work.

Deliberately one dataset and one split. The point is to find out whether the
idea has legs and what it costs, not to produce a benchmark number; the cost
measurement is the reason it is not worth guessing at MI's price first.

Baselines are not rerun here. For Fish we already have ROCKET+Ridge (0.9800,
published) and RocketPFN (0.9642, measured), which anchor the comparison. What
is genuinely missing is "all features + TabPFN" - so this run cannot say whether
selection beats no-selection, only how it lands against ROCKET-based methods.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from benchmark import append_rows, gpu_name  # noqa: E402

FIELDS = [
    "timestamp", "dataset", "resample", "transformer", "selector", "top_n",
    "n_train", "n_test", "n_classes",
    "pool_features", "used_features",
    "accuracy", "balanced_accuracy", "tabpfn_n_estimators",
    "transform_time", "select_time", "tabpfn_time", "total_time",
    "seed", "device", "n_cpus", "gpu_name", "host", "notes",
]

TRANSFORMERS = ["ROCKET", "MiniRocket", "MultiRocket"]
SELECTORS = ["mutual_info", "pca"]


def build_transformer(name: str, seed: int, n_jobs: int):
    from aeon.transformations.collection.convolution_based import (
        MiniRocket, MultiRocket, Rocket,
    )

    if name == "ROCKET":
        # 10000 kernels, not RocketPFN's 1000-per-group: this pipeline is about
        # selecting from a large pool, so the pool has to be large. Two features
        # per kernel means 20000 columns.
        return Rocket(n_kernels=10000, random_state=seed, n_jobs=n_jobs)
    if name == "MiniRocket":
        return MiniRocket(n_kernels=10000, random_state=seed, n_jobs=n_jobs)
    if name == "MultiRocket":
        return MultiRocket(n_kernels=6250, random_state=seed, n_jobs=n_jobs)
    raise ValueError(name)


def select_features(kind, Ztr, ytr, Zte, top_n, seed, n_jobs):
    """Return (train, test, n_used). Fitted on TRAIN ONLY - test never informs it."""
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import StandardScaler

    if kind == "mutual_info":
        scores = mutual_info_classif(Ztr, ytr, random_state=seed, n_jobs=n_jobs)
        idx = np.argsort(scores)[-top_n:]
        return Ztr[:, idx], Zte[:, idx], len(idx)

    if kind == "pca":
        # PCA cannot produce more components than min(n_samples, n_features), so
        # on a 20-series training set it caps at 20 however large top_n is. Cap
        # explicitly and RECORD the number actually used - silently returning
        # fewer columns than MI got would make the two look comparable when they
        # are not.
        k = min(top_n, Ztr.shape[0], Ztr.shape[1])
        Ztr_s = StandardScaler().fit(Ztr)
        p = PCA(n_components=k, random_state=seed).fit(Ztr_s.transform(Ztr))
        return p.transform(Ztr_s.transform(Ztr)), p.transform(Ztr_s.transform(Zte)), k

    raise ValueError(kind)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Fish")
    p.add_argument("--resample", type=int, default=0)
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--transformers", default=",".join(TRANSFORMERS))
    p.add_argument("--selectors", default=",".join(SELECTORS))
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-dir", default=str(ROOT / "data" / "ucr"))
    p.add_argument("--out", default=str(ROOT / "results" / "fs_probe.csv"))
    p.add_argument("--tag", default="")
    args = p.parse_args()

    import os
    import socket

    import torch
    from aeon.benchmarking.resampling import stratified_resample_data
    from aeon.datasets import load_classification
    from tabpfn import TabPFNClassifier

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but torch.cuda.is_available() is False")

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    seed = args.resample
    gpu, host = gpu_name(), socket.gethostname()

    Xtr, ytr = load_classification(args.dataset, split="train", extract_path=args.data_dir)
    Xte, yte = load_classification(args.dataset, split="test", extract_path=args.data_dir)
    if args.resample > 0:
        Xtr, ytr, Xte, yte = stratified_resample_data(Xtr, ytr, Xte, yte,
                                                      random_state=args.resample)
    print(f"[data] {args.dataset}  train={len(Xtr)} test={len(Xte)} "
          f"L={Xtr.shape[-1]} classes={len(np.unique(ytr))}  cpus={n_cpus}", flush=True)

    out_path = Path(args.out)
    for tname in args.transformers.split(","):
        t0 = time.perf_counter()
        tr = build_transformer(tname, seed, n_cpus)
        Ztr = np.asarray(tr.fit_transform(Xtr), dtype=np.float64)
        Zte = np.asarray(tr.transform(Xte), dtype=np.float64)
        t_transform = time.perf_counter() - t0
        print(f"\n[{tname}] pool={Ztr.shape[1]} features  transform={t_transform:.1f}s",
              flush=True)

        for sname in args.selectors.split(","):
            t0 = time.perf_counter()
            Str, Ste, n_used = select_features(sname, Ztr, ytr, Zte,
                                               args.top_n, seed, n_cpus)
            t_select = time.perf_counter() - t0

            t0 = time.perf_counter()
            clf = TabPFNClassifier(device=args.device, random_state=seed)
            clf.fit(Str, ytr)
            y_hat = clf.predict(Ste)
            t_pfn = time.perf_counter() - t0

            acc = accuracy_score(yte, y_hat)
            append_rows(out_path, [{
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "dataset": args.dataset, "resample": args.resample,
                "transformer": tname, "selector": sname, "top_n": args.top_n,
                "n_train": len(Xtr), "n_test": len(Xte),
                "n_classes": int(len(np.unique(ytr))),
                "pool_features": int(Ztr.shape[1]), "used_features": int(n_used),
                "accuracy": float(acc),
                "balanced_accuracy": float(balanced_accuracy_score(yte, y_hat)),
                "tabpfn_n_estimators": int(clf.n_estimators_),
                "transform_time": round(t_transform, 3),
                "select_time": round(t_select, 3),
                "tabpfn_time": round(t_pfn, 3),
                "total_time": round(t_transform + t_select + t_pfn, 3),
                "seed": seed, "device": args.device, "n_cpus": n_cpus,
                "gpu_name": gpu, "host": host, "notes": args.tag,
            }], FIELDS)
            print(f"  {sname:12s} used={n_used:4d}  acc={acc:.4f}  "
                  f"select={t_select:7.1f}s  tabpfn={t_pfn:6.1f}s", flush=True)

    print(f"\n[done] -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
