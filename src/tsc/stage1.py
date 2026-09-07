"""Stage 1: does redundancy or relevance bind under TabPFN's feature budget?

Four selectors at a fixed budget of 100 features, over 20 medium-sized UCR
datasets and 10 resamples. See src/tsc/selectors.py for the 2x2 that the arms
form and why mrmr must be in this stage rather than a later one.

Design notes:

  - The transform and the MI scores are computed ONCE per dataset-resample and
    shared by all four arms. MI is the expensive part and both the mutual_info
    and mrmr arms need it, so sharing it makes the fourth arm nearly free.
  - Every arm gets exactly 100 features. The dataset config requires
    n_train >= 120 so PCA is never capped - a capped arm would confound the
    comparison rather than just weaken it.
  - Redundancy and relevance are recorded per arm. Accuracy alone would make
    this a horse race; the mediators are what let it say WHY.

10 resamples, not 30, is deliberate. The Wilcoxon is paired across DATASETS, so
the test's n is 20 either way - resamples only sharpen each dataset's mean. The
GPU time is better spent on breadth than on precision we do not need.
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from benchmark import append_rows, gpu_name  # noqa: E402
from tsc.fs_probe import build_transformer  # noqa: E402
from tsc.feature_selectors import mediators, select  # noqa: E402

ARMS = ["random", "mutual_info", "mrmr", "pca"]

FIELDS = [
    "timestamp", "dataset", "ucr_type", "resample", "transformer", "arm",
    "top_n", "n_train", "n_test", "n_classes", "pool_features", "used_features",
    "accuracy", "balanced_accuracy", "redundancy", "relevance",
    "tabpfn_n_estimators", "transform_time", "mi_time", "select_time",
    "tabpfn_time", "seed", "device", "n_cpus", "gpu_name", "host", "notes",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=str(ROOT / "configs" / "ucr20_medium.json"))
    p.add_argument("--resample", type=int, default=0)
    p.add_argument("--transformer", default="MultiRocket")
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-dir", default=str(ROOT / "data" / "ucr"))
    p.add_argument("--out", default=None)
    p.add_argument("--only", default=None)
    p.add_argument("--tag", default="")
    args = p.parse_args()

    import os

    import torch
    from aeon.benchmarking.resampling import stratified_resample_data
    from aeon.datasets import load_classification
    from sklearn.feature_selection import mutual_info_classif
    from tabpfn import TabPFNClassifier

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but torch.cuda.is_available() is False")

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    seed = args.resample
    gpu, host = gpu_name(), socket.gethostname()
    arms = args.arms.split(",")

    cfg = json.loads(Path(args.datasets).read_text())
    entries = cfg["datasets"]
    if args.only:
        want = set(args.only.split(","))
        entries = [e for e in entries if e["name"] in want]

    out_path = (Path(args.out) if args.out
                else ROOT / "results" / "stage1" / f"resample_{args.resample:02d}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Resume at dataset granularity: a dataset is only skipped once every arm
    # for it is on disk, so a kill mid-dataset re-does that dataset cleanly
    # rather than leaving it with two of four arms.
    done: set[str] = set()
    if out_path.exists():
        rows = list(_csv.DictReader(out_path.open(newline="")))
        seen: dict[str, set[str]] = {}
        for r in rows:
            seen.setdefault(r["dataset"], set()).add(r["arm"])
        done = {d for d, a in seen.items() if set(arms) <= a}
        if done:
            print(f"[resume] {len(done)} datasets complete", flush=True)
    entries = [e for e in entries if e["name"] not in done]
    if not entries:
        print("[done] nothing left to run")
        return 0

    print(f"[run] {len(entries)} datasets  resample={args.resample}  "
          f"{args.transformer}  budget={args.top_n}  arms={','.join(arms)}",
          flush=True)

    failures = 0
    for e in entries:
        name = e["name"]
        try:
            Xtr, ytr = load_classification(name, split="train", extract_path=args.data_dir)
            Xte, yte = load_classification(name, split="test", extract_path=args.data_dir)
            if args.resample > 0:
                Xtr, ytr, Xte, yte = stratified_resample_data(
                    Xtr, ytr, Xte, yte, random_state=args.resample)

            t0 = time.perf_counter()
            tr = build_transformer(args.transformer, seed, n_cpus)
            Ztr = np.asarray(tr.fit_transform(Xtr), dtype=np.float64)
            Zte = np.asarray(tr.transform(Xte), dtype=np.float64)
            t_transform = time.perf_counter() - t0

            # Shared by the mutual_info and mrmr arms; the single most expensive
            # step, so computing it twice would nearly double the job.
            t0 = time.perf_counter()
            mi = (mutual_info_classif(Ztr, ytr, random_state=seed, n_jobs=n_cpus)
                  if {"mutual_info", "mrmr"} & set(arms) else None)
            t_mi = time.perf_counter() - t0
            print(f"  {name:28s} pool={Ztr.shape[1]} tf={t_transform:.0f}s "
                  f"mi={t_mi:.0f}s", flush=True)

            for arm in arms:
                t0 = time.perf_counter()
                Str, Ste, n_used, _ = select(arm, Ztr, ytr, Zte, args.top_n,
                                             seed, n_cpus, mi=mi)
                t_sel = time.perf_counter() - t0

                t0 = time.perf_counter()
                clf = TabPFNClassifier(device=args.device, random_state=seed)
                clf.fit(Str, ytr)
                y_hat = clf.predict(Ste)
                t_pfn = time.perf_counter() - t0

                med = mediators(Str, ytr, seed, n_cpus)
                acc = accuracy_score(yte, y_hat)
                append_rows(out_path, [{
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "dataset": name, "ucr_type": e["type"], "resample": args.resample,
                    "transformer": args.transformer, "arm": arm, "top_n": args.top_n,
                    "n_train": len(Xtr), "n_test": len(Xte),
                    "n_classes": int(len(np.unique(ytr))),
                    "pool_features": int(Ztr.shape[1]), "used_features": int(n_used),
                    "accuracy": float(acc),
                    "balanced_accuracy": float(balanced_accuracy_score(yte, y_hat)),
                    "redundancy": round(med["redundancy"], 5),
                    "relevance": round(med["relevance"], 5),
                    "tabpfn_n_estimators": int(clf.n_estimators_),
                    "transform_time": round(t_transform, 3),
                    "mi_time": round(t_mi, 3), "select_time": round(t_sel, 3),
                    "tabpfn_time": round(t_pfn, 3),
                    "seed": seed, "device": args.device, "n_cpus": n_cpus,
                    "gpu_name": gpu, "host": host, "notes": args.tag,
                }], FIELDS)
                print(f"      {arm:12s} acc={acc:.4f} red={med['redundancy']:.3f} "
                      f"rel={med['relevance']:.3f} sel={t_sel:5.1f}s", flush=True)
        except Exception as exc:
            failures += 1
            print(f"  {name:28s} FAILED: {type(exc).__name__}: {exc}", flush=True)
            append_rows(out_path, [{
                **{k: "" for k in FIELDS},
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "dataset": name, "ucr_type": e["type"], "resample": args.resample,
                "gpu_name": gpu, "host": host,
                "notes": f"{args.tag} FAILED {type(exc).__name__}: {exc}"[:300],
            }], FIELDS)
            if args.device == "cuda":
                torch.cuda.empty_cache()

    print(f"[done] -> {out_path}" + (f"  ({failures} failed)" if failures else ""),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
