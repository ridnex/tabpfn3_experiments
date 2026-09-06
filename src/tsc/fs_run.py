"""MultiRocket features -> PCA-100 -> TabPFN, over the 20-dataset subset.

The counterpart to run.py, which measures RocketPFN. Same datasets, same
resamples, same published baselines - so this column drops into the existing
comparison rather than needing one of its own.

The probe that motivated this saw MultiRocket+PCA beat RocketPFN on Fish by
four test series out of 175. That is not a margin anything can be built on,
which is the whole reason for running 30 resamples across 20 datasets here.

Note what this design cannot separate: it differs from RocketPFN in BOTH the
transformer (ROCKET -> MultiRocket) and the strategy (grouping -> selection).
A win is a win for the pipeline as a whole, not evidence about which half
caused it. ROCKET+PCA would be the clean one-variable test.
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
from tsc.fs_probe import build_transformer, select_features  # noqa: E402

FIELDS = [
    "timestamp", "dataset", "ucr_type", "resample", "transformer", "selector",
    "top_n", "n_train", "n_test", "n_classes",
    "pool_features", "used_features", "components_capped",
    "accuracy", "balanced_accuracy", "tabpfn_n_estimators",
    "transform_time", "select_time", "tabpfn_time", "total_time",
    "seed", "device", "n_cpus", "gpu_name", "host", "notes",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=str(ROOT / "configs" / "ucr20.json"))
    p.add_argument("--resample", type=int, default=0)
    p.add_argument("--transformer", default="MultiRocket")
    p.add_argument("--selector", default="pca")
    p.add_argument("--top-n", type=int, default=100)
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
    from tabpfn import TabPFNClassifier

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but torch.cuda.is_available() is False")

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    seed = args.resample
    gpu, host = gpu_name(), socket.gethostname()

    cfg = json.loads(Path(args.datasets).read_text())
    entries = cfg["datasets"]
    if args.only:
        want = set(args.only.split(","))
        entries = [e for e in entries if e["name"] in want]

    out_path = (
        Path(args.out) if args.out
        else ROOT / "results" / "fs" / f"resample_{args.resample:02d}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        with out_path.open(newline="") as f:
            done = {r["dataset"] for r in _csv.DictReader(f)}
        if done:
            print(f"[resume] {len(done)} datasets already done", flush=True)
    entries = [e for e in entries if e["name"] not in done]
    if not entries:
        print("[done] nothing left to run")
        return 0

    print(f"[run] {len(entries)} datasets  resample={args.resample}  "
          f"{args.transformer}+{args.selector}-{args.top_n}  gpu={gpu}", flush=True)

    failures = 0
    for e in entries:
        name = e["name"]
        try:
            Xtr, ytr = load_classification(name, split="train", extract_path=args.data_dir)
            Xte, yte = load_classification(name, split="test", extract_path=args.data_dir)
            if args.resample > 0:
                Xtr, ytr, Xte, yte = stratified_resample_data(
                    Xtr, ytr, Xte, yte, random_state=args.resample
                )

            t0 = time.perf_counter()
            tr = build_transformer(args.transformer, seed, n_cpus)
            Ztr = np.asarray(tr.fit_transform(Xtr), dtype=np.float64)
            Zte = np.asarray(tr.transform(Xte), dtype=np.float64)
            t_transform = time.perf_counter() - t0

            t0 = time.perf_counter()
            Str, Ste, n_used = select_features(args.selector, Ztr, ytr, Zte,
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
                "dataset": name, "ucr_type": e["type"], "resample": args.resample,
                "transformer": args.transformer, "selector": args.selector,
                "top_n": args.top_n,
                "n_train": len(Xtr), "n_test": len(Xte),
                "n_classes": int(len(np.unique(ytr))),
                "pool_features": int(Ztr.shape[1]), "used_features": int(n_used),
                # PCA cannot exceed min(n_samples, n_features): BeetleFly, Rock
                # and ShapeletSim have 20 training series, so they get 20
                # components, not 100. Flagged per row so the analysis can say
                # so instead of presenting them as comparable "top-100" cells.
                "components_capped": int(n_used < args.top_n),
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
            cap = " (capped)" if n_used < args.top_n else ""
            print(f"  {name:28s} acc={acc:.4f}  used={n_used}{cap}  "
                  f"tf={t_transform:6.1f}s sel={t_select:6.1f}s "
                  f"pfn={t_pfn:5.1f}s", flush=True)
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
