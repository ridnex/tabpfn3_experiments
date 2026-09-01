"""Run RocketPFN over a fixed UCR subset, one resample per invocation.

One resample index per process on purpose: the validation run (step 5) and the
30-resample sweep (step 6) are then the SAME code path, the sweep is a Slurm
array over --resample, and a failed array element is re-runnable without
touching the rest.

Resample 0 is the archive's default train/test split - the same convention the
published results files use, so our column lines up with theirs index for index.
Resamples 1..29 are stratified resamples of the pooled data, matching the
protocol behind the published 30-resample numbers.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import warnings

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# TabPFN warns once per group that it auto-scaled its ensemble to cover 2000
# features. That is expected and recorded in tabpfn_n_estimators; without this
# it prints 200 times per resample and buries the actual results.
warnings.filterwarnings("once", message="Auto-scaling n_estimators")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from benchmark import append_rows, gpu_name  # noqa: E402  reuse, don't reimplement

FIELDS = [
    "timestamp", "dataset", "ucr_type", "resample", "n_train", "n_test",
    "series_length", "n_channels", "n_classes",
    "accuracy", "balanced_accuracy",
    "n_groups", "n_kernels", "tabpfn_n_estimators",
    "feature_time", "tabpfn_time", "total_time",
    "seed", "device", "n_cpus", "gpu_name", "host", "torch_version",
    "tabpfn_version", "aeon_version", "notes",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=str(ROOT / "configs" / "ucr20.json"))
    p.add_argument("--resample", type=int, default=0)
    p.add_argument("--groups", type=int, default=10)
    p.add_argument("--kernels", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--out", default=None,
        help="default results/rocketpfn/resample_NN.csv. One file per resample "
             "because 30 concurrent array tasks appending to a single CSV "
             "interleave mid-row and corrupt it. analyze.py globs them back.",
    )
    p.add_argument("--only", default=None, help="comma-separated subset, for debugging")
    p.add_argument(
        "--data-dir", default=str(ROOT / "data" / "ucr"),
        help="local .ts cache, populated by scripts/prefetch_ucr.py. Compute nodes "
             "read from here instead of hitting Zenodo mid-run, which has already "
             "cost one job a 504 partway through the sweep.",
    )
    p.add_argument("--tag", default="")
    args = p.parse_args()

    import aeon
    import tabpfn
    import torch
    from aeon.benchmarking.resampling import stratified_resample_data
    from aeon.datasets import load_classification

    from tsc.rocketpfn import RocketPFNClassifier

    if args.device == "cuda" and not torch.cuda.is_available():
        # Falling back to CPU would still produce numbers, just 50x slower ones
        # carrying a "cuda" label. Refuse rather than mislabel a timing column.
        raise SystemExit("--device cuda but torch.cuda.is_available() is False")

    import os

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    cfg = json.loads(Path(args.datasets).read_text())
    entries = cfg["datasets"]
    if args.only:
        want = set(args.only.split(","))
        entries = [e for e in entries if e["name"] in want]
        if not entries:
            raise SystemExit(f"--only matched nothing in {args.datasets}")

    out_path = (
        Path(args.out) if args.out
        else ROOT / "results" / "rocketpfn" / f"resample_{args.resample:02d}.csv"
    )
    # Resume rather than redo: a task that hits its walltime keeps every dataset
    # it finished, and resubmitting picks up where it stopped. Matters more than
    # usual here - the queue wait, not the compute, is the scarce resource.
    done: set[str] = set()
    if out_path.exists():
        import csv as _csv

        with out_path.open(newline="") as f:
            done = {r["dataset"] for r in _csv.DictReader(f)}
        if done:
            print(f"[resume] {len(done)} datasets already in {out_path.name}", flush=True)
    entries = [e for e in entries if e["name"] not in done]
    if not entries:
        print("[done] nothing left to run")
        return 0
    gpu, host = gpu_name(), socket.gethostname()
    print(f"[run] {len(entries)} datasets  resample={args.resample}  "
          f"G={args.groups}x{args.kernels}  device={args.device}  gpu={gpu}", flush=True)

    # Pay the checkpoint download and CUDA context once, outside every timing.
    t0 = time.perf_counter()
    warm = RocketPFNClassifier(n_groups=1, n_kernels=50, device=args.device, random_state=0)
    Xw = np.random.RandomState(0).randn(16, 1, 32)
    warm.fit(Xw, np.tile([0, 1], 8))
    warm.predict(Xw[:4])
    print(f"[warmup] {time.perf_counter() - t0:.1f}s", flush=True)

    for e in entries:
        name = e["name"]
        Xtr, ytr = load_classification(name, split="train", extract_path=args.data_dir)
        Xte, yte = load_classification(name, split="test", extract_path=args.data_dir)
        if args.resample > 0:
            Xtr, ytr, Xte, yte = stratified_resample_data(
                Xtr, ytr, Xte, yte, random_state=args.resample
            )

        clf = RocketPFNClassifier(
            n_groups=args.groups, n_kernels=args.kernels,
            device=args.device, random_state=args.resample, n_jobs=n_cpus,
        )
        t0 = time.perf_counter()
        clf.fit(Xtr, ytr)
        y_hat = clf.predict(Xte)
        total = time.perf_counter() - t0

        acc = accuracy_score(yte, y_hat)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": name, "ucr_type": e["type"], "resample": args.resample,
            "n_train": len(Xtr), "n_test": len(Xte),
            "series_length": Xtr.shape[-1], "n_channels": Xtr.shape[1],
            "n_classes": len(np.unique(ytr)),
            "accuracy": float(acc),
            "balanced_accuracy": float(balanced_accuracy_score(yte, y_hat)),
            "n_groups": args.groups, "n_kernels": args.kernels,
            # One number per group under v3's auto-scaling; joined rather than
            # averaged so an outlier group stays visible.
            "tabpfn_n_estimators": "|".join(str(x) for x in clf.n_estimators_),
            "feature_time": round(clf.timings_["feature_time"], 3),
            "tabpfn_time": round(clf.timings_["tabpfn_time"], 3),
            "total_time": round(total, 3),
            "seed": args.resample, "device": args.device, "n_cpus": n_cpus,
            "gpu_name": gpu, "host": host, "torch_version": torch.__version__,
            "tabpfn_version": tabpfn.__version__, "aeon_version": aeon.__version__,
            "notes": args.tag,
        }
        # Flushed per dataset: a job that hits its Slurm limit keeps everything
        # it finished instead of losing the lot.
        append_rows(out_path, [row], FIELDS)
        print(f"  {name:28s} acc={acc:.4f}  feat={row['feature_time']:7.1f}s  "
              f"pfn={row['tabpfn_time']:7.1f}s  total={total:7.1f}s", flush=True)

    print(f"[done] -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
