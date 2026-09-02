"""Download the UCR subset to a local cache before any GPU job runs.

Zenodo returns intermittent 504s. A GPU job that hits one dies partway through
a resample, having burned the allocation - so the download happens once here,
on a login node, with retries, and the compute nodes only ever read from disk.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ucr"
RETRIES = 5

sys.path.insert(0, str(ROOT / "src"))
from aeon.datasets import load_classification  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(ROOT / "configs" / "ucr20.json"))
args = ap.parse_args()

cfg = json.loads(Path(args.config).read_text())
names = [d["name"] for d in cfg["datasets"]]
DATA.mkdir(parents=True, exist_ok=True)

failed = []
for name in names:
    for attempt in range(1, RETRIES + 1):
        try:
            Xtr, ytr = load_classification(name, split="train", extract_path=str(DATA))
            Xte, yte = load_classification(name, split="test", extract_path=str(DATA))
            # A truncated download parses as a VALID but EMPTY .ts file - Rock
            # came back as train=0, L=0 and was accepted on the first pass. Check
            # the shape against the UCR summary rather than trusting "no
            # exception", and delete the bad cache so the retry re-downloads.
            want = next(d for d in cfg["datasets"] if d["name"] == name)
            if len(Xtr) != want["n_train"] or Xtr.shape[-1] != want["length"]:
                shutil.rmtree(DATA / name, ignore_errors=True)
                raise ValueError(
                    f"got train={len(Xtr)} L={Xtr.shape[-1]}, "
                    f"expected train={want['n_train']} L={want['length']}"
                )
            print(f"  {name:28s} train={len(Xtr):5d} test={len(Xte):5d} "
                  f"L={Xtr.shape[-1]:5d}", flush=True)
            break
        except Exception as exc:
            print(f"  {name:28s} attempt {attempt}/{RETRIES} failed: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            # Linear backoff: the failures are gateway timeouts under load, not
            # rate limiting, so aggressive exponential waits just idle longer.
            time.sleep(10 * attempt)
    else:
        failed.append(name)

if failed:
    print(f"\nFAILED to fetch {len(failed)}: {failed}")
    sys.exit(1)
print(f"\n[done] {len(names)} datasets cached -> {DATA}")
