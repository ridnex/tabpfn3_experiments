"""Identify OpenML dataset IDs by downloading parquet directly from data.openml.org.

The OpenML metadata API is returning 504, so we bypass it: the data server still
serves parquet by ID, and we identify datasets by their schema.
"""
import io
import sys
import urllib.request

import pandas as pd


def url_for(did: int) -> str:
    return f"https://data.openml.org/datasets/{did // 10000:04d}/{did:04d}/dataset_{did}.pq"


def probe(did: int, max_bytes: int = 20_000_000):
    try:
        req = urllib.request.Request(url_for(did), headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=30) as r:
            size = int(r.headers.get("content-length") or 0)
            if size > max_bytes:
                return did, size, None, None
            raw = r.read()
        df = pd.read_parquet(io.BytesIO(raw))
        return did, len(raw), df.shape, list(df.columns)
    except Exception as e:
        return did, None, None, f"ERR {type(e).__name__}"


if __name__ == "__main__":
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    for did in range(lo, hi + 1):
        did, size, shape, cols = probe(did)
        if shape is None:
            continue
        print(f"{did}\t{shape}\t{cols}", flush=True)
