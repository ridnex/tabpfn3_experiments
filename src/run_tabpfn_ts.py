"""TabPFN-TS on the KE02-01 forward split, for comparison with the tabular models.

Runs in its OWN venv (tabpfn-ts-venv): tabpfn-time-series pins pandas 2.3.3 while
the benchmark environment is on 3.0.5, so installing it into the shared env would
silently change every result produced so far.

Setup matches `ke02_2025` exactly - train < 2025-01-01, test = 2025, same dedup,
same 7 sensors - so the number is directly comparable to results_ke02_2025.csv.
The 7 sensors are passed as KNOWN covariates, which is the honest framing: at
prediction time a virtual flow meter really does have the sensor readings.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SENSORS = ["Choke", "DHP", "DHT", "FLP", "FLT", "FTHP", "FTHT"]
TARGET = "Total Massrate"
DATA = "/ibex/user/zharkyy/TabPFN/data/KE02-01_filtered.pq"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-context", type=int, default=32768)
    p.add_argument("--covariates", action="store_true",
                   help="pass the 7 sensors as known covariates")
    p.add_argument("--test-limit", type=int, default=0,
                   help="cap test rows (0 = all); for a quick smoke run")
    p.add_argument("--out", default="")
    args = p.parse_args()

    from tabpfn_time_series import TabPFNTSPipeline, TimeSeriesDataFrame

    df = (
        pd.read_parquet(DATA)
        .drop_duplicates()
        .sort_values("Datetime")
        .reset_index(drop=True)
    )
    cols = [TARGET] + (SENSORS if args.covariates else [])
    ts = df[["Datetime", *cols]].rename(columns={"Datetime": "timestamp", TARGET: "target"})
    ts["item_id"] = "KE02-01"
    tsdf = TimeSeriesDataFrame.from_data_frame(ts, id_column="item_id", timestamp_column="timestamp")

    cut = pd.Timestamp("2025-01-01")
    stamps = tsdf.index.get_level_values("timestamp")
    context = tsdf[stamps < cut]
    future = tsdf[stamps >= cut].copy()
    if args.test_limit:
        future = future.iloc[: args.test_limit]
    y_true = future["target"].to_numpy(dtype="float64")
    # The horizon must not carry the answer. Covariates stay; target is blanked.
    future["target"] = np.nan

    print(f"[data] context={len(context)}  horizon={len(future)}  "
          f"covariates={'7 sensors' if args.covariates else 'none'}", flush=True)

    pipe = TabPFNTSPipeline(max_context_length=args.max_context, tabpfn_mode="local")
    t0 = time.perf_counter()
    pred = pipe.predict(context, future)
    elapsed = time.perf_counter() - t0

    col = "target" if "target" in pred.columns else pred.columns[0]
    y_hat = np.asarray(pred[col], dtype="float64")[: len(y_true)]
    ok = np.isfinite(y_hat)
    if not ok.all():
        print(f"[warn] {(~ok).sum()} non-finite predictions dropped", flush=True)

    rmse = float(np.sqrt(mean_squared_error(y_true[ok], y_hat[ok])))
    print(f"\n[tabpfn_ts] R2   = {r2_score(y_true[ok], y_hat[ok]):.4f}")
    print(f"[tabpfn_ts] RMSE = {rmse:.1f}")
    print(f"[tabpfn_ts] MAE  = {mean_absolute_error(y_true[ok], y_hat[ok]):.1f}")
    print(f"[tabpfn_ts] bias = {y_hat[ok].mean() - y_true[ok].mean():+.0f}")
    print(f"[tabpfn_ts] time = {elapsed:.1f}s   (fit+predict, not separable in this API)")

    if args.out:
        pd.DataFrame({"y_true": y_true[ok], "y_pred": y_hat[ok]}).to_parquet(args.out, index=False)
        print(f"[done] predictions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
