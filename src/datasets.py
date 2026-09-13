"""Dataset registry + loader.

Datasets come from the OpenML-CTR23 regression suite. The OpenML *metadata* API
was down (504) when this was built, so we bypass it entirely: parquet files are
fetched straight from data.openml.org by dataset ID and cached locally. Nothing
here needs the API, which also means compute nodes never depend on it.

Adding a dataset for the next rung of the size ladder = add one REGISTRY entry.
"""
from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Dataset:
    name: str
    openml_id: int  # 0 means "not from OpenML" - see local_file
    target: str
    n_rows: int  # expected, used as a sanity check on load
    task: str = "regression"  # or "classification" - drives metrics, CV and models
    drop: tuple[str, ...] = ()  # columns removed from X (leakage / ID columns)
    # Convert non-numeric columns to a numeric matrix. Off by default so that a
    # new dataset with categoricals fails loudly rather than being silently
    # encoded in a way nobody chose. See _encode() for what it actually does.
    encode_categoricals: bool = False
    # --- local (non-OpenML) sources ---
    local_file: str | None = None  # filename inside DATA_DIR; skips the download
    sheet: str | None = None  # worksheet name, for .xlsx sources
    # --- non-iid data ---
    # Ordering column. When set, load() sorts by it and removes it from X, so the
    # POST-CONDITION "row order == time order" holds for everything downstream.
    # That is what lets benchmark.py split on positional indices alone.
    time_col: str | None = None
    dedup: bool = False  # drop exact duplicate rows before the n_rows check
    # "kfold"      - random folds (iid data only)
    # "timeseries" - forward chaining, expanding window
    # "holdout"    - one split at split_at: everything before is train, from it test
    cv: str = "kfold"
    split_at: str | None = None  # ISO date, cv="holdout" only
    # Resolved by load(), never set in the registry: the row index where the
    # holdout test block starts. Lives on the meta so benchmark.py can build the
    # split without re-reading the timestamps it deliberately dropped from X.
    holdout_n_train: int = 0
    derive: str | None = None  # key into _DERIVERS; adds engineered feature columns


# Sizes verified by direct schema inspection of the parquet files.
REGISTRY: dict[str, Dataset] = {
    # --- ~1k rung (current) ---
    "concrete": Dataset("concrete", 44959, "strength", 1030),
    # --- future rungs, verified present on the data server ---
    "airfoil": Dataset("airfoil", 44957, "sound_pressure", 1503),
    "abalone": Dataset("abalone", 44956, "rings", 4177),
    # ~5k rung. Target `quality` is integer-valued (3-9), i.e. a discretised
    # regression target - keep that in mind when reading R2.
    "white_wine": Dataset("white_wine", 44971, "quality", 4898),
    "red_wine": Dataset("red_wine", 44972, "quality", 1599),
    "cpu_activity": Dataset("cpu_activity", 44978, "usr", 8192),
    "grid_stability": Dataset("grid_stability", 44973, "stab", 10000),
    "naval_propulsion": Dataset("naval_propulsion", 44969, "gt_turbine_decay_state_coefficient", 11934),
    "california_housing": Dataset("california_housing", 44977, "medianHouseValue", 20640),
    "superconductivity": Dataset("superconductivity", 44964, "critical_temp", 21263),
    "protein": Dataset("protein", 44963, "RMSD", 45730),
    "sarcos": Dataset("sarcos", 44976, "V28", 48933),
    "diamonds": Dataset("diamonds", 44979, "price", 53940),
    "video_transcoding": Dataset("video_transcoding", 44974, "utime", 68784),
    # LEAKAGE WARNING: Run1..Run4 are four repeated timing measurements of the
    # SAME kernel configuration. Leaving Run2..Run4 in X while predicting Run1
    # gives every model R^2 ~= 0.99 and is meaningless. They must be dropped.
    "sgemm": Dataset("sgemm", 44961, "Run1", 241600, drop=("Run2", "Run3", "Run4")),
    # --- classification rungs ---
    # ~45k. Binary UP/DOWN electricity price move. Picked over `adult` (48,842)
    # and `bank-marketing` (45,211) because it is the only one of the three with
    # NO missing values, a near-balanced target (42.5% minority) and a single
    # low-cardinality categorical - so the run measures the models rather than
    # our imputation and encoding choices.
    "electricity": Dataset(
        "electricity", 151, "class", 45312,
        task="classification", encode_categoricals=True,
    ),
    # --- feature-order probe (src/feature_order.py): 50-200 features each ---
    "spambase": Dataset("spambase", 44, "class", 4601, task="classification"),
    "mfeat_fourier": Dataset("mfeat_fourier", 14, "class", 2000, task="classification"),
    # Two target columns in the file; latitude is the target, so longitude must
    # leave X or it hands the model the answer's twin.
    "music_origin": Dataset("music_origin", 44965, "latitude", 1059, drop=("longitude",)),
    # Multi-target supply-chain forecasting (16 targets). LBL is the target; the
    # other 15 are future prices of the same products, so they must leave X.
    "scm20d": Dataset(
        "scm20d", 41486, "LBL", 8966,
        drop=tuple(f"MTLp{i}A" for i in range(2, 17)),
    ),
    # ~581k rung: NYC green taxi, Dec 2016. Two hazards, both handled above:
    #   LEAKAGE: `total_amount` = fare + extra + mta_tax + tolls + surcharge +
    #     tip_amount, i.e. it contains the target additively. Verified against the
    #     data: the implied fare lands in $2.50-100 for 99.8% of rows. Dropped.
    #   CATEGORICALS: 9 non-numeric columns, incl. PULocationID (233 levels) and
    #     DOLocationID (259). First dataset in the ladder to need encoding.
    "nyc_taxi": Dataset(
        "nyc_taxi", 44986, "tip_amount", 581835,
        drop=("total_amount",), encode_categoricals=True,
    ),
    "large_1m": Dataset("large_1m", 44998, "TARGET_UNVERIFIED", 1000000),
    # --- real production data, supplied by the user (not OpenML) ---
    # Oil well KE02-01, virtual flow metering: infer mass flow rate from seven
    # pressure/temperature sensors. 10-minute cadence, 2019-02-13 to 2025-09-29.
    #
    # NOT IID - and the harness must not pretend otherwise. Lag-1 autocorrelation
    # of the target is 0.9977 and consecutive rows differ by a median of 54 units
    # against an overall std of 4,960. Under random KFold ~90% of test rows have
    # their own 10-minute neighbour in train, which hands every model R^2 = 0.99
    # for copying a neighbour. Held-out *time periods* give R^2 of -0.27 (middle
    # block) and -1.51 (last block) with the same model. Hence cv="timeseries".
    #
    # n_rows is the post-dedup count: 9,406 of the 135,354 raw rows are exact
    # duplicates, one per repeated timestamp.
    "ke02_well": Dataset(
        "ke02_well", 0, "Total Massrate", 125948,
        local_file="KE02-01_filtered.xlsx", sheet="KE02-01",
        time_col="Datetime", dedup=True, cv="timeseries",
    ),
    # Same file, same columns, different question: train on everything before 2025
    # and predict 2025. The operational framing ("would this have worked if I'd
    # built it in January?") rather than the five-fold sweep above.
    #
    # Note what is NOT the problem here: 2025's target range (15,006-37,344) sits
    # entirely inside the training range, so no test row needs extrapolation past
    # what trees can represent. The difficulty is distribution shift - test std is
    # 1,353 against train std 5,162, so the model must predict a narrow low band
    # that is rare in training.
    "ke02_2025": Dataset(
        "ke02_2025", 0, "Total Massrate", 125948,
        local_file="KE02-01_filtered.xlsx", sheet="KE02-01",
        time_col="Datetime", dedup=True, cv="holdout", split_at="2025-01-01",
    ),
    # Same well, SHUFFLED 80/20 split. This answers a different question from
    # ke02_2025: "the meter dropped out for an hour, can I backfill it from the
    # sensors, given labelled data before and after?" - gap filling, not
    # forecasting. In that deployment the neighbouring rows really are available,
    # so the high score it produces is earned rather than leaked.
    #
    # It would be leakage if quoted as a forecasting result: consecutive rows are
    # 10 minutes apart and differ by a median of 54 units against a std of 4,960,
    # so a shuffled test row almost always has its own near-twin in train.
    # Deduplication does NOT prevent this - the twins are near-identical, never
    # exactly equal, and the 9,406 exact duplicates were already removed anyway.
    "ke02_shuffle": Dataset(
        "ke02_shuffle", 0, "Total Massrate", 125948,
        local_file="KE02-01_filtered.xlsx", sheet="KE02-01",
        time_col="Datetime", dedup=True, cv="shuffle",
    ),
    # Identical to ke02_2025 in every way except the seven engineered physics
    # columns, so the pair is a clean A/B on feature engineering alone.
    "ke02_2025_phys": Dataset(
        "ke02_2025_phys", 0, "Total Massrate", 125948,
        local_file="KE02-01_filtered.xlsx", sheet="KE02-01",
        time_col="Datetime", dedup=True, cv="holdout", split_at="2025-01-01",
        derive="well_physics",
    ),
}


def _encode(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Make every column numeric, and report how each one was handled.

    Two distinct cases hide behind pandas' `category` dtype:

      * Columns that are genuinely numeric but were *stored* as categories
        (`extra` = 0.5/1.0, `mta_tax` = 0.5). Ordinal-coding these would throw
        away their magnitude, so they are parsed back to floats.
      * Columns that are real categories (`PULocationID`, `VendorID`). These get
        integer codes. That imposes a false ordering, but one-hot is not an
        option: ~500 dummies would exceed TabPFN's 200-feature ceiling at this
        row count. Every model receives the identical matrix, so the comparison
        stays like-for-like even though no model gets native categorical support.
    """
    how: dict[str, str] = {}
    out = X.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            how[c] = "numeric"
            continue
        as_num = pd.to_numeric(out[c].astype(str), errors="coerce")
        if as_num.notna().all():
            out[c] = as_num.astype("float64")
            how[c] = "numeric-as-category -> float"
        else:
            out[c] = out[c].astype("category").cat.codes.astype("int32")
            how[c] = f"ordinal codes ({out[c].nunique()} levels)"
    return out, how


def _derive_well_physics(X: pd.DataFrame) -> pd.DataFrame:
    """Add pressure/temperature differentials for a producing oil well.

    Flow through a choke is driven by the pressure DIFFERENCE across it, not by
    either absolute pressure. With only raw sensor columns a tree has to
    reconstruct that difference from axis-aligned splits, which it cannot do
    exactly. These columns hand it over directly.

    The second reason to try them here is drift: absolute pressures fall as the
    well depletes, so a model keyed on them goes stale. A differential may be more
    stable in time even while its inputs move, which is exactly the failure mode
    seen on the forward split.

    Nothing is dropped - the seven raw sensors stay, so this can only add
    information, never remove it.
    """
    out = X.copy()
    dp_choke = X["FTHP"] - X["FLP"]  # across the choke: the main flow driver
    out["dP_choke"] = dp_choke
    out["dP_tubing"] = X["DHP"] - X["FTHP"]  # bottomhole -> wellhead
    out["dT_tubing"] = X["DHT"] - X["FTHT"]
    out["dT_flowline"] = X["FTHT"] - X["FLT"]
    out["ratio_THP_DHP"] = X["FTHP"] / X["DHP"]
    # Choke flow correlation: rate ~ area * sqrt(dP), area ~ opening^2.
    out["choke_area"] = X["Choke"] ** 2
    out["choke_flow"] = out["choke_area"] * np.sqrt(dp_choke.clip(lower=0))
    return out


_DERIVERS = {"well_physics": _derive_well_physics}


def _url(openml_id: int) -> str:
    return (
        f"https://data.openml.org/datasets/"
        f"{openml_id // 10000:04d}/{openml_id:04d}/dataset_{openml_id}.pq"
    )


def _read_local(ds: Dataset, data_dir: Path) -> pd.DataFrame:
    """Read a user-supplied file, caching the parse as parquet.

    Parsing the xlsx costs ~10s through openpyxl and every Slurm job would pay it
    again. Caching also pins the parse: one file on disk that all folds and all
    models read, rather than re-deriving it per job.
    """
    src = data_dir / ds.local_file
    if not src.exists():
        raise FileNotFoundError(f"{ds.name}: {src} not found - copy the file into {data_dir}/")
    # Keyed on the SOURCE file, not the registry name, so several entries over the
    # same spreadsheet (different targets, different CV) share one parsed parquet.
    cache = data_dir / f"{src.stem}.pq"
    if not cache.exists() or cache.stat().st_mtime < src.stat().st_mtime:
        if src.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(src, sheet_name=ds.sheet, engine="openpyxl")
        elif src.suffix.lower() == ".csv":
            df = pd.read_csv(src)
        else:
            raise ValueError(f"{ds.name}: unsupported file type {src.suffix!r}")
        df.to_parquet(cache, index=False)
        print(f"[cache] {ds.name}: parsed {src.name} -> {cache.name} ({len(df)} rows)", flush=True)
    return pd.read_parquet(cache)


def load(name: str, data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.Series, Dataset]:
    """Return (X, y, meta). Downloads and caches the parquet on first use."""
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(REGISTRY)}")
    ds = REGISTRY[name]
    data_dir.mkdir(parents=True, exist_ok=True)

    if ds.local_file:
        df = _read_local(ds, data_dir)
    else:
        cache = data_dir / f"{ds.name}_{ds.openml_id}.pq"
        if not cache.exists():
            req = urllib.request.Request(_url(ds.openml_id), headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = r.read()
            # Validate before caching so a truncated download can't poison the cache.
            pd.read_parquet(io.BytesIO(raw))
            cache.write_bytes(raw)
        df = pd.read_parquet(cache)

    if ds.dedup:
        before = len(df)
        df = df.drop_duplicates()
        print(f"[dedup] {ds.name}: {before} -> {len(df)} rows "
              f"({before - len(df)} exact duplicates removed)", flush=True)

    # Checked AFTER dedup, so the registry records the row count that actually
    # reaches the models rather than the raw file's.
    if len(df) != ds.n_rows:
        raise ValueError(f"{ds.name}: expected {ds.n_rows} rows, got {len(df)}")

    if ds.time_col:
        if ds.time_col not in df.columns:
            raise ValueError(f"{ds.name}: time_col {ds.time_col!r} not in {list(df.columns)}")
        # Establishes the post-condition documented on Dataset.time_col: from here
        # on, positional row order IS time order. benchmark.py's TimeSeriesSplit
        # depends on this and has no other way to check it.
        df = df.sort_values(ds.time_col, kind="mergesort").reset_index(drop=True)

    if ds.split_at:
        if not ds.time_col:
            raise ValueError(f"{ds.name}: split_at needs time_col to be set")
        n_tr = int((df[ds.time_col] < pd.Timestamp(ds.split_at)).sum())
        if not 0 < n_tr < len(df):
            raise ValueError(
                f"{ds.name}: split_at {ds.split_at} leaves an empty side "
                f"({n_tr} train, {len(df) - n_tr} test)"
            )
        # replace() rather than assignment: Dataset is frozen, and this is the
        # only computed field, so returning a modified copy keeps load()'s
        # signature and the registry's immutability both intact.
        ds = replace(ds, holdout_n_train=n_tr)
        print(f"[holdout] {ds.name}: train {n_tr} rows (< {ds.split_at}), "
              f"test {len(df) - n_tr} rows", flush=True)

    if ds.target not in df.columns:
        raise ValueError(f"{ds.name}: target {ds.target!r} not in {list(df.columns)}")

    missing_drop = [c for c in ds.drop if c not in df.columns]
    if missing_drop:
        raise ValueError(f"{ds.name}: drop columns not present: {missing_drop}")

    if ds.task == "classification":
        # Map labels through *sorted* unique values rather than pandas' category
        # order, so class 0/1 means the same thing on every run and machine.
        # roc_auc/log_loss both key off proba[:, 1], so a drifting label order
        # would silently invert the metric instead of erroring.
        labels = sorted(df[ds.target].astype(str).unique())
        y = df[ds.target].astype(str).map({v: i for i, v in enumerate(labels)}).astype("int64")
        print(f"[labels] {ds.name}: {dict(enumerate(labels))}", flush=True)
    else:
        y = df[ds.target].astype("float64")
    # time_col is deliberately NOT a feature. It was only ever an ordering key:
    # under a forward split every test timestamp lies outside the training range,
    # so a tree would saturate at its last leaf, and under a random split it is a
    # near-perfect index into the neighbouring-row lookup. A flow meter reads
    # sensors, not the clock.
    X = df.drop(columns=[ds.target, *ds.drop, *([ds.time_col] if ds.time_col else [])])

    if ds.derive:
        if ds.derive not in _DERIVERS:
            raise KeyError(f"{ds.name}: unknown deriver {ds.derive!r}; known: {sorted(_DERIVERS)}")
        before = list(X.columns)
        X = _DERIVERS[ds.derive](X)
        added = [c for c in X.columns if c not in before]
        if not np.isfinite(X[added].to_numpy(dtype="float64")).all():
            raise ValueError(f"{ds.name}: deriver {ds.derive!r} produced NaN/inf in {added}")
        print(f"[derive] {ds.name}: +{len(added)} features {added}", flush=True)

    if ds.encode_categoricals:
        X, how = _encode(X)
        for col, method in how.items():
            if method != "numeric":
                print(f"[encode] {ds.name}.{col}: {method}", flush=True)

    # The harness converts X with to_numpy(float64); a string column would either
    # blow up here or be silently coerced. Fail loudly instead - handling
    # categoricals is a deliberate decision, not something to stumble into.
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(
            f"{ds.name}: non-numeric feature columns {non_numeric}. "
            "Categorical handling is not implemented - TabPFN wants raw categoricals "
            "while GBDT needs encoding, so this needs an explicit protocol decision."
        )
    return X, y, ds
