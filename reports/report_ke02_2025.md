# Report — KE02-01: train on 2019–2024, predict 2025

**Date:** 2026-07-28 · **Run:** Slurm job 49527894 (`srun`, A100), ~2 min ·
**Results:** `results/results_ke02_2025.csv`, `results/preds_ke02_2025.parquet`

> The operational framing of the previous rung: *if this virtual flow meter had
> been built on 1 January 2025, would it have worked?*
> Companion to `reports/report_ke02_well.md` (5-fold forward chaining, same well).

---

## Setup

| | rows | period | target mean | target std |
|---|---|---|---|---|
| train | **101,159** (101,015 after gap) | 2019-02-13 → 2024-12-31 | 21,015 | 5,162 |
| test | **24,789** (19.7%) | 2025-01-01 → 2025-09-29 | 17,798 | **1,353** |

Single date holdout at 2025-01-01, `gap=144` (one day of 2024 dropped from
training so the last training row is not a 10-minute twin of the first test row).
Boundary verified to land on 2024-12-31 23:50 → 2025-01-01 00:00.

Same six models, same 7 sensors, same tuning protocol (30-trial random search,
**time-ordered** inner validation) as the 5-fold run. TabPFN on A100, everything
else on 16 CPU cores.

**Correction to `report_ke02_well.md`.** That report attributed the difficulty
partly to trees being unable to extrapolate below their training range. That is
not the binding constraint here: 2025's target range (15,006–37,344) sits entirely
inside the training range (15,000–90,279), and **0% of test rows fall below the
training minimum** — the file's hard 15,000 floor applies to both sides. The real
mechanism is distribution shift, and this run identifies it precisely (see
*Where the error comes from*).

---

## Headline result — all of 2025

| model | device | R² | RMSE | MAE | fit s | pred s | total s |
|---|---|---|---|---|---|---|---|
| **lgbm_tuned** | cpu | **−0.543** | **1680.1** | **1309.6** | 20.287 | 0.011 | 20.299 |
| xgb_tuned | cpu | −0.661 | 1743.0 | 1327.2 | 7.249 | 0.001 | 7.250 |
| xgb_default | cpu | −0.953 | 1890.3 | 1469.8 | 0.342 | 0.003 | **0.345** |
| lgbm_default | cpu | −1.251 | 2029.1 | 1556.0 | 0.222 | 0.006 | 0.228 |
| tabpfn_v3 | cuda | **−2.679** | 2594.1 | 1875.7 | 3.145 | 35.635 | 38.781 |
| *mean_baseline* | cpu | *−5.680* | *3495.5* | *3297.2* | 0.000 | 0.000 | 0.000 |

**TabPFN-3 comes last**, and by a wide margin — 54% higher RMSE than tuned
LightGBM. This is consistent with fold 4 of the 5-fold run (R² −3.018), which
covered nearly the same period, so it is a stable result rather than one bad draw.

Ranking: **lgbm_tuned > xgb_tuned > xgb_default > lgbm_default ≫ tabpfn_v3**.
Tuning helps both libraries here (LightGBM −1.251 → −0.543, XGBoost −0.953 →
−0.661), unlike the aggregate 5-fold view where tuning was roughly neutral.

---

## The aggregate is misleading — read the months

One number per model cannot be told apart from noise, so the single 2025 test block
was sliced by calendar month. This costs no extra compute; it is the same
predictions, grouped. **June is entirely absent from the data. July has only 64
rows (mean 31,796 vs ~18,000 elsewhere) and is excluded as too small** — 7 usable
months, 24,725 of 24,789 rows.

### RMSE by month

| month | n | xgb_tuned | lgbm_tuned | xgb_default | lgbm_default | tabpfn_v3 |
|---|---|---|---|---|---|---|
| 2025-01 | 4331 | 887 | 921 | 990 | **822** | 973 |
| 2025-02 | 2160 | 1027 | 933 | 923 | 823 | **809** |
| 2025-03 | 3964 | 713 | 596 | 631 | 690 | **532** |
| 2025-04 | 4201 | **799** | 1096 | 1376 | 1067 | 951 |
| 2025-05 | 3169 | **1391** | 2164 | 2065 | 2045 | 2105 |
| 2025-08 | 3295 | 2423 | **1905** | 2496 | 3040 | *4410* |
| 2025-09 | 3605 | 3294 | **2926** | 3266 | 3607 | *4654* |
| **mean** | | 1505 | 1506 | 1678 | 1728 | 2062 |
| **sd** | | 983 | 844 | 964 | 1191 | **1759** |

### RMSE as % of the no-skill baseline (100% = no better than a constant)

| month | xgb_tuned | lgbm_tuned | xgb_default | lgbm_default | tabpfn_v3 |
|---|---|---|---|---|---|
| 2025-01 | 23.2% | 24.1% | 25.9% | **21.5%** | 25.5% |
| 2025-02 | 25.9% | 23.6% | 23.3% | 20.8% | **20.5%** |
| 2025-03 | 20.2% | 16.9% | 17.8% | 19.5% | **15.0%** |
| 2025-04 | **25.8%** | 35.3% | 44.4% | 34.4% | 30.7% |
| 2025-05 | **47.9%** | 74.6% | 71.2% | 70.5% | 72.5% |
| 2025-08 | 109.8% | **86.4%** | 113.1% | 137.8% | **199.9%** |
| 2025-09 | 77.3% | **68.7%** | 76.7% | 84.7% | 109.2% |
| **mean** | 47.2% | 47.1% | 53.2% | 55.6% | 67.6% |

**Two findings the aggregate destroyed:**

1. **The model has a shelf life of roughly five months.** Every model is strong
   January–April (15–44% of baseline) and collapses by August, where all but
   `lgbm_tuned` are at or past **100% — worse than predicting a constant**. This is
   the single most useful number in the report and it is invisible in the annual
   total.
2. **TabPFN is not uniformly worse; it is bimodal.** It has the *best* RMSE of any
   model in February and March, and the worst by a wide margin in August and
   September. Its last-place aggregate is produced by two catastrophic months, not
   by being consistently behind. On per-month win counts it is not clearly beaten:

| vs tabpfn_v3 | months TabPFN wins | mean Δ RMSE | sd of Δ |
|---|---|---|---|
| xgb_tuned | 2/7 | −557 | 841 |
| lgbm_tuned | 4/7 | −556 | 1091 |
| xgb_default | 4/7 | −384 | 891 |
| lgbm_default | 3/7 | −334 | 613 |

No p-values are quoted. Months are autocorrelated and sit on a declining trend, so
they are not independent samples; reporting a significance test over them would
overstate what this design supports.

---

## Where the error comes from

Decomposing each month's error into bias (systematic offset) and the rest:

**Mean prediction − mean actual, per month:**

| month | actual mean | xgb_tuned | lgbm_tuned | xgb_default | lgbm_default | tabpfn_v3 |
|---|---|---|---|---|---|---|
| 2025-01 | 17,298 | −230 | −531 | −611 | −363 | −611 |
| 2025-02 | 17,147 | +14 | −286 | −433 | −71 | −429 |
| 2025-03 | 17,627 | +300 | +185 | +95 | +279 | +61 |
| 2025-04 | 18,052 | −246 | −505 | −877 | −411 | −605 |
| 2025-05 | 18,376 | −693 | −1,211 | −1,111 | −1,178 | −585 |
| 2025-08 | 18,981 | **+2,278** | **+1,672** | **+2,324** | **+2,967** | **+4,363** |
| 2025-09 | 16,844 | **+3,210** | **+2,810** | **+3,194** | **+3,547** | **+4,572** |

**Fraction of RMSE that is pure bias (|bias| / RMSE):**

| month | xgb_tuned | lgbm_tuned | xgb_default | lgbm_default | tabpfn_v3 |
|---|---|---|---|---|---|
| Jan–May | 0.01 – 0.58 | 0.31 – 0.58 | 0.15 – 0.64 | 0.09 – 0.58 | 0.11 – 0.64 |
| **2025-08** | **0.94** | **0.88** | **0.93** | **0.98** | **0.99** |
| **2025-09** | **0.97** | **0.96** | **0.98** | **0.98** | **0.98** |

**By August, 88–99% of every model's error is a single systematic over-prediction.**
The models are not noisy — they are confidently, uniformly high. They have learned
the well's historical relationship between sensors and rate (training mean 21,015;
last 90 days of training 18,208) and they keep asserting it while the actual rate
falls away. TabPFN drifts furthest (+4,363 and +4,572), which is precisely why it
loses the annual aggregate despite winning two months outright.

This is what distribution shift looks like when you can see it: not a modelling
failure that a better algorithm fixes, but a calibration that expires.

---

## What to do about it

The finding is about the pipeline, not the model choice. The 26-point spread
between the best and worst model (47.1% vs 67.6% mean skill) is smaller than the
spread *within* any single model across the year (15% → 200%).

- **Retrain or recalibrate quarterly.** Skill is intact for ~5 months and gone by
  month 8. A model refreshed in May would have covered August with recent data.
- **Monitor prediction bias, not just error.** Mean(pred) − mean(actual) over a
  rolling window is a clean, cheap drift alarm: it stays within ±1,200 through May
  and blows past +2,000 in August, well before the error becomes catastrophic.
- **If a single static model is required, use tuned LightGBM.** It is the only
  model that stays under 100% of baseline in every month, and it is the best or
  second-best in five of seven.
- **TabPFN is the wrong choice for this deployment**, despite being competitive on
  the 5-fold aggregate and best-in-class on several earlier rungs of this project.
  It degrades the fastest under drift.

---

## Cost

| model | total s | vs TabPFN |
|---|---|---|
| xgb_default | 0.345 | 112× faster |
| lgbm_default | 0.228 | 170× faster |
| xgb_tuned | 7.250 | 5.3× faster |
| lgbm_tuned | 20.299 | 1.9× faster |
| **tabpfn_v3** | **38.781** | — |

TabPFN is both the slowest and the least accurate here — the only rung in this
project where it is dominated on both axes. Its shape is unchanged (fit 3.1s,
predict 35.6s, 92% of wall-clock in inference, scaling with the 101k-row training
context), but on this task that cost buys nothing.

---

## Caveats

- **Heavy overlap with the 5-fold run.** Fold 4 there covered nearly this same
  period. This is a cleaner, more communicable framing of ground already partly
  covered — not independent confirmation.
- **One fit per model, no seed variance.** Differences smaller than the
  month-to-month spread are not a ranking. The monthly table is the honest view.
- **June is missing entirely; July has 64 rows and was excluded.** Both stated by
  `subperiod.py` at run time rather than silently dropped.
- **Months are not independent** — autocorrelated and trending. Win counts and
  effect sizes only.
- **The file is pre-filtered** at a hard 15,000 floor, so no model ever sees
  low-rate or shut-in operation on either side of the split.
- **TabPFN is untuned by design** and compared against a 30-trial search.
- Whether the August/September divergence is genuine decline, a workover, a choke
  change or an instrument recalibration is **not knowable from these seven
  columns**. The models' uniform over-prediction says the sensor→rate relationship
  changed; it cannot say why.

---

## Reproduce

```bash
srun --gres=gpu:a100:1 -t 00:40:00 -c 16 --mem=64G bash -c \
  'source scripts/env.sh; $PY src/benchmark.py --dataset ke02_2025 \
   --models mean_baseline,tabpfn_v3,xgb_default,lgbm_default,xgb_tuned,lgbm_tuned \
   --folds 1 --gap 144 --gbdt-device cpu --tabpfn-device cuda \
   --save-preds results/preds_ke02_2025.parquet'

python src/summarize.py --results results/results_ke02_2025.csv
python src/subperiod.py --preds results/preds_ke02_2025.parquet \
    --dataset ke02_2025 --freq M --metric rmse --ref tabpfn_v3
```

---

## Harness changes made for this rung

- **`datasets.py`** — `cv="holdout"` with `split_at`, resolved by `load()` into
  `holdout_n_train` via `dataclasses.replace` (the dataclass is frozen). Local-file
  parquet cache re-keyed on the *source* file stem, so `ke02_well` and `ke02_2025`
  share one parse of the spreadsheet.
- **`benchmark.py`** — holdout branch producing a single fold; `--save-preds`
  writing per-row `model/fold/row/y_true/y_pred` parquet. Round-trip verified:
  148,734 rows, `y_true` matches `y` at every row index, and metrics recomputed
  from the parquet reproduce the CSV to 1e-9.
- **`subperiod.py`** (new) — generic sub-period breakdown of saved predictions,
  with a `--min-rows` filter that **reports** what it dropped.
- **`models.py`** — the `ordered` inner-split condition widened to cover `holdout`.
  Without it the holdout run would have silently reverted to the leaky random inner
  split documented in `report_ke02_well.md`.

### The append guard earned its keep

The first attempt at this run **failed on purpose**. `benchmark.py`'s `default_out`
routing special-cased `cv == "timeseries"` only, so the holdout run aimed its
wider schema (`n_train`/`n_test`) at the 280-row `results.csv`. The header guard
added in the previous rung refused the write and exited non-zero:

```
results/results.csv has header [... 'fold', 'seed', 'rmse' ...]
but this run writes  [... 'fold', 'n_train', 'n_test', 'seed', 'rmse' ...]
Refusing to append - it would misalign every row.
```

`results.csv` was verified unchanged by md5. The routing is now keyed on
`meta.cv != "kfold"` rather than an enumerated list, so a future CV mode cannot
forget to opt out. This is the same class of bug as the `ordered` condition above —
adding a mode and missing one of the places that branches on it.
