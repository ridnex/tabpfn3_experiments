# Report — KE02-01 well: TabPFN-3 vs XGBoost vs LightGBM under a forward-in-time split

**Date:** 2026-07-28 · **Runs:** Slurm 49523953 (`gpu201-16-l`) and 49524681 (`gpu203-16-l`), A100 ·
**Results:** `results/results_ke02_well.csv` (primary), `results/results_ke02_well_randominner.csv` (superseded, kept as evidence)

> First **real, user-supplied, non-iid** dataset in this project. All seven previous
> rungs were OpenML tables evaluated with random K-fold. That protocol is invalid
> here, and most of this report is about why.

---

## Dataset

| | |
|---|---|
| Source | `data/KE02-01_filtered.xlsx`, sheet `KE02-01` (user-supplied) |
| Raw shape | 135,354 rows × 9 columns |
| After dedup | **125,948 rows × 7 features** |
| Target | `Total Massrate` — 15,000 to 90,279 (hard floor at 15,000; the file is pre-filtered) |
| Features | `Choke`, `DHP`, `DHT`, `FLP`, `FLT`, `FTHP`, `FTHT` |
| Cadence | 10 minutes, 2019-02-13 → 2025-09-29 (2,420 days) |
| Missing values | **none** |
| Duplicates removed | **9,406** exact duplicates, one per repeated timestamp |

The task is **virtual flow metering**: infer mass flow rate from pressure and
temperature sensors.

`Datetime` is deliberately **not** a feature. It orders the rows and defines the
splits, nothing more — under a forward split every test timestamp lies outside the
training range, so a tree would simply saturate at its last leaf.

---

## Why random K-fold is invalid here (and what it would have told us)

The target's **lag-1 autocorrelation is 0.9977**. Consecutive rows differ by a
median of **54 units** against an overall std of **4,960** — a reading ten minutes
away is ~92× closer than a random row. Under 10-fold random CV, ~90% of test rows
have their own neighbour sitting in the training set.

Measured with `LGBMRegressor` at defaults, one model, four splits:

| split | R² |
|---|---|
| random 10-fold | **0.9905** |
| middle 10% contiguous block held out | −0.267 |
| last 10% in time held out | −1.505 |
| constant train-mean baseline, same block | −3.549 |

**0.99 is the score for copying a neighbour.** Had this dataset been run through
the standard harness it would have produced the best numbers in the entire project
and meant nothing. Every result below uses a forward-in-time split instead.

---

## Setup

| | |
|---|---|
| Validation | `TimeSeriesSplit(n_splits=5, gap=144)` — expanding window, train on past → test on future |
| Gap | 144 rows = **1 day** removed between each train block and its test block, so the last training row is never a 10-minute twin of the first test row |
| GPU | NVIDIA A100-SXM4-80GB (`tabpfn_v3` only) |
| CPU | 16 pinned cores (all GBDT configs and the baseline) |
| Tuning | 30-trial random search, early stopping (50 rounds), inner 20% validation, selected on RMSE. TabPFN: library defaults, untuned |
| Software | tabpfn 8.1.0, xgboost 3.2.0, lightgbm 4.7.0, torch 2.11.0+cu128 |

### Fold geometry

| fold | train rows | train period | test period |
|---|---|---|---|
| 0 | 20,849 | 2019-02 → 2020-10 | 2020-10-23 → 2021-07-20 |
| 1 | 41,840 | 2019-02 → 2021-07 | 2021-07-20 → 2023-08-11 |
| 2 | 62,831 | 2019-02 → 2023-08 | 2023-08-11 → 2024-08-05 |
| 3 | 83,822 | 2019-02 → 2024-08 | 2024-08-05 → 2025-01-28 |
| 4 | 104,813 | 2019-02 → 2025-01 | 2025-01-28 → 2025-09-29 |

Test blocks are 20,991 rows each. Asserted before running: no train/test overlap,
every gap ≥ 144 rows, every test timestamp strictly after its train block.

**The well is declining, and it shows in every fold.** Train-set mean rate exceeds
test-set mean rate in all five (27,160→22,956, 25,077→17,016, 22,381→18,589,
21,425→18,699, 20,883→17,897). Each model is asked to predict a regime lower than
anything it trained on. Trees cannot extrapolate below their training range by
construction; this is a structural ceiling on XGBoost and LightGBM here, not a
tuning failure.

---

## Results

| model | device | R² | RMSE | MAE | fit s | pred s | total s |
|---|---|---|---|---|---|---|---|
| lgbm_default | cpu | **−0.609 ± 1.568** | 2665.1 | 2055.6 | 0.148 | 0.004 | **0.152** |
| lgbm_tuned | cpu | −0.686 ± 1.545 | **2655.5** | **1923.5** | 20.952 | 0.012 | 20.964 |
| tabpfn_v3 | cuda | −0.789 ± 1.674 | 2693.4 | 1980.8 | 0.677 | 18.333 | 19.010 |
| xgb_tuned | cpu | −1.009 ± 2.326 | 2935.2 | 2381.7 | 9.125 | 0.004 | 9.129 |
| xgb_default | cpu | −1.436 ± 2.570 | 3352.2 | 2558.0 | 0.101 | 0.002 | 0.104 |
| *mean_baseline* | cpu | *−5.300 ± 6.935* | *5137.1* | *4634.8* | 0.000 | 0.000 | 0.000 |

**Every R² is negative.** That is the honest answer for this dataset, not a bug.
R² is measured against each *test block's* own variance, and the later blocks are
much tighter than the record as a whole (2025 std 1,352 vs 4,960 overall), so R²
is a harsh scale here. `mean_baseline` — predicting the training fold's average and
ignoring the sensors entirely — is included precisely so these numbers are readable.

### Skill against the baseline

RMSE as a percentage of `mean_baseline` RMSE on the same fold. 100% = no skill.

| fold | xgb_default | lgbm_default | xgb_tuned | lgbm_tuned | tabpfn_v3 |
|---|---|---|---|---|---|
| 0 | 93.6% | 84.8% | 94.0% | 81.7% | **76.5%** |
| 1 | 61.0% | 47.0% | 56.8% | 45.0% | **40.7%** |
| 2 | 55.0% | 27.5% | 28.5% | **24.6%** | 31.6% |
| 3 | 43.8% | **34.7%** | 37.8% | 36.4% | 37.4% |
| 4 | 62.9% | 59.3% | **54.3%** | 70.9% | 85.2% |
| **mean** | 63.2% | **50.6%** | 54.3% | 51.7% | 54.3% |

Read this way the models are clearly doing real work — roughly **halving** the
error of a no-skill predictor — even though every R² is negative. Both facts are
true simultaneously, and the R² column alone would have been misleading.

### Paired per-fold comparison vs TabPFN-3

Positive = TabPFN better. All models ran identical folds.

| vs tabpfn_v3 | Δ R² | won | Δ RMSE | won | Δ MAE | won |
|---|---|---|---|---|---|---|
| lgbm_default | −0.179 | 2/5 | −28.3 | 2/5 | +74.9 | 2/5 |
| lgbm_tuned | −0.103 | 2/5 | −37.9 | 2/5 | −57.3 | 2/5 |
| xgb_tuned | +0.220 | 3/5 | +241.8 | 3/5 | +400.9 | 2/5 |
| xgb_default | +0.648 | 4/5 | +658.8 | 4/5 | +577.2 | 4/5 |
| *mean_baseline* | +4.511 | 5/5 | +2443.7 | 5/5 | +2654.0 | 5/5 |

**No p-values are quoted, deliberately.** With 5 folds the smallest value the
Wilcoxon signed-rank test can produce is 0.0625, so nothing in this table *can*
reach p<0.05. Reporting "p = 0.31, not significant" would be describing the test's
resolution, not the data. Win counts and raw differences are what 5 folds support.
The six earlier rungs used 10 folds and could reach p = 0.0020; this one cannot.

**Verdict: TabPFN-3, LightGBM (tuned or not) and XGBoost-tuned are not separable
on this dataset.** The three cluster at 50.6–54.3% mean skill and trade folds
2-to-3. Only `xgb_default` is clearly behind, losing 4/5 on every metric.

---

## The finding that cost a rerun: our own tuning was leaky

The first run (49523953) produced `lgbm_tuned` at R² = **−2.029**, far *worse*
than `lgbm_default` at −0.609. That looked like a fact about LightGBM. It was a
fact about our harness.

`models.py::_inner_split()` carved its validation set with `rng.permutation(n)` —
**a random split**, i.e. the exact leak the outer forward split exists to prevent,
reproduced one level down. The evidence is unambiguous:

| inner split | reported `val_rmse` | actual test RMSE | rounds chosen |
|---|---|---|---|
| random (original) | **247 – 365** | 1,541 – 6,047 | 920 – 1,999 |
| ordered (fixed) | **938 – 5,883** | 1,195 – 5,671 | 2 – 936 |

The random inner split's own error estimate was **10–20× optimistic**. The search
was scoring candidates on neighbour-copying, so it kept buying trees — up to 1,999
of them — because more trees always help interpolation. Those settings then
generalised badly forward in time.

The fix holds out the **last 20% in time** of the train fold instead, matching the
shape of the outer split. `_inner_split(n, seed, ordered=True)`, bound by
`build_registry()` whenever `cv="timeseries"`. Effect:

| model | R² before → after | RMSE before → after | fit s before → after |
|---|---|---|---|
| **lgbm_tuned** | −2.029 → **−0.686** | 3521 → **2655** | 129.2 → **21.0** |
| xgb_tuned | −0.936 → −1.009 | 3019 → 2935 | 66.5 → **9.1** |

LightGBM went from worst model in the table to competitive, and got **6× cheaper**.
XGBoost was a wash on accuracy but **7× cheaper**. In both cases the ordered split
now reports validation errors in the same range as reality, which is the point —
the tuned configs are finally being selected on the objective we care about.

The four models with no inner split (`mean_baseline`, `tabpfn_v3`, `xgb_default`,
`lgbm_default`) reproduced **bit-identically** across the two runs, which confirms
the change was isolated to tuning and that the pipeline is deterministic.

**This bug is latent in the six earlier rungs too, but harmless there** — those
datasets are iid, so a random inner split is the correct choice and `ordered`
stays `False` for them. Their results are unaffected and were re-verified.

---

## Cost

| model | total s/fold | vs TabPFN |
|---|---|---|
| xgb_default | 0.104 | 183× faster |
| lgbm_default | 0.152 | 125× faster |
| xgb_tuned | 9.129 | 2.1× faster |
| **tabpfn_v3** | **19.010** | — |
| lgbm_tuned | 20.964 | 1.1× slower |

TabPFN's shape is unchanged from every previous rung: **fit 0.68s, predict 18.3s**
— 96% of its wall-clock is inference, and it scales with *training* set size, 4.4s
at 20,849 train rows rising to 36.8s at 104,813. That fold-4 figure lands within
15% of the 31.8s predicted by the n^1.97 curve measured at the 581k rung, so that
scaling law continues to hold on genuinely new data.

The economics here are unusually bad for TabPFN, though: `lgbm_default` is
**125× cheaper** and statistically indistinguishable from it. On this dataset the
free-lunch argument for TabPFN does not apply — there is no tuning cost to avoid,
because the untuned LightGBM is already as good as anything else in the table.

---

## What this says about the well

Independent of which model wins: **these seven sensors do not support forward
prediction of flow rate across operating regimes.** Every model beats a constant
by roughly 2×, and every model has negative R² against the test block's own
variance. Fold 4 (predicting Feb–Sep 2025 from everything before) is the worst for
the two strongest models — TabPFN at 85.2% of baseline, `lgbm_tuned` at 70.9% —
i.e. approaching no skill on the most recent, most operationally relevant period.

If a virtual flow meter is the goal, the model choice is not the bottleneck. The
candidates would be recalibration on recent data, features encoding well state or
decline, or accepting a shorter valid horizon.

---

## Caveats

- **Single well, single target.** Conclusions are about KE02-01, not about virtual
  flow metering generally.
- **The file is pre-filtered.** The hard floor at exactly 15,000 means low-rate and
  shut-in periods were already removed by whoever prepared it. The models never see
  that regime, and the 15,000 boundary is an artificial edge in the target.
- **Five folds cannot reach significance** (min Wilcoxon p = 0.0625). Win counts
  and effect sizes only.
- **Folds are not equal-difficulty** — each covers a different year under different
  operating conditions, so the ±1.5 cross-fold std is regime variation, not model
  noise. The paired view is the one to read.
- **TabPFN is untuned by design** and compared against a 30-trial search. That is
  the intended question, not a like-for-like tuning budget.
- **A 344-day gap** exists in the record; whether it is a shutdown or missing data
  is unknown to us, and it sits inside fold 1's test block — the fold where every
  model does worst.
- Timing for the two runs came from different A100 nodes (`gpu201-16-l`,
  `gpu203-16-l`); the primary table is entirely from 49524681, so it is internally
  consistent.

---

## Reproduce

```bash
# copy KE02-01_filtered.xlsx into data/ first
srun --gres=gpu:a100:1 -t 01:30:00 -c 16 --mem=64G bash -c \
  'source scripts/env.sh; $PY src/benchmark.py --dataset ke02_well \
   --models mean_baseline,tabpfn_v3,xgb_default,lgbm_default,xgb_tuned,lgbm_tuned \
   --folds 5 --gap 144 --gbdt-device cpu --tabpfn-device cuda'

python src/summarize.py --results results/results_ke02_well.csv
for m in r2 rmse mae; do
  python src/compare.py --results results/results_ke02_well.csv \
    --dataset ke02_well --ref tabpfn_v3 --metric $m
done
```

No `--tag` — `benchmark.py` writes `notes = args.tag or str(info)`, so passing a
tag discards the tuned runners' `best_rounds`/`val_rmse`. Those diagnostics are
what exposed the inner-split leak, and the last three reports had to caveat their
absence.

---

## Harness changes made for this rung

- **`datasets.py`** — `Dataset` gained `local_file`/`sheet` (user-supplied files,
  parsed once and cached as parquet), `time_col` (sorts, and is excluded from X),
  `dedup`, and `cv`. Post-condition everything downstream relies on: after
  `load()`, **row order is time order**.
- **`benchmark.py`** — `TimeSeriesSplit` selected on `meta.cv`; `--gap`;
  `n_train`/`n_test` columns, needed because forward chaining varies the train
  size 5× across folds and TabPFN's predict time tracks it.
- **`benchmark.py::append_rows()`** — now verifies an existing file's header
  matches the schema being written and raises otherwise. Adding `n_train`/`n_test`
  would otherwise have let a re-run of an older dataset write column-shifted rows
  into the 280-row `results.csv` under its old header — unrecoverable, because the
  old and new rows would look equally valid. Verified: `results.csv` md5 unchanged.
- **`models.py`** — `mean_baseline` runner; `_inner_split(..., ordered=)` and the
  `build_registry(cv=...)` binding described above.
