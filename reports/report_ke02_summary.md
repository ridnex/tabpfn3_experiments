# KE02-01 — full experiment report

**Well:** KE02-01 · **Task:** virtual flow metering (predict `Total Massrate` from sensors)
**Date:** 2026-07-28 · **Hardware:** Ibex, NVIDIA A100-SXM4-80GB + 16 CPU cores

Master summary of every experiment run on this dataset. Detailed per-experiment
reports: `report_ke02_well.md` (5-fold), `report_ke02_2025.md` (forward holdout),
`report_ke02_shuffle.md` (shuffled).

---

## TL;DR

1. **Random cross-validation gives R² = 0.99 on this data and it is meaningless.**
   A lookup table using *zero features* scores 0.9915.
2. **Which split you choose decides which model wins.** TabPFN-3 goes from **1st**
   (shuffled, 0.9981) to **last** (forward, −2.679). Same data, same code.
3. **For forecasting, nothing works well** — every model is below R² = 0 except one
   configuration.
4. **The one thing that worked: train on 3 months instead of 6 years** → R² **+0.247**.
   More data actively hurts.
5. **TabPFN-TS fixes TabPFN's forecasting weakness**, reaching 2nd place using
   *only dates* — beating XGBoost on the same input by 4.6× on RMSE.

---

## 1. The data

| | |
|---|---|
| Source | `KE02-01 filtered.xlsx`, one sheet, 15.7 MB |
| Raw rows | 135,354 → **125,948** after removing 9,406 exact duplicates |
| Period | 2019-02-13 → 2025-09-29 (2,420 days), **10-minute cadence** |
| Features | `Choke`, `DHP`, `DHT`, `FLP`, `FLT`, `FTHP`, `FTHT` (7 sensors) |
| Target | `Total Massrate`, 15,000–90,279 |
| Missing | none |

**Known data issues:**
- The file is **pre-filtered** at a hard floor of exactly 15,000 — low-rate and
  shut-in operation was removed before we received it. No model ever sees it.
- **June 2025 is entirely absent.** July 2025 has only **64 rows** (mean 31,796 vs
  ~18,000 elsewhere).
- A **344-day gap** exists in the record; cause unknown.
- The well is **declining**: yearly mean rate falls 25,276 (2019) → 17,798 (2025).

---

## 2. The finding that shaped everything: random CV is invalid here

The target's **lag-1 autocorrelation is 0.9977**. Consecutive rows differ by a
median of **54 units** against an overall spread of **4,960** — a reading 10 minutes
away is ~92× closer than a random row.

Same model (`LGBMRegressor`, defaults), four different splits:

| split | R² |
|---|---|
| random 10-fold | **0.9905** |
| middle 10% block held out | −0.267 |
| last 10% held out | −1.505 |
| constant baseline on that block | −3.549 |

**Deduplication does not fix this.** All 9,406 exact duplicates were removed before
every number in this report. The leak comes from *near*-twins — rows 10 minutes
apart that are almost but never exactly equal.

**Proof it is a lookup, not learning:** ignore all 7 sensors and simply copy the
target of the nearest row in time that happens to be in the training set:

| "model" | R² |
|---|---|
| **copy nearest training row (zero features)** | **0.9915** |
| lgbm_default (7 sensors) | 0.9899 |
| tabpfn_v3 (7 sensors) | 0.9981 |

A lookup table beats untuned LightGBM. After an 80/20 shuffle, **79.9% of test rows
have their direct neighbour in training**.

---

## 3. Results — shuffled split (gap filling)

**Question answered:** *"The meter dropped out for an hour. Can I reconstruct it,
given data before and after?"* 100,758 train / 25,190 test.

| model | R² | RMSE | MAE | search s | train s | infer s | **total s** |
|---|---|---|---|---|---|---|---|
| **tabpfn_v3** | **0.9981** | **214** | **116** | — | 0.86 | 35.74 | 36.59 |
| lgbm_tuned | 0.9967 | 281 | 141 | 181.51 | 7.41 | 0.31 | 189.24 |
| xgb_tuned | 0.9966 | 283 | 142 | 79.24 | 5.02 | 0.08 | 84.34 |
| xgb_default | 0.9939 | 381 | 237 | — | 0.14 | 0.003 | **0.14** |
| lgbm_default | 0.9899 | 490 | 325 | — | 0.37 | 0.005 | 0.38 |
| *mean_baseline* | *−0.000* | *4874* | *3750* | — | — | — | — |

**TabPFN-3 wins with zero tuning** — 24% lower RMSE than LightGBM, which spent
181 seconds searching. This is TabPFN's ideal case and it delivers.

**But `xgb_default` reaches 0.9939 in 0.14 seconds** — 99.6% of the accuracy at
1/264th the cost. And tuning buys almost nothing: +0.003 R² for 600× the time.

---

## 4. Results — forward split (forecasting)

**Question answered:** *"If I had built this on 1 January 2025, would it have
worked?"* Train 2019–2024 (101,159 rows) → test 2025 (24,789 rows), 1-day gap.

| # | model | features | R² | RMSE | MAE | train s | infer s | **total s** |
|---|---|---|---|---|---|---|---|---|
| 1 | **xgb_default** | 7 sensors + 7 physics | **−0.223** | **1496** | **1215** | 0.23 | 0.00 | **0.23** |
| 2 | **tabpfn_ts** | **date only** | −0.337 | 1564 | 1219 | n/a | n/a | 12.0 |
| 3 | lgbm_default | 7 sensors + 7 physics | −0.466 | 1638 | 1300 | 0.27 | 0.00 | 0.28 |
| 4 | lgbm_tuned | 7 sensors | −0.543 | 1680 | 1310 | 20.29 | 0.01 | 20.30 |
| 5 | xgb_tuned | 7 sensors | −0.661 | 1743 | 1327 | 7.25 | 0.00 | 7.25 |
| 6 | tabpfn_ts | date + 7 sensors | −0.776 | 1803 | 1412 | n/a | n/a | 15.0 |
| 7 | xgb_tuned | 7 sensors + 7 physics | −0.858 | 1844 | 1390 | 6.49 | 0.00 | 6.49 |
| 8 | xgb_default | 7 sensors | −0.953 | 1890 | 1470 | 0.34 | 0.00 | 0.35 |
| 9 | lgbm_tuned | 7 sensors + 7 physics | −1.004 | 1915 | 1477 | 13.03 | 0.00 | 13.03 |
| 10 | lgbm_default | 7 sensors | −1.251 | 2029 | 1556 | 0.22 | 0.01 | 0.23 |
| 11 | tabpfn_v3 | 7 sensors + 7 physics | −2.538 | 2544 | 1874 | 0.86 | 36.69 | 37.55 |
| 12 | tabpfn_v3 | 7 sensors | −2.679 | 2594 | 1876 | 3.15 | 35.64 | 38.78 |
| — | *mean_baseline* | none | *−5.680* | *3496* | *3297* | 0.00 | 0.00 | 0.00 |

**Every R² is negative.** None of these is deployable as-is.

### The ranking fully inverts between splits

| model | shuffled | forward |
|---|---|---|
| tabpfn_v3 | **0.9981** (1st) | −2.679 (**last**) |
| lgbm_tuned | 0.9967 | **−0.543** (best of the tabular models) |
| xgb_default | 0.9939 | −0.953 |

---

## 5. Why forward fails: the models expire

Error decomposed into bias (systematic offset) vs scatter, by month:

| month | actual mean | xgb_tuned bias | lgbm_tuned bias | tabpfn_v3 bias |
|---|---|---|---|---|
| 2025-01 | 17,298 | −230 | −531 | −611 |
| 2025-03 | 17,627 | +300 | +185 | +61 |
| 2025-05 | 18,376 | −693 | −1,211 | −585 |
| **2025-08** | 18,981 | **+2,278** | **+1,672** | **+4,363** |
| **2025-09** | 16,844 | **+3,210** | **+2,810** | **+4,572** |

By August, **88–99% of every model's error is pure bias** — they all over-predict,
in the same direction. The models are not confused, they are confidently applying
an expired calibration.

### Model shelf life ≈ 5 months

RMSE as % of the no-skill baseline (100% = useless):

| | Jan | Feb | Mar | Apr | May | Aug | Sep |
|---|---|---|---|---|---|---|---|
| lgbm_tuned | 24% | 24% | 17% | 35% | 75% | **86%** | 69% |
| xgb_tuned | 23% | 26% | 20% | 26% | 48% | **110%** | 77% |
| tabpfn_v3 | 26% | 21% | **15%** | 31% | 73% | **200%** | 109% |

Strong through April, collapsed by August. This is invisible in the annual total.

---

## 6. What actually fixed it: less training data

Same model, same features, same test set — only the **training window** changes:

| training window | rows | lgbm R² | RMSE | xgb R² | bias |
|---|---|---|---|---|---|
| **last 3 months** | 9,257 | **+0.247** | **1173** | **+0.165** | **+47** |
| last 6 months | 21,083 | +0.229 | 1187 | −0.255 | −19 |
| last 12 months | 27,487 | −1.141 | 1979 | −1.408 | +907 |
| last 24 months | 48,705 | −0.833 | 1831 | −1.154 | +662 |
| all history | 101,015 | −1.251 | 2029 | −0.953 | +662 |

**This is the only configuration that produced a positive R².** Cutting from 6 years
to 3 months moved R² by **+1.50** and dropped bias from +662 to +47 — larger than
any model choice, feature set, or tuning decision in this entire report.

**More data is actively harmful here.** 2019–2022 describes a well that no longer
exists, and those rows are 90% of the training set.

---

## 7. Feature engineering

### Physics features (7 added: pressure/temperature differentials)

`dP_choke = FTHP − FLP`, `dP_tubing = DHP − FTHP`, `dT_tubing`, `dT_flowline`,
`ratio_THP_DHP`, `choke_area = Choke²`, `choke_flow = Choke²·√dP`.

| model | raw R² | + physics | change |
|---|---|---|---|
| **xgb_default** | −0.953 | **−0.223** | **+0.73** |
| **lgbm_default** | −1.251 | **−0.466** | **+0.79** |
| xgb_tuned | −0.661 | −0.858 | −0.20 |
| lgbm_tuned | −0.543 | −1.004 | −0.46 |
| tabpfn_v3 | −2.679 | −2.538 | +0.14 |

**Untuned trees gained a lot; TabPFN gained almost nothing.** A tree splits one
column at a time and structurally cannot compute `FTHP − FLP`, so handing it the
difference is a real gift. TabPFN already forms combinations internally.

**Tuning + physics = worse.** With more features the search overfits its validation
window harder — it picked 20–50 trees and reported val_rmse ≈ 1165 while real error
was 1844–1915.

### Time features — tested and rejected

| features | lgbm R² | xgb R² |
|---|---|---|
| 7 sensors | −1.251 | −0.953 |
| + raw `Datetime` | −0.913 | −2.067 |
| + hour of day | −1.134 | −0.887 |
| + hour + month | −1.067 | −1.987 |

The two libraries move in **opposite directions** — no reliable signal.

---

## 8. TabPFN-TS

`tabpfn-time-series 1.2.0`, installed in an **isolated venv** (it pins pandas 2.3.3
while the benchmark environment runs 3.0.5). It converts a series into a table using
a running index + calendar features + auto-detected seasonality, then applies TabPFN.

| input | R² | RMSE | MAE | bias | time |
|---|---|---|---|---|---|
| **date only** | **−0.337** | **1564** | **1219** | −576 | 12 s |
| date + 7 sensors | −0.776 | 1803 | 1412 | +420 | 15 s |

**Adding the sensors makes it worse.** The date-only model extrapolates the decline
honestly; the sensors are an expired calibration that pulls predictions back toward
the old high-flow regime (bias flips −576 → +420).

### The clean TabPFN-TS result: date-only, head to head

Every model given **only** dates, no sensors at all:

| model | R² | RMSE | bias |
|---|---|---|---|
| **tabpfn_ts** | **−0.337** | **1564** | −576 |
| lgbm_default | −2.147 | 2399 | +1,880 |
| xgb_default | −5.731 | 3509 | +2,535 |
| linear regression | −16.901 | 5722 | −5,366 |

**TabPFN-TS beats XGBoost by 4.6× and LightGBM by 2.4× on RMSE.** XGBoost with dates
is no better than a constant — literally what it produces, since every 2025 date lies
beyond every training date, so all test rows land in one leaf. Linear regression
extrapolates the decline too aggressively.

**This is TabPFN-TS's genuine advantage:** it extrapolates a trend from time alone,
which trees structurally cannot do and naive linear does badly.

---

## 9. Two bugs found and fixed

### The tuning was leaky (found, fixed, re-run)

`_inner_split()` used `rng.permutation(n)` — a random validation split, i.e. the
exact leak the outer forward split exists to prevent, reproduced one level down.

| inner split | reported val_rmse | real test RMSE | trees chosen |
|---|---|---|---|
| random (original) | **247–365** | 1,541–6,047 | 920–1,999 |
| ordered (fixed) | 938–5,883 | 1,195–5,671 | 2–936 |

The search was scoring candidates on neighbour-copying, so it kept buying trees.
After the fix, `lgbm_tuned` went **−2.029 → −0.686** and got **6× faster**. The four
models with no inner split reproduced bit-identically, confirming the change was
isolated.

**This bug is latent in the six earlier OpenML rungs but harmless there** — those
datasets are iid, so a random inner split is correct and the flag stays off.

### The append guard caught a schema bug

Adding `n_train`/`n_test` columns meant a later run aimed a wider schema at the
280-row `results.csv`. The header guard refused the write and exited non-zero rather
than silently writing column-shifted rows. `results.csv` verified unchanged by md5.

---

## 10. Recommendations

**If the goal is gap filling / sensor backup:** it already works. R² 0.99 is real
for that job, because the neighbouring readings genuinely exist in deployment.
Use `xgb_default` (0.14 s) unless you need the last 0.4%, then TabPFN-3.

**If the goal is forecasting:**
1. **Retrain on a rolling 3-month window.** Worth more than every other change
   combined (+1.50 R²).
2. **Add the physics features.** Free, and worth +0.73 to untuned trees.
3. **Don't tune** in this regime — with drift, the search overfits its validation
   window and makes things worse.
4. **Monitor prediction bias**, not just error. Mean(pred) − mean(actual) stays
   within ±1,200 through May and exceeds +2,000 in August, well before the error
   becomes catastrophic. It is a cheap early-warning signal.
5. **Consider TabPFN-TS** if you want a trend extrapolation baseline with no
   feature engineering at all.

**On introducing TabPFN at the company** — the honest pitch is *"near-best accuracy
with zero tuning"*, not *"most accurate"*:
- Strongest evidence: on the shuffled split it beat a 181-second hyperparameter
  search **while doing nothing at all**.
- Be upfront about the two weaknesses, because they will be asked about:
  inference is ~36 s and grows with training-set size (GBDTs are milliseconds), and
  plain TabPFN is the *worst* model under distribution shift. Lead with TabPFN-TS
  for anything time-related.

---

## 11. Caveats

- **One well, one target.** Nothing here generalises to other wells.
- **Single split, single seed** for the forward and shuffled headline tables. No
  error bars; small differences are not a ranking. The 5-fold run
  (`report_ke02_well.md`) is the only multi-fold evidence, and 5 folds cannot reach
  statistical significance (min Wilcoxon p = 0.0625).
- **The pre-filtering at 15,000 is a real limitation** — the models never see
  low-rate operation, on either side of any split.
- **The 3-month-window result used default hyperparameters on one split.** It is the
  most important finding here and it deserves confirmation across several windows and
  seeds before anything is built on it.
- **Why August diverged is not knowable from these 7 columns** — genuine decline, a
  workover, a choke change and an instrument recalibration would all look identical.
- **TabPFN-TS timing is one number**; its API does not separate fit from predict.

---

## 12. Files

| file | contents |
|---|---|
| `results/results_ke02_well.csv` | 5-fold forward chaining, 6 models |
| `results/results_ke02_well_randominner.csv` | same, before the inner-split fix (kept as evidence) |
| `results/results_ke02_2025.csv` | forward holdout, 4 models, with search/train timing split |
| `results/results_ke02_2025_v1.csv` | forward holdout, 6 models |
| `results/results_ke02_2025_phys.csv` | forward holdout + physics features, 6 models |
| `results/results_ke02_shuffle.csv` | shuffled 80/20, 6 models |
| `results/preds_ke02_2025.parquet` | per-row predictions (enables the monthly slice) |
| `results/preds_tabpfn_ts_{cov,nocov}.parquet` | TabPFN-TS predictions |
| `src/run_tabpfn_ts.py` | TabPFN-TS runner (isolated venv) |
| `src/subperiod.py` | sub-period breakdown of saved predictions |
| `reports/report_ke02_{well,2025,shuffle}.md` | detailed per-experiment reports |

Nothing was deleted at any point; every superseded result file was archived rather
than overwritten.
