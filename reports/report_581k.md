# Report — 582k rung: TabPFN-3 vs XGBoost vs LightGBM

**Date:** 2026-07-27 · **Results:** `results/results.csv`

**Runs (split into two jobs):**

| job | node | hardware | models | elapsed |
|---|---|---|---|---|
| 49492477 | `gpu202-09-l` | A100-SXM4-80GB + EPYC 7713P (Milan), 8 cores | xgb_default, lgbm_default, xgb_default_gpu, tabpfn_v3 | 2h07m39s |
| 49492478 | `cn603-09-r` | **no GPU** + Xeon Gold 6148 (Skylake), 8 cores | xgb_tuned, lgbm_tuned | 1h04m06s |

---

## Dataset

| | |
|---|---|
| Name | `nyc_taxi` (NYC green taxi, December 2016) |
| Source | OpenML-CTR23 suite, dataset ID **44986** |
| Shape | **581,835 rows × 17 features** (19 columns raw) |
| Target | `tip_amount` (US dollars) |
| Target range | −10.56 … 250.70, mean 2.375, **median 1.96**, 15.2% exact zeros |
| Missing values | none |
| Categorical features | **9 raw** — first rung in the ladder that needs encoding |
| Task | Regression |

### Target leakage — measured, not assumed

`total_amount` is an accounting identity:

```
total_amount = fare + extra + mta_tax + tolls + improvement_surcharge + tip_amount
```

so it contains the target additively. Verified directly against the data: the
implied `fare_amount` (total minus every other component) falls in a plausible
$2.50–100 range for **99.8%** of rows, confirming the identity holds.

A control experiment quantifies what leaving it in would have bought — same
model, same folds, one column difference:

| `X` contains | XGBoost default R² (3-fold) |
|---|---|
| **clean** — `total_amount` dropped | **0.2523** |
| leaky — `total_amount` kept | **0.5052** |

**One column doubles R².** All results below use the clean matrix.

> **Consequence worth stating plainly.** This parquet has **no `fare_amount`
> column**. Tips track fare closely, so dropping `total_amount` removes *all*
> fare information with no legitimate proxy left. The honest R² here is ~0.29,
> not because the models are weak but because the informative signal was
> entangled with the target. Any published result near 0.5 on this dataset is
> very likely measuring the leak.

### Categorical encoding — a forced choice

Nine columns arrive as pandas `category`. One-hot encoding is **not available**:
`PULocationID` (233 levels) + `DOLocationID` (259) would produce ~500 features
and blow past TabPFN's ceiling of 200 features at this row count.

`datasets.py:_encode()` therefore separates two cases and logs each decision:

| handling | columns |
|---|---|
| numeric stored as category → parsed back to float | `VendorID`, `RatecodeID`, `PULocationID`, `DOLocationID`, `extra`, `mta_tax`, `improvement_surcharge`, `trip_type` |
| genuine category → integer codes | `store_and_fwd_flag` (Y/N) |

Only one column needed arbitrary codes. The location IDs were numeric strings,
so they keep their natural ID values — no invented ordering beyond what the IDs
already carried. **No model gets native categorical support**; all six receive
the identical matrix, which keeps the comparison like-for-like.

---

## Setup

| | |
|---|---|
| Validation | 10-fold CV, `KFold(shuffle=True, random_state=42)` — 523,651 train rows/fold |
| GPU | A100-SXM4-80GB for `tabpfn_v3` and `xgb_default_gpu` |
| CPU | 8 pinned cores |
| Tuning | GBDT: 30-trial random search + early stopping (50 rounds), inner 80/20 split from train folds only. TabPFN: library defaults, untuned |
| Timing | fit / predict / total per fold, CUDA-synchronised; warmup excluded. Tuned `fit_time` **includes** the full 30-trial search |
| Software | tabpfn 8.1.0, xgboost 3.2.0, lightgbm 4.7.0, torch 2.11.0+cu128, Python 3.11 |

RMSE and MAE are in dollars.

---

## Results

| model | device | R² | RMSE | MAE | fit s | pred s | total s |
|---|---|---|---|---|---|---|---|
| **tabpfn_v3** | cuda | **0.2954 ± 0.0292** | **2.277 ± 0.171** | **1.037** | 4.336 | **757.951** | 762.286 ± 0.529 |
| lgbm_tuned | cpu | 0.2928 ± 0.0254 | 2.281 ± 0.164 | 1.065 | 181.329 | 1.994 | 183.323 ± 16.849 |
| xgb_tuned | cpu | 0.2851 ± 0.0268 | 2.293 ± 0.166 | 1.067 | 200.035 | 0.159 | 200.194 ± 25.209 |
| lgbm_default | cpu | 0.2613 ± 0.0208 | 2.331 ± 0.158 | 1.123 | 0.956 | 0.035 | 0.991 ± 0.539 |
| xgb_default | cpu | 0.2607 ± 0.0244 | 2.331 ± 0.157 | 1.080 | 0.878 | 0.020 | 0.898 ± 0.292 |
| xgb_default_gpu | cuda | 0.2607 ± 0.0244 | 2.331 ± 0.157 | 1.080 | 0.289 | 0.007 | **0.296 ± 0.013** |

Sorted by R². Mean ± std across 10 folds.

### Paired per-fold comparison vs TabPFN-3

Positive = TabPFN is better.

| vs tabpfn_v3 | Δ R² | sd of Δ | folds won | p (Wilcoxon) |
|---|---|---|---|---|
| lgbm_tuned | +0.0026 | 0.0066 | 6/10 | **0.3750 — not significant** |
| xgb_tuned | +0.0103 | 0.0085 | 9/10 | 0.0059 |
| lgbm_default | +0.0341 | 0.0100 | 10/10 | 0.0020 |
| xgb_default | +0.0346 | 0.0193 | 10/10 | 0.0020 |
| xgb_default_gpu | +0.0346 | 0.0193 | 10/10 | 0.0020 |

**TabPFN-3 and tuned LightGBM are statistically tied** (p = 0.375, sign flips
across folds: −0.005, +0.003, +0.010, −0.001, +0.009, −0.006, +0.014, +0.004,
+0.001, −0.003). TabPFN beats tuned XGBoost (9/10, p = 0.0059) and both defaults
(10/10, p = 0.0020).

This is the one rung where the raw ranking and the honest verdict disagree:
TabPFN has the highest mean R², but the gap to lgbm_tuned is a **third** of the
per-fold spread of the difference. Reading the leaderboard alone would have
called a win that isn't there.

### MAE tells a different story — and it *is* significant

| vs tabpfn_v3 | Δ MAE | folds won | p |
|---|---|---|---|
| lgbm_tuned | **+0.0288** | **10/10** | **0.0020** |
| xgb_tuned | +0.0307 | 10/10 | 0.0020 |
| xgb_default | +0.0434 | 10/10 | 0.0020 |
| lgbm_default | +0.0866 | 10/10 | 0.0020 |

**TabPFN beats every model on MAE in all 10 folds**, including the LightGBM it
ties on R². Same pattern as the 241k rung: TabPFN is better on the typical case
than squared-error metrics reveal.

But unlike sgemm, the error tails here are **not** pathological:

| model | RMSE/MAE | (sgemm, for contrast) |
|---|---|---|
| tabpfn_v3 | 2.196 | 4.44 |
| xgb_default | 2.159 | 1.78 |
| xgb_tuned | 2.149 | 1.80 |
| lgbm_tuned | 2.141 | 1.76 |
| lgbm_default | 2.075 | 1.49 |

All six models sit at 2.07–2.20. **TabPFN's heavy tail at sgemm was a property
of that dataset, not of TabPFN** — an important correction to the reading in
`report_241k.md`, which could not distinguish the two on one dataset. Here
TabPFN's tail is normal and it still wins MAE, so the MAE advantage is real and
separable from tail behaviour.

---

## Timing — TabPFN's inference cost is quadratic

| rung | train rows/fold | fit s | predict s | step scaling |
|---|---|---|---|---|
| concrete | 927 | 0.498 | 0.491 | — |
| white_wine | 4,408 | 0.425 | 0.764 | 4.75× data → 1.56× |
| california_housing | 18,576 | 0.531 | 2.197 | 4.21× data → 2.87× |
| sgemm | 217,440 | 1.769 | 134.405 | 11.7× data → 61.2× |
| **nyc_taxi** | **523,651** | 4.336 | **757.951** | **2.41× data → 5.64×** |

The last step gives an exponent of `ln(5.64) / ln(2.41)` = **1.97 — essentially
quadratic**, exactly what attention over the training context predicts. The
earlier n^1.6 estimate came from a shorter lever arm and understated it; this
fold was under-forecast by 40%.

`fit` remains near-flat (0.43 → 4.34s across a **565× range** of training-set
size) because it only preprocesses. Predict grew **1,544×** over the same range.

Cost to reach parity with tuned LightGBM at this rung:

| model | predict s | hardware |
|---|---|---|
| xgb_tuned | 0.159 | 8 CPU cores |
| lgbm_tuned | 1.994 | 8 CPU cores |
| **tabpfn_v3** | **757.951** | **A100-80GB** |

TabPFN needs **380× lgbm_tuned's inference time on a far more expensive device**
to achieve a statistically indistinguishable R². Unlike the tuned models' 3-minute
search, this cost is paid on **every** batch and never amortises.

### Projection to 1M rows

At n^1.97, a 1,000,000-row dataset (900k train rows/fold) implies roughly
**~2,200s ≈ 37 min per fold**, i.e. **~6 hours for 10 folds** on one A100 — and
that is at TabPFN's documented 1M-row ceiling. **The 1M rung is not viable at 10
folds on a single GPU.** Options: reduce to 3 folds (~2h), subsample, or accept a
multi-day queue.

---

## Cross-rung summary

| rung | n | winner (R²) | TabPFN vs best tuned GBDT | TabPFN predict s |
|---|---|---|---|---|
| concrete | 1,030 | tabpfn | +0.0064 (8/10, p=0.037) | 0.49 |
| white_wine | 4,898 | tabpfn | +0.0349 (10/10, p=0.002) | 0.76 |
| california_housing | 20,640 | tabpfn | +0.0336 (10/10, p=0.002) | 2.20 |
| sgemm | 241,600 | **xgb_tuned** | −0.0008 (0/10, p=0.002) | 134.41 |
| nyc_taxi | 581,835 | tabpfn (tie) | +0.0026 (6/10, **p=0.375**) | 757.95 |

TabPFN's accuracy advantage is **not** a monotone function of n. It was decisive
at 5k–20k, lost at 241k, and is a statistical tie at 582k. Each rung changed
dataset structure as well as size, so this table cannot separate the two — but it
does refute a simple "TabPFN degrades with scale" story on accuracy. **The cost
side, by contrast, is perfectly monotone and quadratic.**

---

## Findings

1. **TabPFN-3 ties tuned LightGBM** (Δ R² = +0.0026, p = 0.375) and beats tuned
   XGBoost (9/10, p = 0.0059). Highest mean R², but the top gap is not real.
2. **TabPFN wins MAE against everything, 10/10, p = 0.0020** — including the
   model it ties on R². Better typical-case accuracy is its genuine edge here.
3. **Predict cost is quadratic in training-set size (n^1.97)** — 758s/fold on an
   A100 versus 2.0s for tuned LightGBM on 8 CPU cores, a 380× gap.
4. **Leakage is the dominant effect on this dataset**: keeping `total_amount`
   doubles R² (0.25 → 0.51). The benchmark as distributed is hazardous.
5. **Tuning bought little** (+0.024 R² for xgb, +0.032 for lgbm) because the
   target is genuinely noisy — the opposite of sgemm's near-deterministic target
   where the same search bought a 5.5× error reduction.
6. **XGBoost GPU wins decisively at last**: 0.296s total vs 0.898s on CPU (3.0×),
   bit-identical accuracy. It lost at 1k/5k/20k, edged ahead at 241k, and is
   clearly ahead at 582k.

### Caveats

- **Tuned and default GBDT ran on different nodes**, and this time the tuned node
  was **slower** (Xeon Gold 6148, Skylake 2017 vs EPYC 7713P, Milan 2021). So the
  tuned fit times are **inflated** relative to the defaults — the reverse of the
  241k rung, where the tuned node was faster. Cost-of-tuning ratios are not
  comparable across these two rungs. Accuracy is unaffected.
- **`xgb_default_gpu`'s predict is not a true GPU path.** XGBoost warns
  `Falling back to prediction using DMatrix due to mismatched devices` — the model
  is on `cuda:0` while `X_te` is a CPU array. Its **fit** timing is clean; its
  predict timing measures a fallback. Conclusions are unaffected (7 ms either way).
- **Three variables changed at once** vs the 241k rung: size, structure (discrete
  grid → mixed categorical/temporal), and noise level. Do not read this rung as an
  isolated size effect.
- **Encoding is a confound.** Integer codes were forced by TabPFN's 200-feature
  ceiling. LightGBM's native categorical support would likely help it; none of the
  models got it.
- Folds are generated locally (`KFold`, seed 42) rather than from OpenML's
  predefined task splits, because OpenML's metadata API was down (504).
- `best_rounds` / `val_rmse` still unrecorded — `benchmark.py` writes
  `notes = args.tag or str(info)`, so `--tag` discards the tuned diagnostics.

---

## Reproduce

```bash
# fast models + TabPFN, needs the A100 (~2h)
srun --gres=gpu:a100:1 -t 04:00:00 -c 8 --mem=180G bash -c \
  'source scripts/env.sh; $PY src/benchmark.py --dataset nyc_taxi \
   --models xgb_default,lgbm_default,xgb_default_gpu,tabpfn_v3 \
   --gbdt-device cpu --tabpfn-device cuda --tag a100-reference'

# tuned GBDT, no GPU needed (~1h)
srun -t 06:00:00 -c 8 --mem=180G bash -c \
  'source scripts/env.sh; $PY src/benchmark.py --dataset nyc_taxi \
   --models xgb_tuned,lgbm_tuned --gbdt-device cpu --tabpfn-device cpu \
   --tag cpu-only-node-tuned'

python src/summarize.py --dataset nyc_taxi
for m in r2 rmse mae; do python src/compare.py --dataset nyc_taxi --ref tabpfn_v3 --metric $m; done
```
