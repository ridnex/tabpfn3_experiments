# Report — 20k rung: TabPFN-3 vs XGBoost vs LightGBM

**Date:** 2026-07-26 · **Run:** Slurm job 49456769 (interactive `srun`, A100) · **Results:** `results/results.csv`

> Filed as **20k**, not 25k: the dataset has **20,640** rows.

---

## Dataset

| | |
|---|---|
| Name | `california_housing` |
| Source | OpenML-CTR23 suite, dataset ID **44977** |
| Shape | **20,640 rows × 9 columns** |
| Features | **8** (all numeric) |
| Target | `medianHouseValue` (US dollars, continuous) |
| Missing values | none |
| Categorical features | none |
| Task | Regression |

**Feature columns:** `longitude`, `latitude`, `housingMedianAge`, `totalRooms`,
`totalBedrooms`, `population`, `households`, `medianIncome`

Unlike the 5k rung, this target is **genuinely continuous**, so R² is not capped
by target granularity and the values are directly interpretable again. Feature
count (8) is identical to the 1k rung, which makes this the cleanest size
comparison in the ladder so far.

---

## Setup

Identical protocol to previous rungs.

| | |
|---|---|
| Validation | 10-fold CV, `KFold(shuffle=True, random_state=42)` |
| GPU | NVIDIA A100-SXM4-80GB (`device="cuda"`) |
| CPU | 8 pinned cores (`device="cpu"`) |
| Tuning | GBDT: 30-trial random search + early stopping (50 rounds), inner 80/20 validation split from train folds only. TabPFN: library defaults, untuned |
| Timing | fit / predict / total per fold, CUDA-synchronised; warmup excluded |
| Software | tabpfn 8.1.0, xgboost 3.2.0, lightgbm 4.7.0, torch 2.11.0+cu128, Python 3.11 |

RMSE and MAE are in dollars.

---

## Results

| model | device | R² | RMSE | MAE | fit s | pred s | total s |
|---|---|---|---|---|---|---|---|
| **tabpfn_v3** | cuda | **0.8853 ± 0.0099** | **39,038 ± 1,861** | **23,308 ± 702** | 0.531 | 2.197 | **2.728 ± 0.086** |
| xgb_tuned | cpu | 0.8518 ± 0.0101 | 44,399 ± 1,820 | 28,915 ± 824 | 26.192 | 0.008 | 26.200 ± 1.720 |
| lgbm_tuned | cpu | 0.8513 ± 0.0099 | 44,473 ± 1,850 | 29,004 ± 827 | 29.619 | 0.035 | 29.654 ± 3.095 |
| xgb_default | cpu | 0.8316 ± 0.0119 | 47,321 ± 2,053 | 31,437 ± 1,119 | 0.084 | 0.001 | 0.085 ± 0.001 |
| xgb_default_gpu | cuda | 0.8316 ± 0.0119 | 47,321 ± 2,053 | 31,437 ± 1,119 | 0.142 | 0.003 | 0.146 ± 0.010 |
| lgbm_default | cpu | 0.8315 ± 0.0127 | 47,339 ± 2,126 | 31,851 ± 1,039 | 0.085 | 0.002 | 0.087 ± 0.001 |

Sorted by R². Mean ± std across 10 folds. `cuda` = GPU (A100), `cpu` = 8 cores.

### Paired per-fold comparison vs TabPFN-3

| vs tabpfn_v3 | mean Δ R² | sd of Δ | folds won | p (Wilcoxon) |
|---|---|---|---|---|
| lgbm_default | +0.0539 | 0.0086 | **10/10** | 0.0020 |
| xgb_default | +0.0537 | 0.0089 | **10/10** | 0.0020 |
| xgb_default_gpu | +0.0537 | 0.0089 | **10/10** | 0.0020 |
| lgbm_tuned | +0.0340 | 0.0050 | **10/10** | 0.0020 |
| xgb_tuned | +0.0336 | 0.0059 | **10/10** | 0.0020 |

TabPFN sweeps every fold against every config for the second rung running.

---

## Cross-rung trends (1k → 5k → 20k)

### Accuracy margin over *tuned* GBDT

| rung | n | vs xgb_tuned | vs lgbm_tuned |
|---|---|---|---|
| concrete | 1,030 | +0.0105 (8/10) | +0.0064 (8/10) |
| white_wine | 4,898 | +0.0379 (10/10) | +0.0349 (10/10) |
| california_housing | 20,640 | +0.0336 (10/10) | +0.0340 (10/10) |

The margin jumped from 1k→5k and then **held flat** from 5k→20k at roughly
+0.034. Combined with the fact that concrete and california_housing have the
same feature count and both have continuous targets, the most defensible reading
is that the 1k→5k jump was driven largely by **dataset difficulty**, not size —
and that TabPFN's advantage over tuned GBDT is stable at ~0.034 R² so far, not
growing with n.

### Timing — TabPFN's predict cost is now clearly scaling

TabPFN, per fold:

| rung | train rows/fold | fit s | predict s | predict vs previous |
|---|---|---|---|---|
| concrete | 927 | 0.498 | 0.491 | — |
| white_wine | 4,408 | 0.425 | 0.764 | 4.75× data → 1.56× time |
| california_housing | 18,576 | 0.531 | **2.197** | 4.21× data → **2.87× time** |

`fit` is flat across a 20× range of training-set size (0.43–0.53s) — it only
preprocesses. `predict` has now grown 4.5× over that same range, and the
per-step growth is **accelerating** (1.56× then 2.87×), i.e. still sublinear in n
but heading toward linear. GBDT predict stayed at 1–35 ms throughout.

### Cost of tuning keeps rising

| rung | xgb_tuned fit s | lgbm_tuned fit s | tabpfn total s |
|---|---|---|---|
| concrete | 14.6 | 8.5 | 0.99 |
| white_wine | 15.2 | 12.7 | 1.19 |
| california_housing | 26.2 | 29.6 | 2.73 |

TabPFN is still **9.6× faster than tuned XGBoost and 10.9× faster than tuned
LightGBM** end-to-end, while being more accurate.

---

## Findings

1. **TabPFN-3 sweeps all 10 folds against all 5 GBDT configs** (p = 0.0020), and
   beats the best tuned model by **+0.034 R²**.
2. **In dollar terms:** median-house-value RMSE of **$39,038** vs **$44,399** for
   tuned XGBoost — a **$5,361 (12%)** reduction in typical error. MAE improves
   from $28,915 to $23,308, a **19% reduction**.
3. **TabPFN is also ~10× faster than tuned GBDT** end-to-end (2.7s vs 26–30s).
4. **Untuned GBDT is now 32× faster than TabPFN** (0.085s vs 2.73s) but gives up
   0.054 R². The gap between "instant" and "accurate" is widening with n.
5. **TabPFN's predict cost is scaling with training-set size and accelerating** —
   the single most important number to watch at the next rungs.
6. **XGBoost GPU is *still* slower than CPU** at 20k rows (0.146s vs 0.085s),
   with bit-identical accuracy. GBDT-on-GPU has not paid off at any rung yet.

### Caveats

- The accuracy margin plateauing from 5k→20k is based on **three datasets that
  differ in difficulty as well as size**. A controlled subsample study on one
  dataset would be needed to isolate the effect of n.
- Folds are generated locally (`KFold`, seed 42) rather than from OpenML's
  predefined task splits, because OpenML's metadata API was down (504).

---

## Note on the next rung: there is no 100k dataset in CTR23

Scanning OpenML IDs 44956–45090 found **no regression dataset near 100,000 rows**.
The gap is real:

| dataset | ID | rows | usable? |
|---|---|---|---|
| video_transcoding | 44974 | 68,784 | ⚠️ has string columns (`codec`, `o_codec`) + an `id` column — forces the categorical-encoding decision |
| — | 45022 | 71,090 | ❌ **classification**, not regression (target `readmitted`) |
| sgemm | 44961 | 241,600 | ⚠️ **target leakage**: columns `Run1..Run4` are four repeated timings of the same kernel; Run2–4 must be dropped |
| — | 44986 | 581,835 | target column unverified |
| — | 44998 | 1,000,000 | target column unverified |

So a 100k rung requires either accepting ~69k, jumping to 241k, or subsampling a
larger dataset to exactly 100k.

---

## Reproduce

```bash
srun --gres=gpu:a100:1 -t 01:30:00 -c 8 --mem=64G bash -c \
  'source scripts/env.sh; $PY src/benchmark.py --dataset california_housing \
   --models all --gbdt-device cpu --tabpfn-device cuda --tag a100-reference'

python src/summarize.py --dataset california_housing
python src/compare.py  --dataset california_housing --ref tabpfn_v3
```
