# Report — 1k rung: TabPFN-3 vs XGBoost vs LightGBM

**Date:** 2026-07-26 · **Run:** Slurm job 49455654 · **Results:** `results/results.csv` (60 rows)

---

## Dataset

| | |
|---|---|
| Name | `concrete_compressive_strength` |
| Source | OpenML-CTR23 suite, dataset ID **44959** |
| Shape | **1030 rows × 9 columns** |
| Features | **8** (all numeric) |
| Target | `strength` (concrete compressive strength, MPa) |
| Missing values | none |
| Categorical features | none |
| Task | Regression |

**Feature columns:** `cement`, `blast_furnace_slag`, `fly_ash`, `water`,
`superplasticizer`, `coarse_aggregate`, `fine_aggregate`, `age`

Chosen as the first rung because it is small, all-numeric and complete — so the
first run exercised the benchmark harness rather than preprocessing decisions.

---

## Setup

| | |
|---|---|
| Validation | 10-fold CV, `KFold(shuffle=True, random_state=42)` |
| GPU | NVIDIA A100-SXM4-80GB |
| CPU | 8 pinned cores (`--cpus-per-task=8`, OMP threads pinned to match) |
| Tuning | GBDT: 30-trial random search + early stopping (50 rounds), inner 80/20 validation split from train folds only. TabPFN: library defaults, untuned |
| Timing | fit / predict / total per fold, CUDA-synchronised; checkpoint download and CUDA context init pre-warmed **out** of measurement |
| Software | tabpfn 8.1.0, xgboost 3.2.0, lightgbm 4.7.0, torch 2.11.0+cu128, Python 3.11 |

`fit_time` for the tuned GBDT configs **includes the 30-trial search**, because
that search is a real cost paid to reach that accuracy.

---

## Results

| model | device | R² | RMSE | MAE | fit s | pred s | total s |
|---|---|---|---|---|---|---|---|
| **tabpfn_v3** | cuda | **0.9537 ± 0.0230** | **3.487 ± 0.853** | **2.002 ± 0.325** | 0.498 | 0.491 | **0.990 ± 0.202** |
| lgbm_tuned | cpu | 0.9473 ± 0.0215 | 3.739 ± 0.655 | 2.497 ± 0.260 | 8.485 | 0.001 | 8.487 ± 1.127 |
| xgb_tuned | cpu | 0.9432 ± 0.0174 | 3.915 ± 0.607 | 2.553 ± 0.253 | 14.628 | 0.001 | 14.630 ± 2.111 |
| xgb_default | cpu | 0.9345 ± 0.0199 | 4.198 ± 0.615 | 2.710 ± 0.234 | 0.058 | 0.001 | 0.059 ± 0.002 |
| xgb_default_gpu | cuda | 0.9345 ± 0.0199 | 4.198 ± 0.615 | 2.710 ± 0.234 | 0.115 | 0.003 | 0.118 ± 0.008 |
| lgbm_default | cpu | 0.9316 ± 0.0213 | 4.291 ± 0.657 | 2.879 ± 0.253 | 0.055 | 0.001 | 0.055 ± 0.001 |

Sorted by R². Values are mean ± std across the 10 folds.

### Paired per-fold comparison vs TabPFN-3

The ±0.02 spread above is mostly **fold difficulty**, which every model sees
identically. All models run the same folds, so the correct test is paired: the
per-fold difference, whose variance excludes that shared effect.

| vs tabpfn_v3 | mean Δ R² | sd of Δ | folds won | p (Wilcoxon) |
|---|---|---|---|---|
| lgbm_default | +0.0220 | 0.0126 | 10/10 | 0.0020 |
| xgb_default | +0.0191 | 0.0142 | 10/10 | 0.0020 |
| xgb_default_gpu | +0.0191 | 0.0142 | 10/10 | 0.0020 |
| xgb_tuned | +0.0105 | 0.0105 | 8/10 | 0.0195 |
| lgbm_tuned | +0.0064 | 0.0074 | 8/10 | 0.0371 |

Positive Δ means TabPFN-3 is better.

---

## Findings

1. **TabPFN-3 wins on accuracy**, beating every GBDT config on every paired test
   (p < 0.05). Against untuned GBDT it wins all 10 folds.
2. **TabPFN is also faster than tuned GBDT** — 0.99s vs 8.5s (LightGBM, 8.6×) and
   14.6s (XGBoost, 14.8×) — because it runs no hyperparameter search at all.
3. **Untuned GBDT is still ~17× faster than TabPFN** (0.055s vs 0.99s), at a cost
   of ~1.9 points of R². For sub-100ms training, GBDT still owns that regime.
4. **TabPFN's MAE lead (20%) is far larger than its R² lead (<1%).** R² is
   dominated by squared error, so a few hard outliers compress the gap. On the
   typical row TabPFN is substantially more accurate than the R² column suggests.
5. **Tuning matters and is not free:** +0.013 R² for LightGBM at ~150× the fit
   time. A defaults-only comparison would have overstated TabPFN's lead by ~2×.
6. **XGBoost on GPU was slower than on CPU** at this size (0.118s vs 0.059s) with
   bit-identical accuracy — kernel-launch overhead dominates at n≈1000.

### Caveats

- The TabPFN vs `lgbm_tuned` margin (+0.0064, p=0.037) is the weakest result
  here — real, but small.
- **One dataset, 1030 rows.** Nothing here generalizes yet; that is what the
  larger rungs are for.
- OpenML's metadata API was down (504) during this run, so folds were generated
  locally instead of using OpenML's predefined task splits. Reproducible here,
  but not bit-identical to published CTR23 numbers.
- TabPFN's `predict` cost grows with **training**-set size (the train rows are
  its context). At 1k rows that is invisible; it is expected to become the
  dominant effect on the larger rungs.

---

## Reproduce

```bash
sbatch scripts/run_a100.sbatch concrete
python src/summarize.py --dataset concrete
python src/compare.py --dataset concrete --ref tabpfn_v3
```
