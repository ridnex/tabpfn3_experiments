# Report — 5k rung: TabPFN-3 vs XGBoost vs LightGBM

**Date:** 2026-07-26 · **Run:** Slurm job 49456694 (interactive `srun`, A100) · **Results:** `results/results.csv`

---

## Dataset

| | |
|---|---|
| Name | `white_wine` (wine quality — white) |
| Source | OpenML-CTR23 suite, dataset ID **44971** |
| Shape | **4898 rows × 12 columns** |
| Features | **11** (all numeric) |
| Target | `quality` |
| Missing values | none |
| Categorical features | none |
| Task | Regression |

**Feature columns:** `fixed_acidity`, `volatile_acidity`, `citric_acid`,
`residual_sugar`, `chlorides`, `free_sulfur_dioxide`, `total_sulfur_dioxide`,
`density`, `pH`, `sulphates`, `alcohol`

**Important caveat on the target.** `quality` is **integer-valued (3–9)**, i.e. a
discretised regression target rather than a continuous one. This caps how high
any model's R² can go and is why all scores here sit near 0.5–0.6 instead of the
0.93–0.95 seen on the 1k rung. That is a property of the dataset, not a
regression in model quality. The CTR23 suite offers no genuinely continuous
target near 5k rows, so this is a real limitation of the ladder at this size.

Chosen as the 5k rung because at 4898 rows it is the closest available to 5k
while keeping feature count low (11 vs concrete's 8) — so **n is essentially the
only thing that changed** from the 1k rung, which is what makes the scaling
comparison interpretable.

---

## Setup

Identical protocol to the 1k rung.

| | |
|---|---|
| Validation | 10-fold CV, `KFold(shuffle=True, random_state=42)` |
| GPU | NVIDIA A100-SXM4-80GB (`device="cuda"`) |
| CPU | 8 pinned cores (`device="cpu"`, OMP threads pinned to match) |
| Tuning | GBDT: 30-trial random search + early stopping (50 rounds), inner 80/20 validation split from train folds only. TabPFN: library defaults, untuned |
| Timing | fit / predict / total per fold, CUDA-synchronised; checkpoint download and CUDA context init pre-warmed **out** of measurement |
| Software | tabpfn 8.1.0, xgboost 3.2.0, lightgbm 4.7.0, torch 2.11.0+cu128, Python 3.11 |

---

## Results

| model | device | R² | RMSE | MAE | fit s | pred s | total s |
|---|---|---|---|---|---|---|---|
| **tabpfn_v3** | cuda | **0.5985 ± 0.0468** | **0.559 ± 0.030** | **0.352 ± 0.020** | 0.425 | 0.764 | **1.189 ± 0.039** |
| lgbm_tuned | cpu | 0.5636 ± 0.0473 | 0.583 ± 0.030 | 0.397 ± 0.021 | 12.679 | 0.004 | 12.684 ± 2.699 |
| xgb_tuned | cpu | 0.5605 ± 0.0451 | 0.585 ± 0.028 | 0.395 ± 0.017 | 15.240 | 0.002 | 15.242 ± 2.439 |
| xgb_default | cpu | 0.5020 ± 0.0462 | 0.623 ± 0.029 | 0.442 ± 0.017 | 0.066 | 0.001 | 0.066 ± 0.001 |
| xgb_default_gpu | cuda | 0.5020 ± 0.0462 | 0.623 ± 0.029 | 0.442 ± 0.017 | 0.151 | 0.004 | 0.155 ± 0.074 |
| lgbm_default | cpu | 0.4828 ± 0.0414 | 0.635 ± 0.024 | 0.482 ± 0.017 | 0.057 | 0.001 | 0.058 ± 0.000 |

Sorted by R². Values are mean ± std across the 10 folds.
`cuda` = GPU (A100), `cpu` = 8 CPU cores.

### Paired per-fold comparison vs TabPFN-3

| vs tabpfn_v3 | mean Δ R² | sd of Δ | folds won | p (Wilcoxon) |
|---|---|---|---|---|
| lgbm_default | +0.1157 | 0.0231 | **10/10** | 0.0020 |
| xgb_default | +0.0965 | 0.0131 | **10/10** | 0.0020 |
| xgb_default_gpu | +0.0965 | 0.0131 | **10/10** | 0.0020 |
| xgb_tuned | +0.0379 | 0.0117 | **10/10** | 0.0020 |
| lgbm_tuned | +0.0349 | 0.0133 | **10/10** | 0.0020 |

Positive Δ means TabPFN-3 is better. **TabPFN wins every fold against every
config** — a clean sweep, unlike the 1k rung where the tuned models took 2 folds.

---

## Cross-rung comparison (1k → 5k)

### Accuracy margin — TabPFN's lead widened

| vs tabpfn_v3 | 1k (concrete) | 5k (white_wine) |
|---|---|---|
| lgbm_tuned | +0.0064 (8/10, p=0.037) | **+0.0349 (10/10, p=0.002)** |
| xgb_tuned | +0.0105 (8/10, p=0.020) | **+0.0379 (10/10, p=0.002)** |
| xgb_default | +0.0191 (10/10, p=0.002) | +0.0965 (10/10, p=0.002) |
| lgbm_default | +0.0220 (10/10, p=0.002) | +0.1157 (10/10, p=0.002) |

TabPFN's advantage over *tuned* GBDT grew ~5×, from a marginal result to a
decisive one. Note this confounds two changes — more rows **and** a harder,
noisier dataset — so it is not yet evidence that the lead grows with n alone.

### Timing — the context-length effect is now visible

TabPFN, per fold:

| rung | rows | train rows/fold | fit s | predict s |
|---|---|---|---|---|
| concrete | 1030 | ~927 | 0.498 | 0.491 |
| white_wine | 4898 | ~4408 | 0.425 | **0.764** |

Training data grew **4.75×**. `fit` stayed flat (it only preprocesses), while
**`predict` grew 1.56×** — clearly sublinear so far. This is the predicted
consequence of TabPFN using the training rows as *context*: its inference cost
scales with train-set size, unlike GBDT.

GBDT predict time, by contrast, stayed pinned at 1–4 ms at both rungs, because a
fitted tree ensemble does not care how much data built it.

---

## Findings

1. **TabPFN-3 sweeps every fold against every config** (10/10, p = 0.0020 across
   the board) — a stronger result than the 1k rung.
2. **TabPFN also beats tuned GBDT on wall-clock:** 1.19s vs 12.7s (LightGBM,
   10.7×) and 15.2s (XGBoost, 12.8×), because it runs no hyperparameter search.
3. **Untuned GBDT remains ~20× faster than TabPFN** (0.058s vs 1.19s) — but now
   costs 0.10–0.12 R², a far worse trade than at 1k. The "just use default
   LightGBM" option got materially weaker as the problem got harder.
4. **MAE again favours TabPFN more than R² does:** 0.352 vs 0.397 (tuned) and
   0.482 (lgbm default) — a 27% error reduction against the default.
5. **XGBoost on GPU is still slower than on CPU** (0.155s vs 0.066s) at identical
   accuracy. At n≈5k, data-transfer and kernel-launch overhead still dominate.
6. **TabPFN's `predict` cost is starting to scale with n** (+56% for 4.75× the
   data) while its `fit` stays flat — the effect that will eventually determine
   its practical ceiling.

### Caveats

- The `quality` target is discretised (integers 3–9), so absolute R² values are
  not comparable to the 1k rung's continuous target.
- The widened accuracy gap confounds **dataset difficulty** with **dataset size**.
  Separating them would need a controlled subsample of one dataset.
- Folds are generated locally (`KFold`, seed 42) rather than from OpenML's
  predefined task splits, because OpenML's metadata API was down (504).
  Reproducible here, but not bit-identical to published CTR23 numbers.

---

## Environment note added at this rung

**PyTorch has dropped Volta support.** A V100 (compute capability 7.0) fails with
`CUDA error: no kernel image is available for execution on the device` — this
torch build targets CC 7.5/8.0/8.6/9.0/10.0/12.0. On Ibex that leaves **A100
(8.0)** and **RTX 2080 Ti (7.5)** as the only usable GPUs; V100, P100, GTX 1080 Ti
and P6000 are all unusable for TabPFN.

**Use `srun`, not a long `sbatch`.** A100s allocate in seconds interactively,
while a 4-hour `sbatch` reservation sits in `PENDING (Priority)` indefinitely.
The 1k run took 4m13s; the `--time` request in `run_a100.sbatch` is now 30 min.

---

## Reproduce

```bash
srun --gres=gpu:a100:1 -t 00:40:00 -c 8 --mem=64G bash -c \
  'source scripts/env.sh; $PY src/benchmark.py --dataset white_wine --models all \
   --gbdt-device cpu --tabpfn-device cuda --tag a100-reference'

python src/summarize.py --dataset white_wine
python src/compare.py  --dataset white_wine --ref tabpfn_v3
```
