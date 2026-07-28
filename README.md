# TabPFN-3 vs XGBoost vs LightGBM — regression benchmark

Accuracy and wall-clock comparison on OpenML-CTR23 regression datasets, run on Ibex.

## Protocol

| | |
|---|---|
| Task | Regression |
| Data | OpenML-CTR23, parquet pulled directly from `data.openml.org` and cached in `data/` |
| Models | `tabpfn_v3`, `xgb_default`, `xgb_tuned`, `lgbm_default`, `lgbm_tuned`, `xgb_default_gpu` |
| Tuning | GBDT: 30-trial random search, early stopping (50 rounds), inner 80/20 validation split carved from the **train folds only**. TabPFN: library defaults, untuned |
| Splits | 10-fold CV, `KFold(shuffle=True, random_state=42)` |
| Metrics | RMSE, MAE, R² per fold. **R² is the headline** (scale-free, so it aggregates across datasets with different units) |
| Timing | fit / predict / total per fold, CUDA-synchronised. Checkpoint download and CUDA context init are pre-warmed **out** of measurement |
| Hardware | GBDT on 8 pinned CPU cores; TabPFN on A100-SXM4-80GB; same node |

### Two protocol decisions worth knowing

**Tuned GBDT `fit_time` includes the 30-trial search.** That search is a real cost
you pay to get that accuracy, so it belongs in the training time.

**TabPFN inverts the usual fit/predict shape.** It does no gradient training —
the model is frozen, and your training rows are fed in as *context*, like examples
in an LLM prompt. `fit()` only preprocesses; the forward pass over
[train + test] happens in `predict()`. A consequence that will matter on bigger
datasets: **TabPFN's predict time grows with train-set size**, while GBDT's does not.

## Results — `concrete` (n=1030, 8 features)

A100-SXM4-80GB / 8 CPU cores, 10-fold CV.

| model | device | R² | RMSE | fit s | pred s | total s |
|---|---|---|---|---|---|---|
| **tabpfn_v3** | cuda | **0.9537 ± 0.0230** | **3.487** | 0.498 | 0.491 | **0.990** |
| lgbm_tuned | cpu | 0.9473 ± 0.0215 | 3.739 | 8.485 | 0.001 | 8.487 |
| xgb_tuned | cpu | 0.9432 ± 0.0174 | 3.915 | 14.628 | 0.001 | 14.630 |
| xgb_default | cpu | 0.9345 ± 0.0199 | 4.198 | 0.058 | 0.001 | 0.059 |
| xgb_default_gpu | cuda | 0.9345 ± 0.0199 | 4.198 | 0.115 | 0.003 | 0.118 |
| lgbm_default | cpu | 0.9316 ± 0.0213 | 4.291 | 0.055 | 0.001 | 0.055 |

### Paired per-fold comparison (the analysis that matters)

The ±0.02 raw std is mostly **fold difficulty**, which every model sees identically.
Since all models run the same folds, the correct test is paired — the per-fold
difference, whose variance excludes that shared effect.

| vs tabpfn_v3 | mean Δ R² | sd of Δ | folds won | p (Wilcoxon) |
|---|---|---|---|---|
| lgbm_default | +0.0220 | 0.0126 | 10/10 | 0.0020 |
| xgb_default | +0.0191 | 0.0142 | 10/10 | 0.0020 |
| xgb_tuned | +0.0105 | 0.0105 | 8/10 | 0.0195 |
| lgbm_tuned | +0.0064 | 0.0074 | 8/10 | 0.0371 |

### Findings

1. **TabPFN-3 wins on accuracy**, beating every GBDT config on every paired test
   (p < 0.05). Against untuned GBDT it wins all 10 folds.
2. **TabPFN is also faster than tuned GBDT** — 0.99s vs 8.5s (LightGBM) and 14.6s
   (XGBoost), i.e. 8.6× and 14.8× — because it needs no hyperparameter search at all.
3. **But untuned GBDT is ~17× faster than TabPFN** (0.055s vs 0.99s) at 1.9 points
   of R². If you need a model in 50ms, GBDT still owns that regime.
4. **Tuning is worth it and not free**: +0.013 R² for LightGBM at ~150× the fit time.
   A defaults-only comparison would have overstated TabPFN's accuracy lead by ~2×.
5. **XGBoost on GPU is slower than on CPU at this size** (0.118s vs 0.059s) with
   bit-identical accuracy — kernel-launch overhead dominates at n≈1000.
6. Caveat: the TabPFN vs `lgbm_tuned` margin (+0.0064, p=0.037) is the weakest
   result here. It's real but small; one dataset is one dataset.

## Environment notes (Ibex-specific gotchas)

- **torch must be cu128, not the default cu13 build.** Ibex's driver is 570.86.15
  (CUDA 12.8); the default `pip install torch` pulls a CUDA 13 build whose
  `torch.cuda.is_available()` returns False. Install from
  `--index-url https://download.pytorch.org/whl/cu128`.
- **Do not set `--partition=gpu4`** — Ibex forbids naming it directly and routes
  GPU jobs automatically from `--gres` + `--time`. A100s allocate in seconds.
- **TabPFN-3 weights are license-gated.** Accept the license at
  <https://ux.priorlabs.ai> and put `export TABPFN_TOKEN=...` in `.tabpfn_token`
  (mode 600, gitignored).
- **OpenML's metadata API was down (504)** during this work. The harness bypasses
  it entirely by fetching parquet by dataset ID from `data.openml.org`.
  Consequence: folds are generated locally rather than using OpenML's predefined
  task splits, so numbers are reproducible here but not bit-identical to published
  CTR23 results.
- The login node is heavily contended — run everything through compute nodes.

## Usage

```bash
sbatch scripts/run_a100.sbatch concrete     # measurement run (A100, quotable)
python src/summarize.py                     # per-model table
python src/compare.py --ref tabpfn_v3       # paired per-fold test
```

Results append to `results/results.csv`, keyed by
(dataset, model, fold, seed, hardware) — adding a dataset never invalidates
earlier rows.

## Next rungs

Dataset IDs already verified and registered in `src/datasets.py`:

| rung | dataset | n |
|---|---|---|
| ~1k | concrete ✅ | 1030 |
| ~1.5k | airfoil | 1503 |
| ~4k | abalone | 4177 |
| ~10k | grid_stability | 10000 |
| ~20k | california_housing / superconductivity | 20640 / 21263 |
| ~50k | protein / sarcos / diamonds | 45730 / 48933 / 53940 |
| ~70k | video_transcoding | 68784 |
| ~240k | sgemm | 241600 |

Note CTR23 tops out at 241k, so a 500k rung needs a dataset from outside the suite.
TabPFN-3 is rated to 1,000,000 × 200.
