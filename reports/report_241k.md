# Report — 241k rung: TabPFN-3 vs XGBoost vs LightGBM

**Date:** 2026-07-27 · **Results:** `results/results.csv`

**Runs (split into two jobs):**

| job | node | hardware | models | elapsed |
|---|---|---|---|---|
| 49456911 | `gpu108-16-l` | A100-SXM4-80GB + EPYC 7713P (Milan), 8 cores | tabpfn_v3, xgb_default, lgbm_default, xgb_default_gpu | 22m58s |
| 49482955 | `cn604-10` | **no GPU** + EPYC 9655 (Turin), 8 cores | xgb_tuned, lgbm_tuned | 59m34s |

> **This rung substitutes for the requested 100k.** OpenML-CTR23 has no regression
> dataset near 100,000 rows (see the gap table in `report_20k.md`). sgemm at
> **241,600** rows is the next usable rung up from 20,640.

---

## Dataset

| | |
|---|---|
| Name | `sgemm` (SGEMM GPU kernel performance) |
| Source | OpenML-CTR23 suite, dataset ID **44961** |
| Shape | **241,600 rows × 14 features** (18 columns raw, see below) |
| Target | `Run1` — kernel runtime in ms |
| Target range | 13.29 … 3339.63, mean 217.65, **median 69.82** (heavy right skew) |
| Missing values | none |
| Categorical features | none encoded, but see cardinality note |
| Task | Regression |

**Feature columns:** `MWG`, `NWG`, `KWG`, `MDIMC`, `NDIMC`, `MDIMA`, `NDIMB`,
`KWI`, `VWM`, `VWN`, `STRM`, `STRN`, `SA`, `SB`

### Leakage removal — mandatory

The raw parquet has **18** columns: 14 kernel-configuration parameters plus
`Run1..Run4`, which are **four repeated timing measurements of the same kernel
configuration**. Predicting `Run1` with `Run2..Run4` left in `X` gives every
model R² ≈ 0.99 for free and measures nothing. `datasets.py` drops them
explicitly (`drop=("Run2","Run3","Run4")`); the load log confirms `X=(241600, 14)`.

### Every feature is low-cardinality

| feature | levels | feature | levels |
|---|---|---|---|
| MWG, NWG, VWM, VWN | 4 | MDIMC, NDIMC, MDIMA, NDIMB | 3 |
| KWG, KWI, STRM, STRN, SA, SB | 2 | | |

This is a **filtered factorial grid**, not continuous data — 14 ordinal
parameters with 2–4 levels each. That matters for interpretation: axis-aligned
tree splits can partition such a grid almost exactly, which is a structural
advantage for GBDT that did not exist at any earlier rung.

---

## Setup

Protocol identical to earlier rungs, with one hardware deviation (below).

| | |
|---|---|
| Validation | 10-fold CV, `KFold(shuffle=True, random_state=42)` — 217,440 train rows/fold |
| GPU | A100-SXM4-80GB for `tabpfn_v3` and `xgb_default_gpu` |
| CPU | 8 pinned cores |
| Tuning | GBDT: 30-trial random search + early stopping (50 rounds), inner 80/20 split from train folds only. TabPFN: library defaults, untuned |
| Timing | fit / predict / total per fold, CUDA-synchronised; warmup excluded. Tuned `fit_time` **includes** the full 30-trial search |
| Software | tabpfn 8.1.0, xgboost 3.2.0, lightgbm 4.7.0, torch 2.11.0+cu128, Python 3.11 |

**Hardware deviation.** The tuned GBDT configs need no GPU (`warmup()` only
initialises CUDA when a selected model asks for it), so they ran on a CPU-only
node, which allocated in <20s versus a long A100 queue. That node has a **newer,
faster CPU** (EPYC 9655 Turin vs EPYC 7713P Milan). Consequence: the tuned fit
times below are, if anything, **optimistic** — on the same node as the defaults
they would be slower, so the cost-of-tuning conclusion is conservative.
Accuracy is unaffected (deterministic given the seed).

---

## Results

| model | device | R² | RMSE | MAE | fit s | pred s | total s |
|---|---|---|---|---|---|---|---|
| **xgb_tuned** | cpu | **0.9995 ± 0.0000** | **8.224 ± 0.243** | 4.579 | 186.292 | 0.170 | 186.462 ± 2.082 |
| lgbm_tuned | cpu | 0.9994 ± 0.0000 | 9.095 ± 0.190 | 5.156 | 170.207 | 0.245 | 170.452 ± 0.327 |
| tabpfn_v3 | cuda | 0.9987 ± 0.0002 | 13.073 ± 0.864 | **2.944** | 1.769 | **134.405** | 136.174 ± 0.074 |
| lgbm_default | cpu | 0.9902 ± 0.0003 | 36.522 ± 0.560 | 24.492 | 0.281 | 0.009 | 0.290 ± 0.005 |
| xgb_default | cpu | 0.9849 ± 0.0007 | 45.373 ± 1.201 | 25.477 | 0.331 | 0.005 | 0.335 ± 0.105 |
| xgb_default_gpu | cuda | 0.9849 ± 0.0007 | 45.373 ± 1.201 | 25.477 | 0.193 | 0.005 | 0.198 ± 0.010 |

Sorted by R². Mean ± std across 10 folds. RMSE/MAE in milliseconds.

**Read RMSE, not R², at this rung.** Every model scores R² > 0.98, which makes
the differences look like rounding. They are not: xgb_default → xgb_tuned is
R² +0.0146 but **RMSE 45.4 → 8.2, a 5.5× error reduction**. Once R² > 0.95 there
is so little variance left to explain that large error changes barely move it.

### Paired per-fold comparison vs TabPFN-3

All models ran identical folds, so the paired difference is the correct test.
Positive = TabPFN is better.

| vs tabpfn_v3 | Δ R² | Δ RMSE | Δ MAE | R² folds won | p |
|---|---|---|---|---|---|
| xgb_tuned | **−0.0008** | **−4.849** | +1.635 | **0/10** | 0.0020 |
| lgbm_tuned | **−0.0007** | **−3.978** | +2.212 | **0/10** | 0.0020 |
| lgbm_default | +0.0085 | +23.449 | +21.548 | 10/10 | 0.0020 |
| xgb_default | +0.0139 | +32.300 | +22.534 | 10/10 | 0.0020 |
| xgb_default_gpu | +0.0139 | +32.300 | +22.534 | 10/10 | 0.0020 |

**TabPFN-3 loses a rung for the first time.** It is beaten by both tuned GBDTs on
R² and RMSE in **all 10 folds** (p = 0.0020). It still beats both defaults 10/10.

### The metrics disagree — and that is the interesting part

TabPFN loses on RMSE but **wins MAE against every model, 10/10, p = 0.0020**:

| model | MAE | RMSE | RMSE / MAE |
|---|---|---|---|
| **tabpfn_v3** | **2.944** | 13.073 | **4.44** |
| xgb_tuned | 4.579 | 8.224 | 1.80 |
| lgbm_tuned | 5.156 | 9.095 | 1.76 |
| xgb_default | 25.477 | 45.373 | 1.78 |
| lgbm_default | 24.492 | 36.522 | 1.49 |

RMSE penalises large errors quadratically; MAE does not. A Gaussian error
distribution gives RMSE/MAE ≈ 1.25. The GBDTs sit at 1.5–1.8. **TabPFN sits at
4.44** — several times more tail-heavy than anything else here.

So the two metrics are describing different things, and both are true:

- **On the typical kernel, TabPFN is the most accurate model on the board** —
  median-ish error 2.94 ms vs 4.58 ms for tuned XGBoost, a 36% reduction.
- **On the hard tail it is much worse**, and the tail is where the target lives:
  the distribution runs 13 → 3340 ms with a median of only 69.8, so a small
  number of very slow kernels dominate the squared error.

Plausible reading: on a discrete 14-parameter grid, trees can carve out the
extreme-runtime corners exactly, whereas TabPFN's attention-based interpolation
smooths across them. **Which model is "better" here depends entirely on whether
your application cares about typical-case or worst-case error.** Reporting only
R² would have hidden this completely.

---

## Cross-rung trends (1k → 5k → 20k → 241k)

### TabPFN's accuracy lead over tuned GBDT collapses

| rung | n | Δ R² vs xgb_tuned | folds won |
|---|---|---|---|
| concrete | 1,030 | +0.0105 | 8/10 |
| white_wine | 4,898 | +0.0379 | 10/10 |
| california_housing | 20,640 | +0.0336 | 10/10 |
| **sgemm** | **241,600** | **−0.0008** | **0/10** |

### TabPFN predict time — super-linear, and now the dominant cost

| rung | train rows/fold | fit s | predict s | vs previous |
|---|---|---|---|---|
| concrete | 927 | 0.498 | 0.491 | — |
| white_wine | 4,408 | 0.425 | 0.764 | 4.75× data → 1.56× time |
| california_housing | 18,576 | 0.531 | 2.197 | 4.21× data → 2.87× time |
| **sgemm** | **217,440** | 1.769 | **134.405** | **11.7× data → 61.2× time** |

`fit` stayed near-flat across a 235× range of training-set size (0.43 → 1.77s) —
it only preprocesses. `predict` grew **274×**. Between the last two rungs the
scaling is roughly **n^1.6**, i.e. clearly super-linear and no longer sublinear
as it looked at 5k. This is the structural consequence of the training set being
the *context*: every prediction re-attends over all 217k training rows.

### Cost of tuning, and why it is a different kind of cost

| rung | xgb_default RMSE → xgb_tuned RMSE | gain | fit cost ratio |
|---|---|---|---|
| concrete | 4.198 → 3.915 | 1.07× | 251× |
| white_wine | 0.623 → 0.585 | 1.07× | 232× |
| california_housing | 47,321 → 44,399 | 1.07× | 313× |
| **sgemm** | **45.373 → 8.224** | **5.52×** | **564×** |

Three rungs at a flat 1.07×, then a 5.5× jump. sgemm is **near-deterministic** —
kernel runtime is an almost noiseless function of the 14 parameters — so extra
capacity (low learning rate + many trees under early stopping, up to
`MAX_ROUNDS=2000`) has real signal to capture and nothing to overfit. On noisy
targets like `white_wine` most residual error is irreducible, so tuning cannot
buy much. **Tuning buys resolution; it only pays when the data has resolution
left to give.**

Critically, the 186s is a **one-time, amortisable** cost. Serving cost per batch:

| model | predict s | relative |
|---|---|---|
| xgb_default | 0.005 | 1× |
| xgb_tuned | 0.170 | 34× |
| **tabpfn_v3** | **134.405** | **26,900×** |

Tuned XGBoost is 34× slower to score than default (more trees) and still
negligible. TabPFN pays 134s on **every** batch, permanently — that cost never
amortises.

---

## Findings

1. **First rung TabPFN-3 loses.** Both tuned GBDTs beat it on R² and RMSE in
   10/10 folds (p = 0.0020). Its lead over tuned GBDT went +0.034 R² at 20k to
   −0.0008 at 241k.
2. **But TabPFN wins MAE against everything, 10/10** (2.944 vs 4.579). It has the
   best typical-case accuracy and the worst error tail (RMSE/MAE = 4.44 vs ~1.8).
   The headline depends on which error you care about.
3. **TabPFN's inference cost is now prohibitive**: 134.4s per fold vs 0.17s for
   tuned XGBoost — **790× slower**, on an A100 versus 8 CPU cores.
4. **Predict time scales ~n^1.6** and has grown 274× across the ladder while fit
   stayed flat. This, not accuracy, is TabPFN's practical ceiling.
5. **Tuning paid off 5.5× here** versus a flat 1.07× at every earlier rung,
   because the target is near-noiseless.
6. **XGBoost GPU is finally faster than CPU** — 0.193s vs 0.331s fit at 241k,
   with bit-identical accuracy. This is the first rung where GBDT-on-GPU wins;
   at 1k/5k/20k the transfer overhead dominated.

### Caveats

- **Tuned and default GBDT ran on different nodes with different CPUs** (Turin vs
  Milan). Default-vs-tuned *timing* ratios mix two CPUs; the tuned node is
  faster, so the reported cost of tuning is an underestimate. Accuracy is
  unaffected.
- **R² is nearly saturated** (all models > 0.98), so it is a poor discriminator
  at this rung. RMSE and MAE carry the signal.
- **This rung is structurally friendly to trees**: a discrete factorial grid of
  2–4-level parameters. The result should not be read as "GBDT beats TabPFN above
  200k rows" in general — dataset structure changed along with n.
- Only one dataset per rung, so **size and difficulty remain confounded** across
  the whole ladder.
- Folds are generated locally (`KFold`, seed 42) rather than from OpenML's
  predefined task splits, because OpenML's metadata API was down (504).

### Two issues found and fixed while writing this report

- **`src/compare.py` reported RMSE/MAE backwards.** `wins` was hardcoded as
  `(diff > 0).sum()` with the legend "mean diff > 0 => reference is better",
  which is only correct for higher-is-better metrics. Fixed with a
  `LOWER_IS_BETTER` set that sign-flips the difference, so positive always means
  the reference won. Earlier reports used only the `r2` table and are unaffected.
- **`best_rounds` / `val_rmse` were not recorded.** `benchmark.py` writes
  `notes = args.tag or str(info)`, so passing `--tag` discards the per-fold
  diagnostics the tuned runners return. We therefore cannot say how many trees
  the search actually chose at this rung. Worth a schema change (a separate
  `info` column) before the next rung.

---

## Reproduce

```bash
# fast models, needs the A100
srun --gres=gpu:a100:1 -t 00:50:00 -c 8 --mem=96G bash -c \
  'source scripts/env.sh; $PY src/benchmark.py --dataset sgemm \
   --models tabpfn_v3,xgb_default,lgbm_default,xgb_default_gpu \
   --gbdt-device cpu --tabpfn-device cuda --tag a100-reference'

# tuned GBDT, no GPU needed - schedules far faster
srun -t 03:00:00 -c 8 --mem=96G bash -c \
  'source scripts/env.sh; $PY src/benchmark.py --dataset sgemm \
   --models xgb_tuned,lgbm_tuned --gbdt-device cpu --tabpfn-device cpu \
   --tag cpu-only-node-tuned'

python src/summarize.py --dataset sgemm
for m in r2 rmse mae; do python src/compare.py --dataset sgemm --ref tabpfn_v3 --metric $m; done
```
