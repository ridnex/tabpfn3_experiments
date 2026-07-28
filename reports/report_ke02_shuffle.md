# Report — KE02-01 shuffled split: TabPFN-3 vs XGBoost vs LightGBM

**Date:** 2026-07-28 · **Run:** Slurm job 49530185 (A100, `gpu201-09-l`, 16 CPU cores) ·
**Results:** `results/results_ke02_shuffle.csv`

**One-line summary:** TabPFN-3 wins outright with zero tuning — but a lookup table
with no features at all scores 0.9915 on this split, so the high R² is mostly the
task being easy, not the models being good.

---

## Setup

| | |
|---|---|
| Data | `data/KE02-01_filtered.xlsx`, 125,948 rows after removing 9,406 exact duplicates |
| Features | 7 sensors: `Choke`, `DHP`, `DHT`, `FLP`, `FLT`, `FTHP`, `FTHT` |
| Target | `Total Massrate` |
| Split | **random 80/20**, `train_test_split(shuffle=True, random_state=42)` |
| | **100,758 train / 25,190 test** |
| Tuning | 30-trial random search, random inner validation (correct here — the outer split is random too) |
| Hardware | TabPFN on A100; GBDTs on 16 CPU cores |

### What question this split answers

**"My flow meter dropped out for an hour. Can I reconstruct it from the sensors,
given labelled data before and after?"** — gap filling and sensor redundancy.

It does **not** answer "can I predict next month's rate", which is
`reports/report_ke02_2025.md` (forward split, every R² negative). Both are real
engineering tasks; the numbers below are only valid for the first one.

---

## Results

| model | device | R² | RMSE | MAE | search s | train s | inference s | **total s** |
|---|---|---|---|---|---|---|---|---|
| **tabpfn_v3** | cuda | **0.9981** | **213.8** | **115.6** | — | 0.86 | 35.74 | 36.59 |
| lgbm_tuned | cpu | 0.9967 | 280.7 | 141.3 | 181.51 | 7.41 | 0.31 | 189.24 |
| xgb_tuned | cpu | 0.9966 | 282.8 | 141.9 | 79.24 | 5.02 | 0.08 | 84.34 |
| xgb_default | cpu | 0.9939 | 381.1 | 236.5 | — | 0.14 | 0.003 | **0.14** |
| lgbm_default | cpu | 0.9899 | 489.7 | 324.9 | — | 0.37 | 0.005 | 0.38 |
| *mean_baseline* | cpu | *−0.000* | *4874.4* | *3749.6* | — | — | — | — |

`search` = the 30-trial hyperparameter search. `train` = fitting the winning
config on the full training set. Split out because reporting a 3-minute search as
"training time" would hide that the final LightGBM model fits in 7.4 s.

`mean_baseline` predicts the training mean and ignores the sensors. Its R² of
exactly 0.000 is the correctness check for this split: under shuffling, train and
test means coincide, which is precisely what makes R² well-behaved here and
harshly negative on the forward split.

---

## Finding 1 — TabPFN-3 wins, and it is not close on error

TabPFN has **24% lower RMSE** than the best GBDT (213.8 vs 280.7) and **18% lower
MAE**, with **no hyperparameter search at all**. LightGBM spent 181 seconds
searching to land 24% behind.

This is the result the ladder predicted. TabPFN was strongest on the small iid
rungs of this project and it holds that lead at 100k rows on an iid-style split.
It is also the exact opposite of the forward split, where the same model on the
same data comes last (R² −2.679). **The split, not the model, decides the winner.**

## Finding 2 — the high R² is the task, not the models

Ignore all 7 sensors and simply **copy the target from the nearest row in time
that happens to be in the training set**:

| "model" | R² |
|---|---|
| copy nearest training row in time (**zero features**) | **0.9915** |
| lgbm_default | 0.9899 |
| xgb_default | 0.9939 |
| xgb_tuned | 0.9966 |
| lgbm_tuned | 0.9967 |
| tabpfn_v3 | 0.9981 |

A lookup table beats untuned LightGBM. The reason:

- Rows are **10 minutes apart**, and consecutive readings differ by a median of
  **54 units** against an overall spread of **4,960** — a neighbour is ~92× closer
  than a random row.
- After an 80/20 shuffle, **79.9% of test rows have their direct neighbour in the
  training set**; the median distance to the nearest training row is **1 row
  (10 minutes)**.

So every model here is largely interpolating between two nearby known points. The
real contest is over the last 0.7% of R², which is why the five models span only
0.9899–0.9981 while their costs span 0.14 s to 189 s.

**This is legitimate if you are filling gaps** — in that deployment the neighbours
genuinely exist. It is not evidence that the sensors predict flow rate.

## Finding 3 — tuning is not worth it on this split

| library | default R² | tuned R² | gain | time cost |
|---|---|---|---|---|
| XGBoost | 0.9939 | 0.9966 | +0.0027 | 0.14 s → 84.3 s (**600×**) |
| LightGBM | 0.9899 | 0.9967 | +0.0068 | 0.38 s → 189.2 s (**500×**) |

Both searches ran to the round cap (`best_rounds` 2000 and 1996 of a 2000
maximum) — early stopping never fired, because with a random inner validation
split more trees always improve interpolation. Contrast the forward split, where
the same search picked **8 trees** and finished in 6 seconds.

On the forward split tuning was clearly worth it (LightGBM +0.71 R²). Here it buys
+0.007 for three minutes.

## Finding 4 — cost

| model | total s | vs TabPFN |
|---|---|---|
| xgb_default | 0.14 | **264× faster** |
| lgbm_default | 0.38 | 97× faster |
| **tabpfn_v3** | **36.59** | — |
| xgb_tuned | 84.34 | 2.3× slower |
| lgbm_tuned | 189.24 | 5.2× slower |

TabPFN's shape is unchanged from every rung in this project: **fit 0.86 s,
inference 35.74 s** — 98% of its wall-clock is inference, scaling with the size of
the training set it carries as context.

**`xgb_default` reaches R² 0.9939 in 0.14 seconds** — 99.6% of TabPFN's accuracy at
1/264th the cost. If this is a gap-filling job with any volume, that trade is hard
to argue against. TabPFN wins on accuracy; XGBoost at defaults wins on engineering.

---

## Comparison with the forward split

Same data, same models, same code — only the split differs:

| model | shuffled R² | forward R² |
|---|---|---|
| tabpfn_v3 | **0.9981** (1st) | −2.679 (**last**) |
| lgbm_tuned | 0.9967 | **−0.543** (1st) |
| xgb_tuned | 0.9966 | −0.661 |
| xgb_default | 0.9939 | −0.953 |
| lgbm_default | 0.9899 | −1.251 |

TabPFN goes from first to last. The ranking fully inverts. Quoting 0.9981 as
evidence that a forward-looking virtual flow meter works would be wrong by about
3.7 units of R².

---

## Caveats

- **Single split, single seed.** One number per model, no error bars. Differences
  under ~0.001 R² should not be read as a ranking.
- **Valid only for gap filling.** See the comparison above.
- **The file is pre-filtered** at a hard 15,000 floor, so no model sees low-rate or
  shut-in operation.
- **TabPFN is untuned by design** and compared against a 30-trial search — which,
  on this split, makes its win more impressive rather than less.
- The nearest-neighbour figure (0.9915) uses the true target of adjacent rows, so
  it is a diagnostic of how much information leaks across a shuffle, not a
  deployable model.

---

## Reproduce

```bash
srun --gres=gpu:a100:1 -t 00:40:00 -c 16 --mem=64G bash -c \
  'source scripts/env.sh; $PY src/benchmark.py --dataset ke02_shuffle \
   --models mean_baseline,tabpfn_v3,xgb_default,lgbm_default,xgb_tuned,lgbm_tuned \
   --folds 1 --gbdt-device cpu --tabpfn-device cuda'

python src/summarize.py --results results/results_ke02_shuffle.csv
```

Registry entry `ke02_shuffle` in `src/datasets.py` (`cv="shuffle"`); the split
itself is the `cv == "shuffle"` branch in `src/benchmark.py`.
