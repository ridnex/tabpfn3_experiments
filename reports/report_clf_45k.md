# Report — 45k classification rung: TabPFN-3 vs XGBoost vs LightGBM

**Date:** 2026-07-27 · **Run:** Slurm job 49497423 (`srun`, A100, `gpu203-23-l`), 25m33s ·
**Results:** `results/results_clf.csv`

> First **classification** rung. All five previous rungs were regression, so this
> required a genuine task abstraction in the harness — see *Harness changes* below.

---

## Dataset

| | |
|---|---|
| Name | `electricity` (Australian New South Wales electricity market) |
| Source | OpenML ID **151** |
| Shape | **45,312 rows × 8 features** |
| Target | `class` — price movement, binary |
| Label mapping | `{0: DOWN, 1: UP}` (fixed by sorted order, printed on every load) |
| Class balance | 26,075 DOWN / 19,237 UP — **42.5% minority** |
| Missing values | **none** |
| Categorical features | 1 (`day`, 7 levels) |

**Feature columns:** `date`, `day`, `period`, `nswprice`, `nswdemand`,
`vicprice`, `vicdemand`, `transfer`

### Why this dataset

Three candidates near 50k were checked directly against the data server:

| candidate | rows | minority | missing | non-numeric feats | verdict |
|---|---|---|---|---|---|
| `adult` | 48,842 | 23.9% | **6,465 cells** | 8 (incl. `native-country`, 41 levels) | rejected |
| `bank-marketing` | 45,211 | **11.7%** | 0 | 9 | rejected |
| **`electricity`** | **45,312** | **42.5%** | **0** | **1** | **chosen** |

`adult` would have made imputation and high-cardinality encoding part of what is
measured; `bank-marketing`'s imbalance would have put all weight on one metric.
`electricity` isolates model quality. (`jannis` and `higgs` were excluded outright
— their last column is not the target.)

---

## Setup

| | |
|---|---|
| Validation | **StratifiedKFold**(10, shuffle=True, random_state=42) — 40,780 train rows/fold |
| GPU | NVIDIA A100-SXM4-80GB (`tabpfn_v3`) |
| CPU | 8 pinned cores (all four GBDT configs) |
| Tuning | 30-trial random search + early stopping (50 rounds), inner 80/20 split from train folds only, selected on **log-loss**. TabPFN: library defaults, untuned |
| Timed predict | **one `predict_proba` call**; hard labels by argmax — no second `predict()` |
| Headline metric | **ROC-AUC** (threshold-free; the role R² played for regression) |
| Software | tabpfn 8.1.0, xgboost 3.2.0, lightgbm 4.7.0, torch 2.11.0+cu128, Python 3.11 |

Tuning selects on log-loss rather than accuracy because it is a proper scoring
rule — it optimises the calibrated probabilities that both ROC-AUC and log-loss
are read from, instead of a thresholded summary.

---

## Results

| model | device | ROC-AUC | log-loss | accuracy | balanced acc | fit s | pred s | total s |
|---|---|---|---|---|---|---|---|---|
| **lgbm_tuned** | cpu | **0.9883 ± 0.0008** | **0.1355 ± 0.0047** | **0.9480** | **0.9464** | 86.040 | 0.056 | 86.095 ± 2.938 |
| tabpfn_v3 | cuda | 0.9858 ± 0.0008 | 0.1519 ± 0.0040 | 0.9390 | 0.9372 | 0.431 | 6.721 | **7.152 ± 0.039** |
| xgb_tuned | cpu | 0.9851 ± 0.0008 | 0.1545 ± 0.0039 | 0.9385 | 0.9367 | 57.564 | 0.016 | 57.580 ± 1.302 |
| xgb_default | cpu | 0.9707 ± 0.0016 | 0.2259 ± 0.0052 | 0.9096 | 0.9071 | 0.096 | 0.001 | 0.098 ± 0.002 |
| lgbm_default | cpu | 0.9553 ± 0.0023 | 0.2833 ± 0.0053 | 0.8831 | 0.8794 | 0.142 | 0.003 | 0.145 ± 0.002 |

Sorted by ROC-AUC. Mean ± std across 10 stratified folds.

### Paired per-fold comparison vs TabPFN-3

Positive = TabPFN better. All models ran identical folds.

| vs tabpfn_v3 | Δ ROC-AUC | sd of Δ | won | p | Δ accuracy | p | Δ log-loss | p |
|---|---|---|---|---|---|---|---|---|
| **lgbm_tuned** | **−0.0025** | 0.0006 | **0/10** | **0.0020** | −0.0090 | 0.0020 | −0.0164 | 0.0020 |
| xgb_tuned | +0.0007 | 0.0010 | 7/10 | **0.0645** | +0.0004 | 0.9219 | +0.0026 | 0.1602 |
| xgb_default | +0.0151 | 0.0016 | 10/10 | 0.0020 | +0.0293 | 0.0020 | +0.0740 | 0.0020 |
| lgbm_default | +0.0305 | 0.0018 | 10/10 | 0.0020 | +0.0558 | 0.0020 | +0.1314 | 0.0020 |

Three clean verdicts, and unusually for this project they all agree across
metrics:

1. **Tuned LightGBM beats TabPFN-3 decisively** — every fold, every metric
   (p = 0.0020 throughout). The margin is small in absolute terms (+0.0025 AUC)
   but the per-fold sd of the difference is only 0.0006, so it is ~4σ and never
   once reverses.
2. **TabPFN-3 ties tuned XGBoost.** p = 0.065 / 0.92 / 0.16 / 0.77 across the four
   metrics; accuracy and balanced accuracy actually split 4/10 to TabPFN. Not a
   difference.
3. **TabPFN-3 beats both defaults 10/10 on every metric** (p = 0.0020).

Final ranking: **lgbm_tuned > tabpfn_v3 ≈ xgb_tuned ≫ xgb_default > lgbm_default**

Note this is the first rung where **the metrics do not disagree**. At 241k and
582k, TabPFN won MAE while losing or tying RMSE, and that split carried the
interpretation. Here ROC-AUC, accuracy, balanced accuracy and log-loss all rank
the models identically, so the conclusion needs no hedging.

---

## Cost

TabPFN's position is far better here than at any large regression rung:

| model | total s | vs TabPFN |
|---|---|---|
| xgb_default | 0.098 | 73× faster |
| lgbm_default | 0.145 | 49× faster |
| **tabpfn_v3** | **7.152** | — |
| xgb_tuned | 57.580 | **8.0× slower** |
| lgbm_tuned | 86.095 | **12.0× slower** |

**The model that beats TabPFN takes 12× longer end-to-end.** TabPFN reaches
within 0.0025 AUC of the best result in 7 seconds with no hyperparameter search
at all, versus 86 seconds of tuning. If tuning budget is the constraint rather
than inference latency, TabPFN is the better default here.

That said, the shape of its cost is unchanged: **fit 0.43s, predict 6.72s**.
Inference is 94% of its wall-clock, and it scales with training-set size — at
40,780 train rows the predict cost is right on the regression curve
(≈n^2 through the 20k regression rung's 2.20s).

### Cost of tuning

| model | default AUC → tuned AUC | gain | fit cost |
|---|---|---|---|
| XGBoost | 0.9707 → 0.9851 | **+0.0145** | 0.096s → 57.6s (600×) |
| LightGBM | 0.9553 → 0.9883 | **+0.0331** | 0.142s → 86.0s (605×) |

LightGBM gained **2.3× more from tuning** than XGBoost — it was the *worst*
model untuned and the *best* tuned. Its defaults (31 leaves, 100 trees) are far
more conservative than XGBoost's, leaving much more headroom. **Judging these two
libraries by their defaults would have ranked them exactly backwards.**

---

## Harness changes made for this rung

Classification support was added properly rather than as a one-off script, so
further classification rungs need only a registry entry:

- **`datasets.py`** — `task` field on `Dataset`; classification targets mapped
  through *sorted* unique values so class 0/1 is stable across runs and machines
  (a drifting label order would silently invert ROC-AUC rather than raise).
- **`models.py`** — five classifier runners reusing `_Timer`, `_inner_split` and
  both search spaces unchanged; tuned variants select on log-loss.
- **`benchmark.py`** — `StratifiedKFold`, `score_classification()`, and a
  separate `results/results_clf.csv`.
- **`compare.py` / `summarize.py`** — `log_loss` added to `LOWER_IS_BETTER`;
  summarize detects task from the columns present.

### Two bugs found during verification

- **`summarize.py` crashed on a column named `head`** — attribute access
  `r.head` resolved to `pandas.Series.head`, the *method*, and failed to format.
  Fixed by using bracket indexing throughout.
- **XGBoost's `predict_proba` rows miss 1.0 by ~3e-8** (float32), while
  LightGBM's are exact. sklearn renormalises internally and warns, so no number
  was ever wrong — but only XGBoost was being renormalised, an asymmetry between
  two models under direct comparison. `score_classification()` now normalises
  every model's probabilities identically.

**Regression path verified intact:** `summarize.py` output for `concrete` and
`nyc_taxi` is byte-identical to `report_1k.md` and `report_581k.md`.
`results/results.csv` was not touched.

---

## Caveats

- **Single dataset.** Conclusions are about `electricity`, not about
  classification in general. The tuned-LightGBM win in particular may not
  transfer — on the 241k regression rung XGBoost tuned better than LightGBM.
- **All models ran on the same node**, unlike the 241k and 582k regression rungs
  where the tuned configs used a separate CPU node. Timing comparisons here are
  therefore cleaner than in those two reports.
- **TabPFN is untuned** by design, and it is being compared against a 30-trial
  search. That is the intended question ("what do you get for free?"), but it is
  not a like-for-like tuning budget.
- **`day` was encoded as a plain numeric** (its labels were numeric strings). No
  model received native categorical handling, keeping the matrix identical for
  all five.
- Folds are local `StratifiedKFold(seed=42)`, not OpenML's predefined task
  splits, because OpenML's metadata API was down (504).
- `best_rounds` / `val_logloss` are still discarded when `--tag` is passed
  (`benchmark.py` writes `notes = args.tag or str(info)`), so we cannot report
  how many trees the searches chose.

---

## Reproduce

```bash
srun --gres=gpu:a100:1 -t 01:30:00 -c 8 --mem=64G bash -c \
  'source scripts/env.sh; $PY src/benchmark.py --dataset electricity \
   --models tabpfn_v3,xgb_default,lgbm_default,xgb_tuned,lgbm_tuned \
   --gbdt-device cpu --tabpfn-device cuda --tag a100-reference'

python src/summarize.py --results results/results_clf.csv --dataset electricity
for m in roc_auc accuracy log_loss balanced_accuracy; do
  python src/compare.py --results results/results_clf.csv \
    --dataset electricity --ref tabpfn_v3 --metric $m
done
```
