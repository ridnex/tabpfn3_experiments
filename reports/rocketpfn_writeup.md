# RocketPFN on UCR: what we found

An independent reproduction of *RocketPFN: Accurate Time Series Classification via
In-Context Learning* (arXiv:2606.21786) on 20 UCR datasets, 30 resamples, against
published baselines. Numbers: [`rocketpfn.md`](rocketpfn.md). This file is the reading
of them.

## The claim and the verdict

The paper's claim is that a **training-free** classifier — Rocket random convolutional
features fed to a frozen TabPFN as in-context examples — reaches 0.900 mean accuracy on
92 UCR datasets and is statistically indistinguishable from HIVE-COTE 2.0, the strongest
published method on the archive.

On our 20-dataset subset the claim survives, but in its weaker form.

| method | avg rank | mean accuracy |
|---|---|---|
| HC2 | **2.30** | **0.9047** |
| TS-CHIEF | 3.15 | 0.8893 |
| **RocketPFN** | **3.20** | **0.8927** |
| DrCIF | 3.85 | 0.8832 |
| ROCKET | 3.95 | 0.8853 |
| InceptionTime | 4.55 | 0.8722 |

Third of six. It beats ROCKET — its own feature extractor with a trained ridge head —
and it beats DrCIF and InceptionTime, both of which train on the target task. It does
not beat HC2: 5 wins, 13 losses, 2 ties, Wilcoxon two-sided p = 0.078.

**"Statistically indistinguishable" is doing real work in that sentence.** p = 0.078 is
a failure to reject, not evidence of equality, and the win/loss split of 5–13 points one
way. The honest summary is: *RocketPFN loses to HC2 by about one accuracy point, and 20
datasets is not enough to call that loss significant.* The paper's 92 datasets would
have roughly twice our power; whether the same gap survives there is exactly the
question our subset cannot answer.

## What is genuinely surprising

Nothing is trained. There is no gradient step, no hyperparameter search, no
cross-validation on the target task. The training series are pushed through Rocket, the
resulting 2000 features are handed to TabPFN as context rows, and the answer comes out
of a single forward pass. That this lands within a point of a method built from four
ensembled representations, each internally tuned, is the actual result — and it
reproduced here without any tuning on our side.

The comparison against plain ROCKET is the cleanest evidence. Same features, same
kernels, same seeds in spirit; the only difference is ridge regression versus in-context
TabPFN. Rank 3.95 → 3.20. The PFN head is worth something on its own.

## Where it fails, and why that pattern matters

The losses are not uniform. Three datasets carry most of the gap:

| dataset | RocketPFN | HC2 | gap |
|---|---|---|---|
| Rock | 0.776 | 0.877 | −0.101 |
| ScreenType | 0.697 | 0.754 | −0.057 |
| SmoothSubspace | 0.957 | 0.980 | −0.023 |

`Rock` is 20 training series of length 2844 — a 4-class problem with five examples per
class. `ScreenType` is a device-usage problem where discrimination lives in long-range
periodic structure. Both are cases where a fixed random feature bank is a poor basis and
where HC2's interval and shapelet members earn their keep. The failure mode is
*representational*, not statistical: no amount of in-context cleverness recovers
information the kernels never extracted.

Conversely it wins on `ACSF1`, `PhalangesOutlinesCorrect` and both `UWaveGestureLibrary`
channels — larger training sets where TabPFN's context has enough rows to work with.

## Cost, which is the part nobody advertises

Median 30.5s per dataset per resample; 7.5 GPU-hours for the full 20 × 30 sweep on one
A100. Rocket feature extraction is 11% of that. The other 89% is TabPFN inference.

The cost structure is inverted relative to every baseline here:

- **Training cost: zero.** HC2 on this subset takes hours of CPU per dataset to fit.
- **Inference cost: high, and it scales with the test set, not the training set.**

Look at the ordering: `PhalangesOutlinesCorrect` (1800 train) costs 62s while
`Rock` (20 train, length 2844) costs 30s. Every test series is a forward pass through a
transformer carrying the whole training set as context. A trained classifier amortises;
this one pays per prediction, forever.

That makes the deployment story specific rather than universal. RocketPFN is the right
tool when you have many small classification problems and cannot afford to fit a model
for each — a new sensor type every week, a cold-start problem, an interactive setting.
It is the wrong tool for one fixed problem with a large test stream, where HC2's or even
ROCKET's one-time fit is quickly repaid.

## Deviations from the paper

These are real and should be weighed before treating our numbers as a clean replication.

1. **TabPFN v3, not v2.5.** The paper's model. We used what is installed and current.
   v3 is generally stronger, so if anything this biases *toward* the paper's claim.
2. **20 datasets, not 92.** Stratified by UCR problem type, seed 42, fixed in
   `configs/ucr20.json` and committed before a single result existed — deliberately, so
   the selection cannot be read as cherry-picking. But it costs statistical power, which
   is precisely why the Wilcoxon comes back inconclusive.
3. **v3 changes the grouping arithmetic.** It shows each ensemble member ~200 features
   and auto-scaled `n_estimators` from 8 to 10 to cover Rocket's 2000 — measured, not
   assumed, and logged per run. So one "group" is ~10 forward passes, not one. The
   paper's G=10 grouping exists to fit under v2.5's feature cap; under v3 that constraint
   is handled internally and the two designs are not quite the same computation.
4. **v3's class cap is 160, not 10.** The paper excludes 20 of the archive's datasets for
   having more than 10 classes. Under v3 that exclusion is unnecessary — those datasets
   are runnable and were left out here only to keep the pool identical to the paper's.
5. **Baselines were never rerun.** HC2 and the rest are the published per-resample
   results. Only our column is new, which removes a whole class of "did you tune the
   baseline fairly" objection.
6. **Installing aeon moved the shared environment** (pandas 3.0.5→2.3.3, numpy
   2.4.6→2.3.5, scikit-learn 1.9.0→1.8.0). The repo's earlier committed regression
   results were produced under the older pins.

## One methodological note

aeon's `benchmarking.stats.wilcoxon_test` returns a **one-sided** p-value per ordered
pair. It reported 0.9611 for RocketPFN vs HC2, which reads as "no difference whatsoever"
and is the opposite of what it means — the complement, 0.039, says HC2 is better. The
analysis now uses scipy's two-sided test (p = 0.0778). Worth knowing for anyone else
using that function to make a "no significant difference" claim.

## What we would do next

- **Extend to all 92 datasets.** The single highest-value follow-up: our inconclusive
  Wilcoxon is a power problem, and 92 datasets would resolve it either way. Cost is
  roughly 35 GPU-hours by extrapolation.
- **Include the 20 excluded many-class datasets.** v3 allows it; the paper's exclusion
  is an artefact of v2.5 and dropping it would be a genuine extension rather than a
  replication.
- **Ablate G.** G is a free knob and we have never measured the accuracy/cost curve
  beyond confirming G=10 is not worse than G=1.
- **Test the cold-start premise directly.** The method's real selling point is
  zero training cost, and no experiment here measures it. A subsample of training
  series would show whether RocketPFN degrades more gracefully than HC2 — which, if
  true, is a more useful claim than parity on the full archive.
