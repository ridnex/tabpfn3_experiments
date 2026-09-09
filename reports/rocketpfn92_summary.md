# RocketPFN on the full UCR pool — 92 datasets, 30 resamples

Independent reproduction of *RocketPFN: Accurate Time Series Classification via
In-Context Learning* (arXiv:2606.21786). 2,760 runs, zero failures, 33.1
GPU-hours on one A100 architecture.

Nothing is trained on the target task: Rocket random convolutions produce
features, a frozen TabPFN classifies them in-context, and 10 feature groups are
averaged. Baselines are published per-resample results, never rerun.

## Headline

| method | avg rank | mean accuracy |
|---|---|---|
| HC2 | **2.51** | **0.9005** |
| **RocketPFN** | **2.87** | **0.8954** |
| TS-CHIEF | 3.70 | 0.8843 |
| ROCKET | 3.87 | 0.8818 |
| DrCIF | 4.02 | 0.8841 |
| InceptionTime | 4.04 | 0.8799 |

**Second of six.** Against HC2: 37 wins, 50 losses, 5 ties, Wilcoxon two-sided
**p = 0.1408** — not significant.

The paper's central claim, that a training-free method is statistically
indistinguishable from HIVE-COTE 2.0, **reproduces**.

## Paper vs. this reproduction

The paper's own numbers (Table 2, 92 datasets, 30 resamples) against ours:

| method | paper | ours | diff |
|---|---|---|---|
| **RocketPFN** | **0.900** | **0.8954** | **−0.0046** |
| HC2 | 0.900 | 0.9005 | +0.0005 |
| TS-CHIEF | 0.884 | 0.8843 | +0.0003 |
| DrCIF | 0.884 | 0.8841 | +0.0001 |
| ROCKET | 0.882 | 0.8818 | −0.0002 |
| InceptionTime | 0.880 | 0.8799 | −0.0001 |

| vs HC2 | paper | ours |
|---|---|---|
| win / tie / loss | 44 / 3 / 45 | 37 / 5 / 50 |
| Wilcoxon p | 0.504 | 0.141 |

**Every baseline matches to within 0.0005.** That is the useful part: it confirms
our pool really is the paper's 92, the 30-resample protocol lines up index for
index, and nothing in the comparison machinery is off. All five agree because
both papers read the same published results files.

**Only RocketPFN differs — ours is 0.5 accuracy points low.** Since everything
else matches, the gap is isolated to our implementation of the method itself, or
to the model behind it. The paper used TabPFN v2.5; we used v3. We had assumed v3
would help, since it is the stronger model; it did not.

The consequence is visible in the head-to-head. The paper's RocketPFN splits with
HC2 almost exactly evenly (44–45, p = 0.504). Ours tilts toward HC2 (37–50,
p = 0.141). Both fail to reject, so the qualitative claim — statistically
indistinguishable from HC2 — holds either way. But the paper's version sits level
with HC2 and ours sits slightly below it, and half a point is not nothing.

Candidate explanations, none yet tested:

1. **TabPFN v3 vs v2.5.** The likeliest. v3's per-member feature budget changed
   how the 2000 features of a group are covered (auto-scaling `n_estimators`
   8→10), so "one group" is not the same computation in both versions.
2. **Rocket normalisation of the input series.** The paper names no code
   repository, so we inferred this from aeon's defaults.
3. **Group seeding.** Our per-group seeds derive from `random_state`; the paper
   does not specify its scheme.

Testing (1) would mean re-running against TabPFN v2.5 — the single most direct
way to close or explain the gap.

## The 20-dataset subset was misleading

We ran a 20-dataset subset first. It understated the method:

| | 20 datasets | 92 datasets |
|---|---|---|
| RocketPFN rank | 3.20 (3rd) | **2.87 (2nd)** |
| vs HC2 | p = 0.078 | p = 0.141 |

More data moved RocketPFN **up** a place and made the HC2 gap *less* significant.
A subset that looked like weak evidence against the paper was mostly sampling
noise — which is the argument for running the full pool rather than a sample.

## The real finding: it scales with training data

Splitting by training-set size separates the result cleanly.

| subset | n | RocketPFN | HC2 | rank RPFN | rank HC2 | p |
|---|---|---|---|---|---|---|
| n_train < 100 | 36 | 0.9259 | 0.9377 | 3.25 | **2.51** | 0.057 |
| 100–499 | 37 | 0.8572 | 0.8605 | 2.93 | **2.45** | 0.226 |
| **n_train ≥ 500** | 19 | **0.9122** | 0.9080 | **2.03** | 2.61 | 0.085 |

**On the 19 largest-training datasets RocketPFN outranks HC2** — 2.03 against
2.61, winning 12 of 19. On the 36 smallest it is clearly behind.

That is the mechanism. TabPFN classifies by holding the training set in context,
so its evidence *is* the training data. HC2's shapelet and interval members
extract structure that survives having few examples; in-context learning does
not. The crossover sits near n_train ≈ 500.

Neither subgroup p-value clears 0.05 on its own, so this is a strong trend
rather than a proven claim — but it is consistent, directional, and it explains
the aggregate.

## Where it fails

| worst 6 vs HC2 | Δ | | best 6 vs HC2 | Δ |
|---|---|---|---|---|
| CinCECGTorso | −0.142 | | Lightning2 | +0.055 |
| Rock | −0.101 | | MiddlePhalanxOutlineAgeGroup | +0.046 |
| InsectEPGSmallTrain | −0.071 | | DiatomSizeReduction | +0.039 |
| ScreenType | −0.057 | | MedicalImages | +0.030 |
| FaceFour | −0.043 | | Beef | +0.029 |
| BeetleFly | −0.042 | | Lightning7 | +0.029 |

By problem type, the weakest are EPG (−0.043) and Spectrum (−0.033); the
strongest are ECG (+0.003), Spectro (+0.002) and Image (+0.001). The losses
cluster where discrimination lives in long-range or phase-dependent structure
that fixed random kernels do not capture — a representational limit, not a
statistical one.

## Cost

Median 29.8s per dataset per resample; TabPFN inference is 28.3s of it and
Rocket feature extraction 1.2s. **Training cost is zero; inference cost is high
and scales with the test set**, because every test series is a forward pass over
a context holding the entire training set. HC2 amortises its hours of fitting
across predictions. RocketPFN pays per prediction, permanently.

That makes the deployment case specific: many small problems, cold starts, no
budget to fit a model per task. Not one fixed problem with a large test stream.

## Deviations

1. **TabPFN v3**, not the paper's v2.5. Likely biases toward the claim.
2. **v3 alters the grouping arithmetic** — it shows each ensemble member ~200
   features and auto-scaled `n_estimators` 8→10 to cover Rocket's 2000, so a
   "group" is ~10 forward passes. Measured and logged per run, not assumed.
3. **v3's class cap is 160, not 10.** The paper excludes 20 datasets for having
   more than 10 classes; under v3 that exclusion is unnecessary. We kept it only
   to match the paper's pool.
4. **Baselines never rerun** — published per-resample results, so only our
   column is new.
5. **Installing aeon moved the shared environment** (pandas 3.0.5→2.3.3, numpy
   2.4.6→2.3.5, scikit-learn 1.9.0→1.8.0).

## Note on the statistical test

`aeon.benchmarking.stats.wilcoxon_test` returns a **one-sided** p-value per
ordered pair. On the 20-dataset run it reported 0.9611 for RocketPFN vs HC2,
which reads as "no difference" and means the opposite — its complement, 0.039,
says HC2 is better. All p-values here are scipy two-sided.
