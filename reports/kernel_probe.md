# Does the feature count need to scale with dataset size?

**No.** Hypothesis rejected.

RocketPFN fixes 1000 kernels per group (2000 features) for every dataset — from
BeetleFly's 20 training series to ElectricDevices' 8926. The paper ablates G and
feature width but never the kernel count, so the interaction with training-set
size was untested. On small datasets that is a 50–100:1 feature-to-row ratio,
and TabPFN's prior was trained where rows outnumber columns. The obvious guess
was that fewer features would help.

## Result — 5 datasets, resamples 0–4, paired on identical splits

| dataset | n_train | k=10 | k=50 | k=100 | **k=1000** | HC2 |
|---|---|---|---|---|---|---|
| BeetleFly | 20 | 0.8900 | **0.9100** | 0.8900 | 0.9000 | 0.9700 |
| CinCECGTorso | 40 | 0.7754 | 0.8257 | 0.8242 | **0.8442** | 0.9958 |
| FaceFour | 24 | 0.8886 | 0.9341 | 0.9432 | **0.9477** | 0.9955 |
| InsectEPGSmallTrain | 17 | 0.9052 | 0.9237 | 0.9165 | **0.9245** | 1.0000 |
| Rock | 20 | 0.8040 | **0.8360** | 0.8160 | 0.8200 | 0.8960 |
| **mean** | | 0.8526 | 0.8859 | 0.8780 | **0.8873** | 0.9715 |

| vs k=1000 | mean Δ | wins |
|---|---|---|
| k=10 (20 features) | −0.0346 | 0/5 |
| k=50 (100 features) | −0.0014 | 2/5 |
| k=100 (200 features) | −0.0093 | 0/5 |

**Nothing beat the default.** k=10 is clearly worse. k=100 is worse. k=50 is a
statistical tie, winning 2 of 5.

**The negative result is stronger than it looks.** These five datasets were
chosen *because* RocketPFN scored badly on them, which biases toward finding an
improvement — part of a bad score is bad luck, and regression to the mean
flatters any change. The design was tilted in favour of the hypothesis and it
still found nothing.

## What this rules out, and what it points to

The feature-to-row ratio is **not** the mechanism behind the small-data
weakness. 2000 features for 17 training rows is not the problem.

That leaves the other explanation: **inductive bias**. ROCKET features are
random projections carrying no assumption about time series. HC2's members carry
strong hand-built ones — shapelets find discriminative subsequences, intervals
compute statistics over windows, dictionaries count symbolic patterns. With 20
examples a strong prior beats flexible inference, and no amount of tuning the
*quantity* of random features supplies a prior.

Note the scale of what is being explained: on these datasets HC2 averages 0.9715
against our 0.8873. That is a 8.4-point gap, not a tuning gap.

The implied next experiment is to add features that carry a time-series prior —
`catch22`'s 22 curated features, or interval summary statistics — as an extra
group alongside the random ones. For a 20-row dataset, 22 well-chosen features
plausibly beat 2000 random ones.

## One incidental finding worth keeping

**k=50 (100 features) matches k=1000 (2000 features) to within 0.0014.** A
twentieth of the features for the same accuracy. That is not an accuracy result
but it is a cost one, and it is consistent with the MultiRocket+PCA sweep, where
100 selected features tied RocketPFN on datasets large enough for the selector
to work. Two independent routes to the same conclusion: **RocketPFN carries far
more features than it needs.**
