# MultiRocket + PCA-100 + TabPFN — 20 datasets, 30 resamples

Tests whether *selecting* 100 features beats RocketPFN's *grouping* (10 passes
over 2000 features) as a way to fit inside TabPFN's per-member feature budget.

**It does not.** The idea looked strong on a single dataset and did not survive
the full protocol.

## Average rank (lower is better)

| method | avg rank | mean accuracy |
|---|---|---|
| HC2 | 2.45 | 0.9047 |
| TS-CHIEF | 3.35 | 0.8893 |
| RocketPFN | 3.45 | 0.8927 |
| ROCKET | 4.25 | 0.8853 |
| DrCIF | 4.30 | 0.8832 |
| **MultiRocket+PCA** | **5.05** | **0.8580** |
| InceptionTime | 5.15 | 0.8722 |

Second from last. Against RocketPFN: 5 wins, 15 losses, Wilcoxon p = 0.0107 —
a significant loss, not a wash.

## Where it fails: PCA's component ceiling

PCA cannot return more components than training rows. Six of the 20 datasets
have fewer than 100 training series, so they got fewer than 100 components
however large `top_n` was set. Splitting on that flag explains the whole result:

| subset | n | MR+PCA | RocketPFN | diff | W/L | p |
|---|---|---|---|---|---|---|
| all | 20 | 0.8580 | 0.8927 | −0.0347 | 5/15 | 0.0107 |
| **uncapped** | 14 | 0.8649 | 0.8793 | −0.0144 | 5/9 | 0.2166 |
| **capped** | 6 | 0.8419 | 0.9241 | −0.0822 | 0/6 | 0.0312 |

On the capped datasets it loses **every single one**. BeetleFly (20 training
series → 20 components) drops from 0.8817 to 0.6083; Rock from 0.776 to 0.652.

On datasets where PCA can actually produce 100 components it is roughly a tie —
1.4 accuracy points behind, not statistically separable (p = 0.22).

So the honest reading: **selection is competitive with grouping only when there
is enough training data to build the projection, and collapses when there is
not.** Grouping has no such dependency, which is a real structural advantage
rather than an incidental one.

## The cost side

10.5s per dataset per resample against RocketPFN's 30.5s — roughly 3x cheaper,
and the GPU share is ~1-2s because TabPFN sees 100 features in one forward pass
instead of 2000 across ten. On the 14 uncapped datasets that is a genuine
trade: a third of the cost for ~1.4 points, inside the noise.

## A methodological note on the probe

The single-dataset probe picked MultiRocket+PCA because it scored 0.9886 on
Fish, ahead of everything including HC2.

Fish turned out to be one of only 5 datasets out of 20 where it beats
RocketPFN, and it remains a win (0.9789 vs 0.9642) — but it is unrepresentative,
and the probe's headline number does not generalise at all.

Two biases were flagged before the run and both landed:

1. **Fish was chosen after seeing the 20-dataset RocketPFN results**, partly
   because RocketPFN underperformed there. That is the wrong direction to pick
   from - it is where a challenger flatters itself.
2. **Best-of-6 on one split is a winner's curse.** 0.9886 was the maximum of six
   noisy draws on 175 test series, so regression was the expected outcome.

The probe was still worth running: it established that the pipeline works and
that mutual information costs seconds rather than the minutes estimated. It just
could never have supported an accuracy claim, and did not.

## What would be worth testing next

- **ROCKET + PCA**, the clean one-variable ablation. This run changed both the
  transformer and the strategy, so it cannot say which half caused the loss.
- **A larger `top_n` on the uncapped datasets.** 100 features may simply be too
  few; TabPFN accepts 200 per ensemble member.
- **Selection as a cheap first stage rather than a replacement** - fewer groups
  over a pre-filtered pool, keeping grouping's robustness at lower cost.
