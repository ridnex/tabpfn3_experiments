# Where RocketPFN beats HC2, and where it doesn't

92 datasets, 30 resamples. **37 wins, 50 losses, 5 ties.**

## What predicts the outcome

Spearman correlation of (RocketPFN − HC2) with dataset properties:

| property | rho | p |
|---|---|---|
| **training set size** | **+0.274** | **0.008** |
| series length | **−0.265** | **0.011** |
| test/train ratio | **−0.255** | **0.014** |
| **test set size** | +0.014 | 0.895 |

Three significant effects and one clean null.

**Test set size predicts nothing at all** (p = 0.90). That matters, because the
test/train ratio *does* correlate — so the ratio effect is not about having a
large test set, it is the training-size effect wearing a different hat. More
labelled training data helps; how much unlabelled test data sits beside it is
irrelevant to accuracy.

**Series length is as strong a predictor as training size**, and negative. Long
series hurt, independently of how many you have.

## Win rate by test/train ratio

| ratio | n | mean Δ | wins |
|---|---|---|---|
| test < train | 19 | **+0.0071** | **13/19 (68%)** |
| 1–5× | 48 | −0.0071 | 17/48 (35%) |
| ≥5× | 25 | −0.0105 | 7/25 (28%) |

When training data outnumbers test data, RocketPFN wins two thirds of the time.
When the test set is 5× larger, it wins barely a quarter.

## The 10 biggest wins

| dataset | train | test | length | RocketPFN | HC2 | Δ |
|---|---|---|---|---|---|---|
| Lightning2 | 60 | 61 | 637 | 0.8251 | 0.7699 | +0.0552 |
| MiddlePhalanxOutlineAgeGroup | 400 | 154 | 80 | 0.7348 | 0.6894 | +0.0455 |
| DiatomSizeReduction | 16 | 306 | 345 | 0.9678 | 0.9291 | +0.0387 |
| MedicalImages | 381 | 760 | 99 | 0.8414 | 0.8111 | +0.0303 |
| Beef | 30 | 30 | 470 | 0.7500 | 0.7211 | +0.0289 |
| Lightning7 | 70 | 73 | 319 | 0.8219 | 0.7932 | +0.0288 |
| PhalangesOutlinesCorrect | 1800 | 858 | 80 | 0.8683 | 0.8430 | +0.0253 |
| MiddlePhalanxTW | 399 | 154 | 80 | 0.5935 | 0.5699 | +0.0236 |
| ACSF1 | 100 | 100 | 1460 | 0.8667 | 0.8480 | +0.0187 |
| MiddlePhalanxOutlineCorrect | 600 | 291 | 80 | 0.8491 | 0.8306 | +0.0186 |

The Phalanx family (5 of the top 10) shares a profile: short series (length 80),
several hundred training examples, and problems where HC2 itself scores poorly —
MiddlePhalanxTW at 0.57, MiddlePhalanxOutlineAgeGroup at 0.69. RocketPFN does
best on **hard, short, data-rich** problems.

## The 10 biggest losses

| dataset | train | test | length | RocketPFN | HC2 | Δ |
|---|---|---|---|---|---|---|
| CinCECGTorso | 40 | 1380 | 1639 | 0.8510 | 0.9930 | −0.1420 |
| Rock | 20 | 50 | 2844 | 0.7760 | 0.8773 | −0.1013 |
| InsectEPGSmallTrain | 17 | 249 | 601 | 0.9289 | 1.0000 | −0.0711 |
| ScreenType | 375 | 375 | 720 | 0.6968 | 0.7540 | −0.0572 |
| FaceFour | 24 | 88 | 350 | 0.9364 | 0.9792 | −0.0428 |
| BeetleFly | 20 | 20 | 512 | 0.8817 | 0.9233 | −0.0417 |
| ToeSegmentation2 | 36 | 130 | 343 | 0.9262 | 0.9638 | −0.0377 |
| BirdChicken | 20 | 20 | 512 | 0.9150 | 0.9483 | −0.0333 |
| Ham | 109 | 105 | 431 | 0.8289 | 0.8597 | −0.0308 |
| RefrigerationDevices | 375 | 375 | 720 | 0.7864 | 0.8168 | −0.0304 |

The top three are extreme on both axes at once: 17–40 training series **and**
601–2844 points per series. Rock is 20 series of length 2844 — 142 time points
per training example.

The losses are also more severe than the wins: the worst loss is −0.142, the
best win +0.055. The distribution is asymmetric, which is why the method can
lose 50 of 92 head-to-head while sitting only 0.005 behind on mean accuracy.

## Reading

The two failure axes are **few training examples** and **long series**, and they
compound. Both point the same way: a random convolutional feature bank needs
data to average its noise out, and a longer series gives each random kernel more
room to fire on something irrelevant. HC2's shapelet and interval members impose
structure instead of averaging over randomness, which is exactly the right trade
when examples are scarce and series are long.

The `ScreenType` / `RefrigerationDevices` losses do not fit that story — 375
training series, length 720, and still a 3–6 point loss. Both are appliance
power-consumption problems where the discriminating signal is long-range and
periodic. That is a third, representational failure mode, distinct from the
data-scarcity one.
