"""Feature selectors for the fixed-budget experiment, as a 2x2 factorial.

TabPFN shows each ensemble member ~200 features, so a budget of 100 out of a
50000-feature Rocket pool is a hard constraint rather than a convenience. The
hypothesis under test is that under such a budget REDUNDANCY, not relevance, is
the binding constraint: a duplicated feature wastes a slot, which costs nothing
for ridge regression over all 50000 columns but is expensive here.

The four arms isolate the two factors:

                relevance   redundancy-aware   supervised
    random          no            no               no
    mutual_info     yes           no               yes
    mrmr            yes           yes              yes
    pca             no            yes              no

random vs mutual_info isolates relevance; mutual_info vs mrmr isolates
redundancy (same MI scores, the ONLY difference is the penalty); pca vs mrmr
isolates supervision. PCA alone would confound the last two, which is why mrmr
is in Stage 1 rather than deferred.
"""
from __future__ import annotations

import numpy as np

# 10x the budget. Greedy mRMR is O(k * p * n) and p is ~50000, so scoring every
# candidate at all 100 steps is billions of operations per dataset-resample.
# Restricting candidates to the top 1000 by MI is the standard fix and costs
# almost nothing: mRMR maximises relevance MINUS redundancy, so a feature with
# near-zero relevance can only win a slot if its redundancy term is strongly
# negative, which correlation-based redundancy does not produce. The constraint
# pulls mRMR TOWARD MI - i.e. against our own hypothesis, the safe direction.
MRMR_PREFILTER = 1000


def _mi(Z, y, seed, n_jobs):
    from sklearn.feature_selection import mutual_info_classif

    return mutual_info_classif(Z, y, random_state=seed, n_jobs=n_jobs)


def select(kind, Ztr, ytr, Zte, top_n, seed, n_jobs, mi=None):
    """Return (train, test, n_used, info). Fitted on TRAIN ONLY.

    `mi` is passed in so the MI scores are computed once per dataset-resample
    and shared by the mutual_info and mrmr arms - it is the expensive part.
    """
    rng = np.random.default_rng(seed)
    p = Ztr.shape[1]

    if kind == "random":
        idx = rng.choice(p, size=min(top_n, p), replace=False)
        return Ztr[:, idx], Zte[:, idx], len(idx), {}

    if kind == "mutual_info":
        idx = np.argsort(mi)[-top_n:]
        return Ztr[:, idx], Zte[:, idx], len(idx), {}

    if kind == "mrmr":
        cand = np.argsort(mi)[-min(MRMR_PREFILTER, p):]
        Zc, mic = Ztr[:, cand], mi[cand]
        # Rocket can emit constant columns (a PPV feature that is always 0),
        # which make corrcoef produce nan. Zero is the right fill: a constant
        # feature is uninformative, not maximally redundant.
        C = np.nan_to_num(np.abs(np.corrcoef(Zc, rowvar=False)))
        sel = [int(np.argmax(mic))]
        red = C[:, sel[0]].copy()
        while len(sel) < min(top_n, len(cand)):
            score = mic - red / len(sel)
            score[sel] = -np.inf
            nxt = int(np.argmax(score))
            sel.append(nxt)
            red += C[:, nxt]
        idx = cand[sel]
        return Ztr[:, idx], Zte[:, idx], len(idx), {}

    if kind == "pca":
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Capped at min(n_samples, n_features). The medium-dataset config keeps
        # n_train >= 120 precisely so this never binds and all four arms get an
        # equal budget - a capped PCA would confound the comparison.
        k = min(top_n, Ztr.shape[0], p)
        sc = StandardScaler().fit(Ztr)
        pca = PCA(n_components=k, random_state=seed).fit(sc.transform(Ztr))
        return (pca.transform(sc.transform(Ztr)),
                pca.transform(sc.transform(Zte)), k, {})

    raise ValueError(kind)


def mediators(Str, ytr, seed, n_jobs):
    """Redundancy and relevance of a selected set - the mechanism, not the score.

    Reporting only accuracy would make this a horse race. These two numbers are
    what turn "PCA won" into a claim that generalises: if accuracy tracks
    redundancy after controlling for relevance, the budget argument holds for
    any fixed-context learner, not just this pipeline.
    """
    C = np.nan_to_num(np.abs(np.corrcoef(Str, rowvar=False)))
    off = ~np.eye(C.shape[0], dtype=bool)
    return {
        "redundancy": float(C[off].mean()) if C.shape[0] > 1 else 0.0,
        "relevance": float(_mi(Str, ytr, seed, n_jobs).mean()),
    }
