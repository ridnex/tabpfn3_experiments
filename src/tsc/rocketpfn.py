"""RocketPFN: Rocket features + TabPFN as a frozen in-context classifier.

Follows arXiv:2606.21786. Nothing is trained on the target task: Rocket's
kernels are random, and TabPFN's weights are frozen - the training set enters
as context, the way examples enter an LLM prompt.

    G independent groups of k=1000 random kernels
      -> 2k = 2000 features per group  (max pooling + proportion of positives)
      -> one TabPFN forward pass per group
      -> average the per-group class probabilities, then argmax

The grouping is not cosmetic. TabPFN caps input at 2000 features, so the paper
splits Rocket's usual 10,000-kernel budget into 10 groups of 1000 rather than
truncating it. Averaging probabilities (not votes) keeps the confidence
information each group produces.

DEVIATION FROM THE PAPER, worth knowing before reading any accuracy number:
the paper used TabPFN v2.5, this uses v3. v3 caps each ENSEMBLE MEMBER at 200
features and auto-scales its member count to cover all 2000, so a group is
~10 forward passes rather than one. Same features reach the model, but through
internal subsampling rather than in a single pass, and the cost per group is
roughly 10x. `n_estimators_` is recorded per run so this stays visible.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

N_GROUPS = 10
N_KERNELS = 1000


class RocketPFNClassifier(ClassifierMixin, BaseEstimator):
    """Training-free time series classifier.

    Parameters
    ----------
    n_groups : int
        Independent Rocket feature groups, each classified separately. G=10
        reproduces Rocket's default 10,000-kernel budget.
    n_kernels : int
        Kernels per group. 1000 kernels -> 2000 features, TabPFN's input cap.
    device : str
        Passed to TabPFN. "cuda" is the reference; "cpu" works but is slow.
    random_state : int
        Seeds every group deterministically. Two runs with the same value give
        bit-identical probabilities.
    n_jobs : int
        Threads for Rocket's kernel convolutions (numba). TabPFN is unaffected.

    Attributes
    ----------
    timings_ : dict
        Seconds in feature extraction vs TabPFN, kept separate because the two
        scale with different things - Rocket with series length, TabPFN with
        training-set size.
    n_estimators_ : list[int]
        TabPFN ensemble members actually used per group. Not a constant under
        v3's auto-scaling, and it drives the runtime, so it is recorded rather
        than assumed.
    """

    def __init__(
        self,
        n_groups: int = N_GROUPS,
        n_kernels: int = N_KERNELS,
        device: str = "cuda",
        random_state: int = 0,
        n_jobs: int = 1,
    ):
        self.n_groups = n_groups
        self.n_kernels = n_kernels
        self.device = device
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Store the context. No parameters are learned here or anywhere."""
        X = np.asarray(X)
        if X.ndim == 2:  # (n, length) -> (n, 1, length), aeon's channel-first layout
            X = X[:, None, :]
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self._X_train = X
        self._y_train = y_idx
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "_X_train")
        from aeon.transformations.collection.convolution_based import Rocket
        from tabpfn import TabPFNClassifier

        X = np.asarray(X)
        if X.ndim == 2:
            X = X[:, None, :]

        t_feat = t_pfn = 0.0
        self.n_estimators_ = []
        proba = np.zeros((len(X), len(self.classes_)), dtype="float64")

        for g in range(self.n_groups):
            # Derived per group rather than reusing random_state, so groups are
            # genuinely independent draws and the run is still reproducible.
            seed = self.random_state * 1000 + g

            t0 = time.perf_counter()
            rocket = Rocket(
                n_kernels=self.n_kernels, random_state=seed, n_jobs=self.n_jobs
            )
            # Kernels are random, so fitting on train leaks nothing - but the
            # normalisation statistics are per-series, and fit/transform must be
            # the same object or train and test features are not comparable.
            Ztr = rocket.fit_transform(self._X_train)
            Zte = rocket.transform(X)
            t_feat += time.perf_counter() - t0

            t0 = time.perf_counter()
            clf = TabPFNClassifier(device=self.device, random_state=seed)
            clf.fit(Ztr, self._y_train)
            p = clf.predict_proba(Zte)
            t_pfn += time.perf_counter() - t0
            self.n_estimators_.append(int(clf.n_estimators_))

            # Averaged, not voted: a group that is 51/49 should not count the
            # same as one that is 99/1.
            proba += p

        proba /= self.n_groups
        self.timings_ = {"feature_time": t_feat, "tabpfn_time": t_pfn}
        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
