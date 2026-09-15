"""Plain TabPFN on raw UCR series: one time step = one column, no encoding.

The control arm for RocketPFN (arXiv:2606.21786). The paper's Appendix C puts
plain TabPFN v3 on flattened series at 0.844 mean accuracy; this reproduces
that measurement with our own runner.

Deliberately untouched input: no z-normalisation, truncation or downsampling on
our side. TabPFN's own preprocessing is the only transform, so the number is
"what the model does with the raw series", not "what our pipeline does".

Library defaults throughout. v3 caps each ensemble member at a feature budget
and auto-scales the member count so every time step is seen at least once;
`n_estimators_` is recorded because it drives runtime and varies with length.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class FlatPFNClassifier(ClassifierMixin, BaseEstimator):
    """TabPFN fitted directly on the flattened series.

    Parameters
    ----------
    device : str
        Passed to TabPFN.
    random_state : int
        Passed to TabPFN.
    n_jobs : int
        Unused; accepted so run.py can build either classifier the same way.

    Attributes
    ----------
    timings_ : dict
        Same keys as RocketPFNClassifier; feature_time is always 0.
    n_estimators_ : list[int]
        One entry (a single TabPFN), a list to match RocketPFNClassifier.
    """

    def __init__(self, device: str = "cuda", random_state: int = 0, n_jobs: int = 1):
        self.device = device
        self.random_state = random_state
        self.n_jobs = n_jobs

    @staticmethod
    def _flatten(X):
        X = np.asarray(X)
        if X.ndim == 3:
            if X.shape[1] != 1:
                raise ValueError(f"univariate only, got {X.shape[1]} channels")
            X = X[:, 0, :]
        return X

    def fit(self, X, y):
        from tabpfn import TabPFNClassifier

        self.classes_, y_idx = np.unique(y, return_inverse=True)
        t0 = time.perf_counter()
        self._clf = TabPFNClassifier(device=self.device, random_state=self.random_state)
        self._clf.fit(self._flatten(X), y_idx)
        self._fit_time = time.perf_counter() - t0
        self.n_estimators_ = [int(self._clf.n_estimators_)]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "_clf")
        t0 = time.perf_counter()
        proba = self._clf.predict_proba(self._flatten(X))
        self.timings_ = {
            "feature_time": 0.0,
            "tabpfn_time": self._fit_time + time.perf_counter() - t0,
        }
        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
