"""Model registry: TabPFN-3 + XGBoost/LightGBM at defaults and lightly tuned.

Timing contract (decided up front, see README):
  * fit_time     - everything needed to go from raw train data to a usable model.
                   For the *tuned* GBDT configs this deliberately INCLUDES the
                   30-trial random search, because that search is a real cost you
                   pay to get that accuracy.
  * predict_time - producing predictions for the whole test fold in one batched
                   call (never row-by-row; TabPFN docs warn that per-row predict
                   is ~100x slower and it would not be a like-for-like number).
  * TabPFN inverts the usual shape: fit() only stores/preprocesses the training
    rows, and the actual forward pass over [train + test] happens in predict().
    That asymmetry is the point, not a bug.

GPU work is synchronised before stopping any timer, otherwise CUDA's async
dispatch would report near-zero times.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np


def _tuned(fn: Callable, ordered: bool) -> Callable:
    """Bind the inner-split mode onto a tuned runner.

    Runner.run() calls fn positionally, so `ordered` has to be bound here rather
    than threaded through the call site.
    """
    return partial(fn, ordered=ordered)

N_TRIALS = 30
EARLY_STOPPING_ROUNDS = 50
MAX_ROUNDS = 2000
INNER_VAL_FRAC = 0.2


def _sync(device: str) -> None:
    """Block until queued GPU work is done, so timers measure real elapsed work."""
    if device == "cuda":
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()


class _Timer:
    def __init__(self, device: str):
        self.device = device

    def __enter__(self):
        _sync(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        _sync(self.device)
        self.elapsed = time.perf_counter() - self.t0
        return False


# --------------------------------------------------------------------------
# search spaces
# --------------------------------------------------------------------------
def _sample_xgb(rng: np.random.RandomState) -> dict:
    return {
        "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.3)))),
        "max_depth": int(rng.randint(3, 11)),
        "subsample": float(rng.uniform(0.5, 1.0)),
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "min_child_weight": float(np.exp(rng.uniform(np.log(1), np.log(20)))),
        "reg_lambda": float(np.exp(rng.uniform(np.log(1e-3), np.log(10)))),
        "reg_alpha": float(np.exp(rng.uniform(np.log(1e-3), np.log(10)))),
    }


def _sample_lgbm(rng: np.random.RandomState) -> dict:
    return {
        "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.3)))),
        "num_leaves": int(np.exp(rng.uniform(np.log(8), np.log(256)))),
        "min_child_samples": int(rng.randint(5, 101)),
        "subsample": float(rng.uniform(0.5, 1.0)),
        "subsample_freq": 1,
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "reg_lambda": float(np.exp(rng.uniform(np.log(1e-3), np.log(10)))),
        "reg_alpha": float(np.exp(rng.uniform(np.log(1e-3), np.log(10)))),
    }


def _inner_split(n: int, seed: int, ordered: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Carve a validation set out of the TRAIN fold only - never touches test.

    ordered=True holds out the LAST 20% in time instead of a random 20%, and is
    required whenever the outer split is time-based.

    A random inner split is exactly as leaky as a random outer split: every
    validation row's 10-minute neighbour sits in the inner training set, so the
    search scores candidates on how well they copy a neighbour. Measured on
    ke02_well - the random inner split reported val_rmse of 247-365 while those
    same selected models scored 1,541-6,047 on the real future test block, and it
    drove LightGBM to 920-1,994 trees when its default 100 generalised better.
    Tuning then makes the model worse, which reads as a fact about LightGBM but
    is a fact about the validation split.
    """
    n_val = max(1, int(round(n * INNER_VAL_FRAC)))
    if ordered:
        # Train on the earlier rows, validate on the later ones - the same shape
        # as the outer forward-chaining split, one level down.
        idx = np.arange(n)
        return idx[:-n_val], idx[-n_val:]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    return idx[n_val:], idx[:n_val]


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _logloss(y_true, proba) -> float:
    """Inner-validation score for the classification searches.

    Log-loss rather than accuracy: it is a proper scoring rule, so the search
    optimises calibrated probabilities, which is what ROC-AUC and log-loss are
    both read off. Tuning on accuracy would optimise a thresholded summary and
    leave the probabilities worse.
    """
    from sklearn.metrics import log_loss

    # labels= pins the column order: a trial whose validation slice happens to
    # miss a class would otherwise be scored against a different label mapping.
    return float(log_loss(y_true, proba, labels=[0, 1]))


# --------------------------------------------------------------------------
# runners
# --------------------------------------------------------------------------
@dataclass
class Runner:
    name: str
    device: str
    fn: Callable

    def run(self, X_tr, y_tr, X_te, seed: int, n_jobs: int):
        return self.fn(X_tr, y_tr, X_te, seed, n_jobs, self.device)


def _run_mean_baseline(X_tr, y_tr, X_te, seed, n_jobs, device):
    """Predict the training fold's mean, ignoring the features entirely.

    Exists for splits where R2 comes out negative for everything. R2 is measured
    against the TEST block's own variance, so on a dataset with regime drift a
    genuinely useful model can still score below zero. Without a reference point
    the reader cannot tell "bad model" from "hard fold"; this row is that point.
    """
    with _Timer("cpu") as t_fit:
        mu = float(np.mean(y_tr))
    with _Timer("cpu") as t_pred:
        y_pred = np.full(len(X_te), mu, dtype="float64")
    return y_pred, t_fit.elapsed, t_pred.elapsed, {"train_mean": mu}


def _run_tabpfn(X_tr, y_tr, X_te, seed, n_jobs, device):
    from tabpfn import TabPFNRegressor

    model = TabPFNRegressor(device=device, random_state=seed)
    with _Timer(device) as t_fit:
        model.fit(X_tr, y_tr)
    with _Timer(device) as t_pred:
        y_pred = model.predict(X_te)
    return np.asarray(y_pred, dtype="float64"), t_fit.elapsed, t_pred.elapsed, {}


def _run_xgb_default(X_tr, y_tr, X_te, seed, n_jobs, device):
    from xgboost import XGBRegressor

    model = XGBRegressor(n_jobs=n_jobs, random_state=seed, device=device)
    with _Timer(device) as t_fit:
        model.fit(X_tr, y_tr)
    with _Timer(device) as t_pred:
        y_pred = model.predict(X_te)
    return np.asarray(y_pred, dtype="float64"), t_fit.elapsed, t_pred.elapsed, {}


def _run_lgbm_default(X_tr, y_tr, X_te, seed, n_jobs, device):
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(n_jobs=n_jobs, random_state=seed, verbose=-1)
    with _Timer(device) as t_fit:
        model.fit(X_tr, y_tr)
    with _Timer(device) as t_pred:
        y_pred = model.predict(X_te)
    return np.asarray(y_pred, dtype="float64"), t_fit.elapsed, t_pred.elapsed, {}


def _run_xgb_tuned(X_tr, y_tr, X_te, seed, n_jobs, device, ordered=False):
    from xgboost import XGBRegressor

    X_tr = np.asarray(X_tr, dtype="float64")
    y_tr = np.asarray(y_tr, dtype="float64")
    rng = np.random.RandomState(seed)
    tr_i, va_i = _inner_split(len(X_tr), seed, ordered)

    with _Timer(device) as t_search:
        best, best_rmse, best_rounds = None, np.inf, MAX_ROUNDS
        for _ in range(N_TRIALS):
            params = _sample_xgb(rng)
            m = XGBRegressor(
                n_estimators=MAX_ROUNDS,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                n_jobs=n_jobs,
                random_state=seed,
                device=device,
                **params,
            )
            m.fit(X_tr[tr_i], y_tr[tr_i], eval_set=[(X_tr[va_i], y_tr[va_i])], verbose=False)
            score = _rmse(y_tr[va_i], m.predict(X_tr[va_i]))
            if score < best_rmse:
                best_rmse, best = score, params
                best_rounds = int(getattr(m, "best_iteration", MAX_ROUNDS) or MAX_ROUNDS) + 1
    # Refit the winner on the FULL train fold at the round count validation chose.
    # Timed separately from the search: reporting a 30-trial search as "training
    # time" hides that the final model can be 8 trees fitted in milliseconds.
    with _Timer(device) as t_refit:
        model = XGBRegressor(
            n_estimators=best_rounds, n_jobs=n_jobs, random_state=seed, device=device, **best
        )
        model.fit(X_tr, y_tr)
    with _Timer(device) as t_pred:
        y_pred = model.predict(np.asarray(X_te, dtype="float64"))
    return (
        np.asarray(y_pred, dtype="float64"),
        t_search.elapsed + t_refit.elapsed,  # fit_time keeps its old meaning
        t_pred.elapsed,
        {"best_rounds": best_rounds, "val_rmse": best_rmse,
         "search_time": t_search.elapsed, "refit_time": t_refit.elapsed},
    )


def _run_lgbm_tuned(X_tr, y_tr, X_te, seed, n_jobs, device, ordered=False):
    import lightgbm as lgb
    from lightgbm import LGBMRegressor

    X_tr = np.asarray(X_tr, dtype="float64")
    y_tr = np.asarray(y_tr, dtype="float64")
    rng = np.random.RandomState(seed)
    tr_i, va_i = _inner_split(len(X_tr), seed, ordered)
    callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)]

    with _Timer(device) as t_search:
        best, best_rmse, best_rounds = None, np.inf, MAX_ROUNDS
        for _ in range(N_TRIALS):
            params = _sample_lgbm(rng)
            m = LGBMRegressor(
                n_estimators=MAX_ROUNDS, n_jobs=n_jobs, random_state=seed, verbose=-1, **params
            )
            m.fit(
                X_tr[tr_i],
                y_tr[tr_i],
                eval_set=[(X_tr[va_i], y_tr[va_i])],
                eval_metric="rmse",
                callbacks=callbacks,
            )
            score = _rmse(y_tr[va_i], m.predict(X_tr[va_i]))
            if score < best_rmse:
                best_rmse, best = score, params
                best_rounds = int(m.best_iteration_ or MAX_ROUNDS)
    with _Timer(device) as t_refit:
        model = LGBMRegressor(
            n_estimators=best_rounds, n_jobs=n_jobs, random_state=seed, verbose=-1, **best
        )
        model.fit(X_tr, y_tr)
    with _Timer(device) as t_pred:
        y_pred = model.predict(np.asarray(X_te, dtype="float64"))
    return (
        np.asarray(y_pred, dtype="float64"),
        t_search.elapsed + t_refit.elapsed,
        t_pred.elapsed,
        {"best_rounds": best_rounds, "val_rmse": best_rmse,
         "search_time": t_search.elapsed, "refit_time": t_refit.elapsed},
    )


# --------------------------------------------------------------------------
# classification runners
#
# Same shape as the regression ones, with two deliberate differences:
#   * the timed predict call is predict_proba, ONCE. Hard labels are derived
#     from it by argmax rather than issuing a second predict() - that would
#     double-count inference and is not what a deployed system does.
#   * tuned runners select on log-loss (see _logloss).
# --------------------------------------------------------------------------
def _run_tabpfn_clf(X_tr, y_tr, X_te, seed, n_jobs, device):
    from tabpfn import TabPFNClassifier

    model = TabPFNClassifier(device=device, random_state=seed)
    with _Timer(device) as t_fit:
        model.fit(X_tr, y_tr)
    with _Timer(device) as t_pred:
        proba = model.predict_proba(X_te)
    return np.asarray(proba, dtype="float64"), t_fit.elapsed, t_pred.elapsed, {}


def _run_xgb_default_clf(X_tr, y_tr, X_te, seed, n_jobs, device):
    from xgboost import XGBClassifier

    model = XGBClassifier(n_jobs=n_jobs, random_state=seed, device=device)
    with _Timer(device) as t_fit:
        model.fit(X_tr, y_tr)
    with _Timer(device) as t_pred:
        proba = model.predict_proba(X_te)
    return np.asarray(proba, dtype="float64"), t_fit.elapsed, t_pred.elapsed, {}


def _run_lgbm_default_clf(X_tr, y_tr, X_te, seed, n_jobs, device):
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(n_jobs=n_jobs, random_state=seed, verbose=-1)
    with _Timer(device) as t_fit:
        model.fit(X_tr, y_tr)
    with _Timer(device) as t_pred:
        proba = model.predict_proba(X_te)
    return np.asarray(proba, dtype="float64"), t_fit.elapsed, t_pred.elapsed, {}


def _run_xgb_tuned_clf(X_tr, y_tr, X_te, seed, n_jobs, device, ordered=False):
    from xgboost import XGBClassifier

    X_tr = np.asarray(X_tr, dtype="float64")
    y_tr = np.asarray(y_tr)
    rng = np.random.RandomState(seed)
    tr_i, va_i = _inner_split(len(X_tr), seed, ordered)

    with _Timer(device) as t_fit:
        best, best_score, best_rounds = None, np.inf, MAX_ROUNDS
        for _ in range(N_TRIALS):
            params = _sample_xgb(rng)
            m = XGBClassifier(
                n_estimators=MAX_ROUNDS,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                eval_metric="logloss",
                n_jobs=n_jobs,
                random_state=seed,
                device=device,
                **params,
            )
            m.fit(X_tr[tr_i], y_tr[tr_i], eval_set=[(X_tr[va_i], y_tr[va_i])], verbose=False)
            score = _logloss(y_tr[va_i], m.predict_proba(X_tr[va_i]))
            if score < best_score:
                best_score, best = score, params
                best_rounds = int(getattr(m, "best_iteration", MAX_ROUNDS) or MAX_ROUNDS) + 1
        # Refit the winner on the FULL train fold at the round count validation chose.
        model = XGBClassifier(
            n_estimators=best_rounds, n_jobs=n_jobs, random_state=seed, device=device, **best
        )
        model.fit(X_tr, y_tr)
    with _Timer(device) as t_pred:
        proba = model.predict_proba(np.asarray(X_te, dtype="float64"))
    return (
        np.asarray(proba, dtype="float64"),
        t_fit.elapsed,
        t_pred.elapsed,
        {"best_rounds": best_rounds, "val_logloss": best_score},
    )


def _run_lgbm_tuned_clf(X_tr, y_tr, X_te, seed, n_jobs, device, ordered=False):
    import lightgbm as lgb
    from lightgbm import LGBMClassifier

    X_tr = np.asarray(X_tr, dtype="float64")
    y_tr = np.asarray(y_tr)
    rng = np.random.RandomState(seed)
    tr_i, va_i = _inner_split(len(X_tr), seed, ordered)
    callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)]

    with _Timer(device) as t_fit:
        best, best_score, best_rounds = None, np.inf, MAX_ROUNDS
        for _ in range(N_TRIALS):
            params = _sample_lgbm(rng)
            m = LGBMClassifier(
                n_estimators=MAX_ROUNDS, n_jobs=n_jobs, random_state=seed, verbose=-1, **params
            )
            m.fit(
                X_tr[tr_i],
                y_tr[tr_i],
                eval_set=[(X_tr[va_i], y_tr[va_i])],
                eval_metric="binary_logloss",
                callbacks=callbacks,
            )
            score = _logloss(y_tr[va_i], m.predict_proba(X_tr[va_i]))
            if score < best_score:
                best_score, best = score, params
                best_rounds = int(m.best_iteration_ or MAX_ROUNDS)
        model = LGBMClassifier(
            n_estimators=best_rounds, n_jobs=n_jobs, random_state=seed, verbose=-1, **best
        )
        model.fit(X_tr, y_tr)
    with _Timer(device) as t_pred:
        proba = model.predict_proba(np.asarray(X_te, dtype="float64"))
    return (
        np.asarray(proba, dtype="float64"),
        t_fit.elapsed,
        t_pred.elapsed,
        {"best_rounds": best_rounds, "val_logloss": best_score},
    )


def build_registry(
    gbdt_device: str = "cpu",
    tabpfn_device: str = "cuda",
    task: str = "regression",
    cv: str = "kfold",
) -> dict[str, Runner]:
    # A time-based outer split demands a time-based INNER split too, or the
    # hyperparameter search optimises the very leakage the outer split removed.
    # See _inner_split(). Only the tuned runners have an inner split at all.
    # Every time-based cv mode belongs here - a new mode added to this list and
    # forgotten here reverts silently to the leaky random split.
    tune = partial(_tuned, ordered=cv in ("timeseries", "holdout"))

    if task == "classification":
        return {
            "tabpfn_v3": Runner("tabpfn_v3", tabpfn_device, _run_tabpfn_clf),
            "xgb_default": Runner("xgb_default", gbdt_device, _run_xgb_default_clf),
            "xgb_tuned": Runner("xgb_tuned", gbdt_device, tune(_run_xgb_tuned_clf)),
            "lgbm_default": Runner("lgbm_default", gbdt_device, _run_lgbm_default_clf),
            "lgbm_tuned": Runner("lgbm_tuned", gbdt_device, tune(_run_lgbm_tuned_clf)),
            "xgb_default_gpu": Runner("xgb_default_gpu", "cuda", _run_xgb_default_clf),
        }
    return {
        # Not a competitor - the reference point that makes a negative R2 readable.
        "mean_baseline": Runner("mean_baseline", "cpu", _run_mean_baseline),
        "tabpfn_v3": Runner("tabpfn_v3", tabpfn_device, _run_tabpfn),
        "xgb_default": Runner("xgb_default", gbdt_device, _run_xgb_default),
        "xgb_tuned": Runner("xgb_tuned", gbdt_device, tune(_run_xgb_tuned)),
        "lgbm_default": Runner("lgbm_default", gbdt_device, _run_lgbm_default),
        "lgbm_tuned": Runner("lgbm_tuned", gbdt_device, tune(_run_lgbm_tuned)),
        # bonus row: same XGBoost, one flag different
        "xgb_default_gpu": Runner("xgb_default_gpu", "cuda", _run_xgb_default),
    }
