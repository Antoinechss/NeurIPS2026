"""Baseline: per-time-step ridge regression from EMG channels to angles.

The EMG channel vector at time ``t`` maps to the joint-angle vector at
``t``: windows flatten to ``(B*T, C)`` for a multi-output ridge, and the
prediction reshapes back to ``(B, n_joints, T)``. Training keeps one time
step in ``STEP`` so the design matrix fits in memory.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy

STEP = 20


def _steps(a):
    """``(B, K, T) -> (B*T, K)``: one row per time step."""
    return np.moveaxis(a, 1, -1).reshape(-1, a.shape[1])


class TimeRidge:
    """Per-time-step ridge: ``(B, C, T) -> (B, n_joints, T)``."""

    def __init__(self, n_joints):
        self.n_joints = n_joints
        self.clf = make_pipeline(StandardScaler(), Ridge())

    def fit(self, train_loader):
        feats, targets = [], []
        for X, y, _info in train_loader:
            feats.append(_steps(to_numpy(X)[..., ::STEP]))
            targets.append(_steps(to_numpy(y)[..., ::STEP]))
        self.clf.fit(np.concatenate(feats), np.concatenate(targets))
        return self

    def predict(self, X):
        X = to_numpy(X)
        B, _C, T = X.shape
        pred = self.clf.predict(_steps(X))
        return np.moveaxis(pred.reshape(B, T, self.n_joints), -1, 1)


class Solver(CompetSolver):

    name = "Ridge"

    requirements = ["scikit-learn"]

    def get_model(self, meta):
        return TimeRidge(meta["n_joints"])
