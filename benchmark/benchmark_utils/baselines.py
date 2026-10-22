"""Floor model, shared by the ``MeanPose`` baseline and the objective's
``get_one_result``."""

import numpy as np

from benchmark_utils.data import to_numpy


class ConstantPose:
    """Always predict the train-mean pose, constant over time.

    ``predict((B, C, T)) -> (B, n_joints, T)``: every window gets the same
    per-joint angle vector, broadcast along time.
    """

    def __init__(self, n_joints):
        self.value = np.zeros(n_joints, dtype=np.float32)

    def fit(self, train_loader):
        total, count = 0.0, 0
        for _X, y, _info in train_loader:
            y = to_numpy(y)                       # (B, J, T)
            total = total + y.mean(axis=-1).sum(axis=0)
            count += len(y)
        if count:
            self.value = (total / count).astype(np.float32)
        return self

    def predict(self, X):
        B, _C, T = to_numpy(X).shape
        return np.broadcast_to(
            self.value[None, :, None], (B, len(self.value), T)
        ).copy()
