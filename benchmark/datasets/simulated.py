"""Synthetic dataset: no data needed, runs anywhere (smoke tests and CI).

It returns the same keys and test subsets as the real dataset. Joint-angle
trajectories are smooth random curves in a plausible range, and the EMG
channels are an instantaneous linear mixture of the joint angles plus noise,
so a per-time-step linear readout recovers the pose above the noise floor.
"""

import numpy as np
from benchopt import BaseDataset

from benchmark_utils.data import ArrayWindows, get_device, make_loader


class Dataset(BaseDataset):

    name = "Simulated"

    requirements = []

    parameters = {
        # 16-channel wrist EMG -> 20 joint angles, as in the real task.
        "n_chans, n_joints, n_times": [(16, 20, 400)],
        "n_train, n_test": [(100, 50)],
        "sfreq": [200.0],
    }

    test_parameters = {
        "n_chans, n_joints, n_times": [(4, 5, 80)],
        "n_train, n_test": [(10, 5)],
        "sfreq": [200.0],
    }

    def _make_windows(self, rng, n, mixing):
        # Cumulative noise, smoothed and scaled to radians.
        y = np.cumsum(rng.standard_normal((n, self.n_joints, self.n_times)),
                      axis=-1)
        kernel = np.ones(9) / 9.0
        y = np.apply_along_axis(
            lambda s: np.convolve(s, kernel, mode="same"), -1, y
        )
        y = 0.5 * y / np.abs(y).max(axis=-1, keepdims=True) + 0.35
        X = np.einsum("cj,njt->nct", mixing, y)
        X += 2.0 * rng.standard_normal(X.shape)
        subject = rng.integers(0, 4, size=n)
        return X.astype(np.float32), y.astype(np.float32), subject

    def get_data(self):
        rng = np.random.default_rng(self.get_seed())
        mixing = rng.standard_normal((self.n_chans, self.n_joints)) * 0.3
        device = get_device()

        train_set = ArrayWindows(
            *self._make_windows(rng, self.n_train, mixing))
        test_sets = {}
        for name in ("public", "private"):
            test_sets[name] = make_loader(
                ArrayWindows(*self._make_windows(rng, self.n_test, mixing)),
                device=device)
        return dict(
            train_set=train_set, test_sets=test_sets,
            n_joints=self.n_joints, n_chans=self.n_chans,
            n_times=self.n_times, sfreq=self.sfreq, device=device,
            num_workers=0,
        )
