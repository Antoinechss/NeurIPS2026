"""Objective for the EMG-to-pose competition.

Regress hand joint-angle trajectories from wrist surface EMG. The solver gets
the training dataset and trains its model on the platform; the objective then
feeds it torch batches ``(B, C, T)`` of 16-channel EMG windows and expects
``predict(X) -> (B, n_joints, T')`` in **radians** (a coarser time axis is
nearest-resampled to the target's ``T``).

Ranking metric: **mean angular MAE** in degrees, over joints and time.
Reported alongside: the mean and standard deviation of the per-subject MAE,
the RMSE (degrees) and the R2 averaged over joints. Each held-out subset is
scored on its own: ``public`` (shown live) and ``private`` (hidden, final
ranking). Metrics are emitted as ``objective_<metric>_<subset>``; the phase
config maps them to leaderboard keys.
"""

import numpy as np
from benchopt import BaseObjective

from benchmark_utils.data import resample_labels, subject_ids, to_numpy
from benchmark_utils.metrics import SequenceRegressionScores

DEG = 180.0 / np.pi


class Objective(BaseObjective):

    name = "EMG-pose"
    url = "https://github.com/x-datascience-datacamp/codabench_emg2pose"

    requirements = ["scikit-learn"]

    min_benchopt_version = "1.10.0"

    # Each solver trains once (no convergence curve).
    sampling_strategy = "run_once"

    def set_data(self, train_set, test_sets, n_joints, **meta):
        # test_sets: ``name -> loader``; the test labels stay here so the
        # objective owns evaluation.
        self.train_set = train_set
        self.test_sets = test_sets
        self.n_joints = n_joints
        self.meta = meta  # n_chans, n_times, sfreq, device, num_workers

    def get_objective(self):
        return dict(train_set=self.train_set, n_joints=self.n_joints,
                    **self.meta)

    def evaluate_result(self, model):
        results = {}
        for name, loader in self.test_sets.items():
            scores = SequenceRegressionScores()
            for X, y, info in loader:
                pred = to_numpy(model.predict(X))     # (B, J, T')
                y = to_numpy(y)                       # (B, J, T)
                pred = resample_labels(pred, y.shape[-1])
                scores.update(pred, y, subject_ids(info, len(y)))
            scores = scores.compute()
            results.update({
                f"angular_mae_{name}": scores["mae"] * DEG,
                f"angular_mae_subject_mean_{name}":
                    scores["mae_group_mean"] * DEG,
                f"angular_mae_subject_std_{name}":
                    scores["mae_group_std"] * DEG,
                f"angular_rmse_{name}": scores["rmse"] * DEG,
                f"r2_{name}": scores["r2"],
            })
        return results

    def get_one_result(self):
        # The mean-pose floor, to validate the metric in ``benchopt test``.
        from benchmark_utils.baselines import ConstantPose
        return dict(model=ConstantPose(self.n_joints))
