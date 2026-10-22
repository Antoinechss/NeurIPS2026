"""Metric of the competition: angular errors of joint-angle sequences.

Numpy only, so the objective never needs the neuralbench stack. The
semantics mirror neuralbench's ``emg/pose`` evaluation.
"""

import numpy as np


class SequenceRegressionScores:
    """Streaming scores of channel-major sequence targets ``(B, C, T)``.

    Follows neuralbench's dense-target evaluation (``emg/pose``): every time
    step is a row and every channel an output, and a step whose target has a
    NaN in any channel is left out. Call :meth:`update` once per batch, then
    :meth:`compute`:

    - ``mae``: mean absolute error over every (step, channel), pooled;
    - ``mae_group_mean`` / ``mae_group_std``: mean and sample standard
      deviation (``ddof=1``, NaN below two groups) of the per-group MAE,
      neuralbench's ``GroupedMetric`` with ``reduction="mean"``/``"std"``;
    - ``n_groups``: number of groups with at least one scored step;
    - ``rmse``: root mean squared error, pooled;
    - ``r2``: coefficient of determination of each channel across steps,
      averaged over channels (torchmetrics ``R2Score``, uniform average).
    """

    def __init__(self):
        self._groups, self._abs_sums, self._counts = [], [], []
        self._sq_err = self._y_sum = self._y_sq_sum = 0.0
        self._n_steps = 0

    def update(self, y_pred, y_true, groups):
        """Add a batch: ``y_pred`` and ``y_true`` of shape ``(B, C, T)``,
        ``groups`` of shape ``(B,)`` (one group per window)."""
        y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)
        err = y_pred - y_true
        labelled = ~np.isnan(y_true).any(axis=1)          # (B, T)
        if not labelled.all():
            unlabelled = ~labelled[:, None, :]
            err = np.where(unlabelled, 0.0, err)
            y_true = np.where(unlabelled, 0.0, y_true)

        self._groups.append(np.asarray(groups).reshape(-1))
        self._abs_sums.append(np.abs(err).sum(axis=(1, 2), dtype=np.float64))
        self._counts.append(labelled.sum(axis=1) * y_true.shape[1])
        # Per-channel sums for the RMSE and the R2.
        self._sq_err = self._sq_err + np.square(err).sum(
            axis=(0, 2), dtype=np.float64)
        self._y_sum = self._y_sum + y_true.sum(axis=(0, 2), dtype=np.float64)
        self._y_sq_sum = self._y_sq_sum + np.square(y_true).sum(
            axis=(0, 2), dtype=np.float64)
        self._n_steps += int(labelled.sum())

    def compute(self):
        """Return the scores as a dict of floats (``n_groups`` an int)."""
        groups = np.concatenate(self._groups)
        abs_sums = np.concatenate(self._abs_sums)
        counts = np.concatenate(self._counts).astype(np.float64)

        _, inverse = np.unique(groups, return_inverse=True)
        inverse = inverse.reshape(-1)
        group_abs = np.bincount(inverse, weights=abs_sums)
        group_counts = np.bincount(inverse, weights=counts)
        group_mae = (group_abs[group_counts > 0]
                     / group_counts[group_counts > 0])

        n_values = counts.sum()
        mean_y = self._y_sum / max(self._n_steps, 1)
        ss_tot = self._y_sq_sum - self._y_sum * mean_y
        ss_res = self._sq_err
        # torchmetrics' guard for (near-)constant channels.
        res_ok = ~np.isclose(ss_res, 0.0, atol=1e-4)
        tot_ok = ~np.isclose(ss_tot, 0.0, atol=1e-4)
        r2 = np.ones_like(ss_res)
        both = res_ok & tot_ok
        r2[both] = 1.0 - ss_res[both] / ss_tot[both]
        r2[res_ok & ~tot_ok] = 0.0

        nan = float("nan")
        return dict(
            mae=float(abs_sums.sum() / n_values) if n_values else nan,
            mae_group_mean=float(group_mae.mean()) if len(group_mae) else nan,
            mae_group_std=(float(group_mae.std(ddof=1))
                           if len(group_mae) > 1 else nan),
            n_groups=int(len(group_mae)),
            rmse=float(np.sqrt(ss_res.sum() / n_values)) if n_values else nan,
            r2=float(r2.mean()) if self._n_steps > 1 else nan,
        )
