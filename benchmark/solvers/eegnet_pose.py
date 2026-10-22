"""Baseline: EEGNet with a dense readout, trained end to end (GPU-ready).

The task wants a *sequence* out, not one label per window: with
``final_conv_length=1`` EEGNet's head slides along time and emits one
joint-angle vector per remaining time step, ``(B, n_joints, T')``. The
objective nearest-resamples ``T'`` up to the target's ``T``. Angles are
regressed in a standardized space (train mean and std kept as buffers).
"""

import torch
from braindecode.models import EEGNet
from torch import nn

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import resample_labels


class DensePose(nn.Module):
    """EEGNet emitting a joint-angle sequence, in radians."""

    def __init__(self, n_chans, n_joints, n_times):
        super().__init__()
        self.net = EEGNet(n_chans=n_chans, n_outputs=n_joints,
                          n_times=n_times, final_conv_length=1)
        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std", torch.ones(1))

    def forward(self, X):
        out = self.net(X)                       # standardized (B, J, T')
        # A singleton time axis gets squeezed by EEGNet's head: restore it.
        return out[..., None] if out.ndim == 2 else out

    def fit(self, train_loader, lr=1e-3, n_epochs=20):
        total = sq = count = 0.0
        for _X, y, _info in train_loader:
            total += y.sum()
            sq += (y ** 2).sum()
            count += y.numel()
        mean = total / count
        self.y_mean.fill_(float(mean))
        self.y_std.fill_(float((sq / count - mean ** 2).clamp_min(1e-8)
                               .sqrt()))

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.train()
        for _ in range(n_epochs):
            for X, y, _info in train_loader:
                opt.zero_grad()
                pred = self(X)                           # (B, J, T')
                target = resample_labels(y, pred.shape[-1])
                loss = loss_fn(pred, (target - self.y_mean) / self.y_std)
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        return self(X) * self.y_std + self.y_mean       # (B, J, T') radians


class Solver(CompetSolver):

    name = "EEGNet"

    requirements = ["pip::braindecode"]

    def get_model(self, meta):
        return DensePose(meta["n_chans"], meta["n_joints"],
                         meta["n_times"]).to(meta["device"])
