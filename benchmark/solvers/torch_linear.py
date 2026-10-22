"""Baseline: a per-time-step linear network in plain PyTorch.

The smallest end-to-end torch example of the submission contract: a linear
layer applied at each time step maps ``(B, C, T) -> (B, n_joints, T)``.
Angles are regressed in a standardized space (train mean and std kept as
buffers). Swap the network for your own architecture and the rest holds.
"""

import torch
from torch import nn

from benchmark_utils.base_solver import CompetSolver


class TorchLinearPose(nn.Module):
    """Per-time-step map ``(B, C, T) -> (B, n_joints, T)``, in radians."""

    def __init__(self, n_chans, n_joints):
        super().__init__()
        self.linear = nn.Linear(n_chans, n_joints)
        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std", torch.ones(1))

    def forward(self, X):
        # (B, C, T) -> (B, T, C) -> linear -> (B, T, J) -> (B, J, T)
        return self.linear(X.transpose(1, 2)).transpose(1, 2)

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        return self(X) * self.y_std + self.y_mean       # radians


class Solver(CompetSolver):

    name = "Torch-Linear"

    def get_model(self, meta):
        return TorchLinearPose(meta["n_chans"], meta["n_joints"]).to(
            meta["device"])

    def fit(self, model, train_loader):
        # One pass for the target scale, so the loss is well conditioned.
        total = sq = count = 0.0
        for _X, y, _info in train_loader:
            total += y.sum()
            sq += (y ** 2).sum()
            count += y.numel()
        mean = total / count
        model.y_mean.fill_(float(mean))
        model.y_std.fill_(
            float((sq / count - mean ** 2).clamp_min(1e-8).sqrt()))

        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(30):  # epochs
            for X, y, _info in train_loader:
                opt.zero_grad()
                loss = loss_fn(model(X), (y - model.y_mean) / model.y_std)
                loss.backward()
                opt.step()
