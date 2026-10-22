"""Sample submission: the starting point participants copy and improve.

A submission is a benchopt solver: a ``submission.py`` defining
``class Solver(CompetSolver)`` with a ``get_model`` method (and optionally
``fit``). See ``benchmark/benchmark_utils/base_solver.py`` for the contract.

Run it locally exactly as the platform does::

    benchopt run benchmark -d Simulated -s solution/submission.py
    # real data (large download, GPU recommended)
    benchopt prepare benchmark -d EMG2Pose
    benchopt run benchmark -d EMG2Pose -s solution/submission.py
"""

import torch
from torch import nn

from benchmark_utils.base_solver import CompetSolver


class ConvPose(nn.Module):
    """Temporal convolutions ``(B, C, T) -> (B, n_joints, T)``, in radians."""

    def __init__(self, n_chans, n_joints, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_chans, hidden, kernel_size=9, padding=4), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=9, padding=4), nn.GELU(),
            nn.Conv1d(hidden, n_joints, kernel_size=1),
        )

    def forward(self, X):
        # Per-window standardization of each channel.
        X = (X - X.mean(-1, keepdim=True)) / (X.std(-1, keepdim=True) + 1e-6)
        return self.net(X)

    def fit(self, train_loader, n_epochs=20):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.train()
        for _ in range(n_epochs):
            for X, y, _info in train_loader:
                opt.zero_grad()
                loss = nn.functional.l1_loss(self(X), y)
                loss.backward()
                opt.step()

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        return self(X)


class Solver(CompetSolver):

    name = "submission"

    def get_model(self, meta):
        return ConvPose(meta["n_chans"], meta["n_joints"]).to(meta["device"])
