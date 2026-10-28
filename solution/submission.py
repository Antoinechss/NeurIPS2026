"""Sample submission: the starting point participants copy and improve.

A submission is a benchopt solver: a ``submission.py`` defining
``class Solver(CompetSolver)``, next to its trained weights. The platform only
evaluates: it calls ``load_model`` and scores ``predict``. ``fit`` runs
locally and must reproduce the weights. See
``benchmark/benchmark_utils/base_solver.py`` for the contract.

Train locally, which writes ``weights.pt`` next to this file, then zip this
folder and upload it::

    benchopt run benchmark -d Simulated -s solution/submission.py   # smoke
    benchopt prepare benchmark -d EMG2Pose
    benchopt run benchmark -d EMG2Pose -s solution/submission.py
    # check the upload exactly as the platform runs it (no training)
    benchopt run benchmark -d EMG2Pose -s solution/submission.py \
        -o "EMG-pose[training=False]"

A training run overwrites ``weights.pt``: train on EMG2Pose last.
"""

import torch
from torch import nn

from benchmark_utils.base_solver import CompetSolver

WEIGHTS = "weights.pt"


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

    def load_model(self, meta):
        model = ConvPose(meta["n_chans"], meta["n_joints"])
        weights = meta["submission_dir"] / WEIGHTS
        if weights.exists():
            model.load_state_dict(torch.load(weights, map_location="cpu"))
        return model.to(meta["device"])

    def save_model(self, model, path):
        torch.save(model.state_dict(), path / WEIGHTS)
