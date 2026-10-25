"""Baseline: VEMG2Pose (Salter et al. 2024), emg2pose's regression model.

NeuralBench's reference recipe, trained on the platform: L1 loss on the
angles in radians, Adam at 1e-3, a fixed epoch budget instead of early
stopping, no augmentation. Its valid convolutions consume a left context
before the scored samples, recreated here by padding each window on the left.
"""

import torch
from braindecode.models import VEMG2Pose
from torch import nn
from torch.nn import functional as F

from benchmark_utils.base_solver import CompetSolver

# Samples the default VEMG2Pose consumes before its first prediction.
LEFT_CONTEXT = 1790


class PoseNet(nn.Module):
    """VEMG2Pose emitting ``(B, n_joints, n_times)`` angles in radians."""

    def __init__(self, n_chans, n_joints, n_times, sfreq):
        super().__init__()
        self.net = VEMG2Pose(n_chans=n_chans, n_outputs=n_joints,
                             sfreq=sfreq, n_times=n_times + LEFT_CONTEXT)

    def forward(self, X):
        # Reflect needs a window longer than the context (short demo data).
        mode = "reflect" if X.shape[-1] > LEFT_CONTEXT else "replicate"
        X = F.pad(X.float(), (LEFT_CONTEXT, 0), mode=mode)
        return self.net(X).transpose(1, 2)              # (B, J, T)

    def fit(self, train_loader, lr=1e-3, n_epochs=20):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for _ in range(n_epochs):
            for X, y, _info in train_loader:
                opt.zero_grad()
                labelled = ~torch.isnan(y)
                loss = (self(X) - y).abs()[labelled].mean()
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        return self(X)


class Solver(CompetSolver):

    name = "VEMG2Pose"

    requirements = ["pip::braindecode"]

    parameters = {"n_epochs": [20]}

    def get_model(self, meta):
        return PoseNet(meta["n_chans"], meta["n_joints"], meta["n_times"],
                       meta["sfreq"]).to(meta["device"])

    def fit(self, model, train_loader):
        model.fit(train_loader, n_epochs=self.n_epochs)
