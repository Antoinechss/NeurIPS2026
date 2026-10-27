"""Baseline: VEMG2Pose (Salter et al. 2024), emg2pose's regression model.

NeuralBench's reference recipe, trained on the platform: L1 loss on the
angles in radians, Adam at 1e-3, a fixed epoch budget instead of early
stopping, no augmentation. Its valid convolutions consume a left context
before the scored samples, recreated here by padding each window on the left.

The decoder rolls the pose out step by step at 50 Hz, so a small batch leaves
the GPU waiting on kernel launches: training uses batches of 256, compiles
the encoder, and stops at a wall-clock budget.
"""

import time

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

    def fit(self, train_loader, lr=1e-3, n_epochs=20, max_hours=None):
        # Compile the convolutional encoder only: compiling the rollout loop
        # traces all 250 steps. CPU runs (CI) stay eager, they lack a toolchain.
        if next(self.parameters()).is_cuda:
            self.net.encoder = torch.compile(self.net.encoder)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        start = time.time()
        deadline = start + 3600 * max_hours if max_hours else None
        self.train()
        for epoch in range(n_epochs):
            # Summed on the device: a per-step .item() would sync the GPU.
            total, n_steps = 0.0, 0
            for X, y, _info in train_loader:
                opt.zero_grad()
                labelled = ~torch.isnan(y)
                loss = (self(X) - y).abs()[labelled].mean()
                loss.backward()
                opt.step()
                total, n_steps = total + loss.detach(), n_steps + 1
                if deadline and time.time() > deadline:
                    break
            print(f"VEMG2Pose epoch {epoch + 1}: train L1 "
                  f"{float(total) / max(n_steps, 1):.4f} rad over {n_steps} "
                  f"steps, {(time.time() - start) / 60:.1f} min", flush=True)
            if deadline and time.time() > deadline:
                break
        return self

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        return self(X)


class Solver(CompetSolver):

    name = "VEMG2Pose"

    requirements = ["pip::braindecode"]

    parameters = {"n_epochs": [100], "batch_size": [256], "max_hours": [4.5]}

    def get_model(self, meta):
        return PoseNet(meta["n_chans"], meta["n_joints"], meta["n_times"],
                       meta["sfreq"]).to(meta["device"])

    def fit(self, model, train_set):
        model.fit(self.make_loader(train_set, batch_size=self.batch_size),
                  n_epochs=self.n_epochs, max_hours=self.max_hours)
