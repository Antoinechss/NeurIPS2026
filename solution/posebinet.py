"""PoseBiNet: an offline, bidirectional EMG-to-pose decoder.

Training record (fill in after the final run):
    n_train_subjects = 158
    training time    = TODO h on 1x RTX 6000 Ada 48 GB (RunPod), + ~25 min data loading
    calibration      = TODO steps/min on that GPU
    command          = benchopt run benchmark -d "EMG2Pose[n_train_subjects=158,num_workers=12]" \
                           -s solution/posebinet.py

Why: the challenge scores whole 5-s windows offline, so the pose at time t may use the EMG
recorded after t too. The published emg2pose models (VEMG2Pose and its successors) are
causal, built for streaming: they only look at the past. This model looks both ways.
- Encoder: the VEMG2Pose TDS convolutional encoder (2 kHz -> 25 Hz features), with its
  1790-sample receptive field centred on t (895 reflected samples padded on each side)
  instead of ending at t.
- Decoder: a 2-layer bidirectional LSTM over the features resampled to 50 Hz, so every frame
  sees the whole window, then a linear read-out of the 20 absolute joint angles. No velocity
  integration, hence no drift. The read-out starts from the median training pose.
- Training recipe of vemg2pose_plus: band rotation / gain / noise augmentation, AdamW (no
  decay on biases and norms), warmup + cosine, grad clipping, EMA weights, fp16 encoder,
  test-time averaging over band rotations, optional seed ensemble.
The log reports the share of time spent waiting for data: if it is high, the data loader,
not the GPU, sets the speed.

Everything is seeded by `seed`: `fit` reproduces the shipped weights up to GPU noise.
"""

import copy
import math
import time

import torch
from braindecode.models import VEMG2Pose
from torch import nn
from torch.nn import functional as F

from benchmark_utils.base_solver import CompetSolver

WEIGHTS = "posebinet.pt"
CONTEXT = 1790  # receptive field of the VEMG2Pose encoder, in samples


class PoseBiNet(nn.Module):
    def __init__(self, n_chans, n_joints, sfreq, enc_channels=256, feature_dim=64,
                 hidden=256, layers=2, rate=50.0, drop_prob=0.1):
        super().__init__()
        # Borrow the encoder of VEMG2Pose; its causal LSTM decoder is dropped.
        self.encoder = VEMG2Pose(
            n_chans=n_chans, n_outputs=n_joints, sfreq=sfreq, n_times=2 * CONTEXT,
            encoder_channels=enc_channels, feature_dim=feature_dim,
            drop_prob=drop_prob).encoder
        self.sfreq, self.rate = float(sfreq), float(rate)
        self.lstm = nn.LSTM(feature_dim, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True,
                            dropout=drop_prob if layers > 1 else 0.0)
        self.head = nn.Linear(2 * hidden, n_joints)

    def forward(self, X):
        T = X.shape[-1]
        left = CONTEXT // 2
        mode = "reflect" if T > CONTEXT - left else "replicate"
        X = F.pad(X.float(), (left, CONTEXT - left), mode=mode)
        # Convolutions in fp16 (the bulk of the compute), the LSTM in fp32.
        with torch.autocast("cuda", dtype=torch.float16, enabled=X.is_cuda):
            feats = self.encoder(X)                       # (B, F, T / 80 + 1)
        with torch.autocast("cuda", enabled=False):
            n_frames = max(1, round(T / self.sfreq * self.rate))
            feats = F.interpolate(feats.float(), size=n_frames, mode="linear",
                                  align_corners=False)
            seq, _ = self.lstm(feats.transpose(1, 2))    # (B, n_frames, 2 * hidden)
            pose = self.head(seq).transpose(1, 2)         # (B, J, n_frames)
            return F.interpolate(pose, size=T, mode="linear", align_corners=False)


class Ensemble(nn.Module):
    """Average of PoseBiNets, with test-time averaging over band rotations."""

    def __init__(self, members, tta_shifts=(0,)):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.tta_shifts = tuple(tta_shifts)

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        X = X.float()
        B = X.shape[0]
        # All rotations in one batch: one big forward pass instead of k small ones.
        Xs = torch.cat([torch.roll(X, s, dims=1) for s in self.tta_shifts])
        out = 0.0
        for m in self.members:
            out = out + m(Xs).view(len(self.tta_shifts), B, -1,
                                   Xs.shape[-1]).mean(0)
        return out / len(self.members)


# ---------------------------------------------------------------- augmentation

def augment(X, g, max_shift=1, p_shift=0.5, chan_gain=0.15, glob_gain=0.25,
            noise=0.05):
    """Batch augmentation on X's device. X: (B, C, T); `g` a seeded Generator
    on the same device."""
    B, C, _ = X.shape
    dev = X.device
    if max_shift > 0:
        shifts = torch.randint(-max_shift, max_shift + 1, (B,), generator=g,
                               device=dev)
        keep = torch.rand(B, generator=g, device=dev) < p_shift
        shifts = torch.where(keep, shifts, torch.zeros_like(shifts))
        idx = (torch.arange(C, device=dev)[None, :] - shifts[:, None]) % C
        X = torch.gather(X, 1, idx[:, :, None].expand_as(X))
    gain = torch.exp(chan_gain * torch.randn(B, C, 1, generator=g, device=dev)
                     + glob_gain * torch.randn(B, 1, 1, generator=g, device=dev))
    X = X * gain
    if noise > 0:
        std = X.std(dim=-1, keepdim=True)
        X = X + noise * std * torch.randn(X.shape, generator=g, device=dev)
    return X


# -------------------------------------------------------------------- training

def pose_loss(pred, y, vel_weight, lag):
    """L1 on the joint angles over the labelled samples, plus, if `vel_weight`
    > 0, `vel_weight` x L1 on how much each angle changes over `lag` samples.
    Masked sums rather than boolean indexing: no GPU -> CPU sync per step."""
    mask = (~torch.isnan(y)).float()
    y = torch.nan_to_num(y)
    loss = ((pred - y).abs() * mask).sum() / mask.sum().clamp_min(1)
    if vel_weight > 0 and pred.shape[-1] > lag:
        dv = (pred[..., lag:] - pred[..., :-lag]) - (y[..., lag:] - y[..., :-lag])
        ok = mask[..., lag:] * mask[..., :-lag]
        loss = loss + vel_weight * (dv.abs() * ok).sum() / ok.sum().clamp_min(1)
    return loss


def train_member(model, loader, *, seed, lr, weight_decay, max_steps, warmup_steps,
                 ema_decay, max_hours, vel_weight, lag, reuse, aug):
    torch.manual_seed(seed)
    dev = next(model.parameters()).device
    cuda = dev.type == "cuda"
    g = torch.Generator(device=dev).manual_seed(seed + 1)

    # Start the read-out at the median pose of one batch (L1-optimal constant).
    _, y0, _ = next(iter(loader))
    with torch.no_grad():
        y0 = y0.transpose(0, 1).reshape(y0.shape[1], -1)
        model.head.bias.copy_(torch.nanmedian(y0, dim=1).values)
    del y0

    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    params = list(model.parameters())
    ema_params = list(ema.parameters())
    # Weight decay on weight matrices and kernels only, not on biases and norms.
    opt = torch.optim.AdamW(
        [{"params": [p for p in params if p.ndim >= 2],
          "weight_decay": weight_decay},
         {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}],
        lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    if cuda:  # compile the convolutional encoder only
        model.encoder = torch.compile(model.encoder)

    def lr_at(step):
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    start = time.time()
    deadline = start + 3600 * max_hours if max_hours else None
    step, epoch, waited = 0, 0, 0.0
    model.train()
    while step < max_steps:
        total, n = 0.0, 0
        batches = iter(loader)
        while step < max_steps:
            t0 = time.time()
            try:
                X, y, _info = next(batches)
            except StopIteration:
                break
            waited += time.time() - t0
            for _ in range(reuse):  # extra steps on the same batch, new augmentation
                for group in opt.param_groups:
                    group["lr"] = lr_at(step)
                loss = pose_loss(model(augment(X, g, **aug)), y, vel_weight, lag)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                with torch.no_grad():
                    d = min(ema_decay, (1 + step) / (10 + step))
                    torch._foreach_lerp_(ema_params, [p.detach() for p in params],
                                         1 - d)
                total, n, step = total + loss.detach(), n + 1, step + 1
                if step % 500 == 0:
                    elapsed = time.time() - start
                    l1 = float(total) / n
                    print(f"[seed {seed}] step {step}/{max_steps} lr {lr_at(step):.2e} "
                          f"loss {l1:.4f} ({math.degrees(l1):.1f} deg) "
                          f"{elapsed / 60:.1f} min, waiting for data "
                          f"{100 * waited / elapsed:.0f}%", flush=True)
                if step >= max_steps:
                    break
            if deadline and time.time() > deadline:
                break
        epoch += 1
        print(f"[seed {seed}] epoch {epoch} done, step {step}, "
              f"loss {float(total) / max(n, 1):.4f}, "
              f"{(time.time() - start) / 60:.1f} min", flush=True)
        if deadline and time.time() > deadline:
            print(f"[seed {seed}] stopped by the {max_hours} h budget", flush=True)
            break
    for b_ema, b in zip(ema.buffers(), model.buffers()):
        b_ema.copy_(b)
    ema.lstm.flatten_parameters()
    return ema


class Solver(CompetSolver):

    name = "posebinet"

    requirements = ["pip::braindecode"]

    parameters = {
        "n_models": [1],
        "max_steps": [25000],  # placeholder, calibrated on the rented GPU
        "batch_size": [256],
        "lr": [1e-3],
        "weight_decay": [1e-2],
        "warmup_steps": [1000],
        "ema_decay": [0.999],
        "drop_prob": [0.1],
        "enc_channels": [256],
        "hidden": [256],
        "vel_weight": [0.0],
        "reuse": [1],
        "max_shift": [1],
        "tta": [True],
        "max_hours": [11.0],  # hard stop for the organisers' 12 h V100 retrain
        "seed": [0],
    }

    def _build(self, meta):
        return PoseBiNet(meta["n_chans"], meta["n_joints"], meta["sfreq"],
                         enc_channels=self.enc_channels, hidden=self.hidden,
                         drop_prob=self.drop_prob)

    def load_model(self, meta):
        weights = meta["submission_dir"] / WEIGHTS
        n_models = self.n_models
        state = None
        if weights.exists():
            state = torch.load(weights, map_location="cpu")
            n_models = state["n_models"]
        torch.manual_seed(self.seed)  # same initial weights on every run
        shifts = (-1, 0, 1) if self.tta else (0,)
        model = Ensemble([self._build(meta) for _ in range(n_models)], shifts)
        if state is not None:
            model.load_state_dict(state["state_dict"])
        model = model.to(meta["device"])
        for m in model.members:
            m.lstm.flatten_parameters()
        return model

    def fit(self, model, train_set):
        loader = self.make_loader(train_set, batch_size=self.batch_size)
        hours_each = self.max_hours / len(model.members)
        lag = max(1, round(0.02 * self.meta["sfreq"]))  # speed over 20 ms
        trained = []
        for i, member in enumerate(model.members):
            trained.append(train_member(
                member, loader, seed=self.seed + 1000 * i, lr=self.lr,
                weight_decay=self.weight_decay, max_steps=self.max_steps,
                warmup_steps=self.warmup_steps, ema_decay=self.ema_decay,
                max_hours=hours_each, vel_weight=self.vel_weight, lag=lag,
                reuse=self.reuse, aug=dict(max_shift=self.max_shift)))
        model.members = nn.ModuleList(trained)

    def save_model(self, model, path):
        torch.save({"n_models": len(model.members),
                    "state_dict": model.state_dict()}, path / WEIGHTS)
