"""Data utilities: lazy loaders of ``(X, y, info)`` batches.

The benchmark passes **PyTorch dataloaders** between components, not
materialized arrays, because the EMG recordings are large. Every loader
yields ``(X, y, info)`` batches:

- ``X``    : float32 tensor ``(B, C, T)``, windows of ``C`` EMG channels.
- ``y``    : float32 tensor ``(B, n_joints, T)``, joint angles in radians.
- ``info`` : per-window metadata kept away from the model, so the metrics
             can group windows by subject (``subject_id``, or ``record_id``
             for the simulated data).

Tensors stay torch tensors end to end; conversion to numpy happens only at
the scikit-learn boundaries (linear heads and metrics), via :func:`to_numpy`.
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, default_collate


def to_numpy(x):
    """Convert a torch tensor (or array-like) to a numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def subject_ids(info, n):
    """Subject id of each of the ``n`` windows of a batch, from its ``info``.

    Falls back on ``record_id`` (simulated data), then on a single id.
    """
    sid = info.get("subject_id", info.get("record_id"))
    if sid is None:
        return np.zeros(n, dtype=np.int64)
    return to_numpy(sid).reshape(-1)


def get_device():
    """``cuda`` when available, else ``cpu``; ``COMPET_DEVICE`` overrides.

    Datasets move every batch onto this device and advertise it through
    ``meta["device"]``, so a model can place itself there in ``get_model``.
    """
    forced = os.environ.get("COMPET_DEVICE")
    if forced:
        return forced
    return "cuda" if torch.cuda.is_available() else "cpu"


def resample_labels(seq, new_len):
    """Nearest-neighbour resample along the last axis to ``new_len`` steps.

    Aligns a prediction on a coarser time axis with the target's ``T``.
    Type-preserving: torch in, torch out; numpy in, numpy out.
    """
    length = seq.shape[-1]
    if length == new_len:
        return seq
    if isinstance(seq, torch.Tensor):
        idx = torch.floor(
            torch.arange(new_len, device=seq.device) * (length / new_len)
        ).long().clamp(0, length - 1)
        return seq[..., idx]
    seq = np.asarray(seq)
    idx = np.floor(np.arange(new_len) * (length / new_len)).astype(int)
    idx = np.clip(idx, 0, length - 1)
    return seq[..., idx]


class ArrayWindows(Dataset):
    """Windows held in arrays, yielding ``(X, y, info)`` torch tensors.

    ``X`` and ``y`` may be memory-mapped: each window is read on access.
    """

    def __init__(self, X, y, subject_id=None):
        self.X, self.y = X, y
        self.subject_id = (np.zeros(len(X), dtype=np.int64)
                           if subject_id is None
                           else np.asarray(subject_id, dtype=np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        X = torch.as_tensor(np.asarray(self.X[i]), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(self.y[i]), dtype=torch.float32)
        return X, y, {"subject_id": int(self.subject_id[i])}


def _move_collate(device):
    """Default collate, then move ``X`` and ``y`` onto ``device``."""
    def collate(batch):
        X, y, info = default_collate(batch)
        return X.to(device), y.to(device), info
    return collate


def make_loader(X, y, subject_id=None, batch_size=32, shuffle=False,
                device="cpu"):
    """Wrap arrays in ``ArrayWindows`` and a ``DataLoader`` on ``device``."""
    return DataLoader(
        ArrayWindows(X, y, subject_id), batch_size=batch_size,
        shuffle=shuffle, collate_fn=_move_collate(device),
    )
