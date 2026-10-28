"""Data utilities: lazy datasets and loaders of ``(X, y, info)`` windows.

The benchmark passes **PyTorch datasets and dataloaders** between
components, not materialized arrays, because the EMG recordings are large.
The training set reaches the solver as a dataset, batched by
:func:`make_loader`. Every loader yields ``(X, y, info)`` batches:

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
from torch.utils.data import DataLoader, Dataset


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
    ``meta["device"]``, so a model can place itself there in ``load_model``.
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


class DeviceLoader:
    """Move batches to ``device`` in the main process.

    Worker processes (``num_workers > 0``) cannot move tensors to CUDA, so
    the wrapped loader collates CPU tensors and this wrapper moves them.
    Attribute access delegates to the wrapped loader.
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        for X, y, info in self.loader:
            yield X.to(self.device, non_blocking=True), y.to(
                self.device, non_blocking=True), info

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)


def make_loader(dataset, batch_size=256, shuffle=False, device="cpu",
                num_workers=0):
    """Batch a dataset of ``(X, y, info)`` windows onto ``device``."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        pin_memory=str(device).startswith("cuda"),
    )
    return DeviceLoader(loader, device)
