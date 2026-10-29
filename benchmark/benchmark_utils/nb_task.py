"""Data loading through the neuralbench ``emg/pose`` task config.

Built on :func:`neuralbench.experiment_config.merge_task_config` for config
composition and :class:`neuralbench.data.Data` for the datasets::

    datasets, meta = load_task("emg", "pose", data_dir=...)

Differences with running neuralbench itself:

- everything runs locally with an explicit ``data_dir`` (the study path and
  the exca infra are overridden);
- the returned datasets follow the competition contract (``(X, y, info)``
  CPU tensors, see ``benchmark_utils.data``), so nothing downstream is tied
  to neuralset/neuralbench types.

Like the rest of the neuro stack, this module is import-heavy; import it
only from ``datasets/`` modules (never from solvers).
"""

import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

from benchmark_utils.data import to_numpy


class _LogOnce(logging.Filter):
    """Keep the first record of each message, drop the repeats."""

    def __init__(self):
        super().__init__()
        self._seen = set()

    def filter(self, record):
        key = (record.name, record.levelno, record.getMessage())
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


# Shared instance: ``addFilter`` is then idempotent across calls.
_LOG_ONCE = _LogOnce()


def _quiet_neuro_logs():
    """Keep the neuro stack's logs out of the participant output.

    Call after importing the stack: both libs set their logger level on
    import, which would undo this.
    """
    for name in ("neuralset", "exca"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.WARNING)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setStream(sys.stdout)
    # The extractor logs the same warning for every recording.
    logging.getLogger("neuralset.extractors.neuro").addFilter(_LOG_ONCE)
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")


def _task_data_config(modality, task, data_dir, overrides):
    """The task's merged ``data`` config, localized: everything runs from
    ``data_dir`` with no exca cluster, caller overrides applied last."""
    from neuralbench.experiment_config import merge_task_config

    data_dir = Path(data_dir)
    cfg = merge_task_config(modality, task, None)["data"]
    cfg.update({
        "study.source.path": str(data_dir),
        # Keep every cache next to the data.
        "study.source.infra.folder": str(data_dir / "cache"),
        "neuro.infra.cluster": None,
        "neuro.infra.folder": str(data_dir / "cache"),
        # exca chmods every cache entry it reads, which warns on each one
        # when the data is mounted read-only; None disables the chmod.
        "neuro.infra.permissions": None,
    })
    # NEURALBENCH_EXTRACT_CPUS=1 serializes extraction: concurrent neuro and
    # target extractors corrupt the study's TimelineLoader jsonl on
    # many-core hosts.
    extract_cpus = os.environ.get("NEURALBENCH_EXTRACT_CPUS")
    if extract_cpus:
        n = int(extract_cpus)
        cfg.update({
            "neuro.infra.cluster": None,
            "neuro.infra.cpus_per_task": n,
            "target.infra.cluster": None,
            "target.infra.cpus_per_task": n,
        })
    cfg.update(overrides or {})
    return cfg


def download_study(modality, task, data_dir, overrides=None):
    """One-time download of the task's study data (``Dataset.prepare``).

    A ``study.source.query`` naming subjects in ``overrides`` restricts the
    download to them.

    Skipped when ``data_dir`` is not writable: on the worker the staged data
    is mounted read-only, and ``Study.download()`` ends with a ``chmod``.
    """
    if not os.access(data_dir, os.W_OK):
        return

    import neuralset as ns
    _quiet_neuro_logs()

    cfg = _task_data_config(modality, task, data_dir, overrides)
    study = dict(cfg["study"]["source"])
    study["path"] = Path(study["path"]) / study["name"]
    # The first download leaves a marker that skips any later one, even for
    # new subjects; overwrite ignores it, and eegdash fetches missing files.
    ns.Study(**study).download(overwrite=True)


def require_prepared(modality, task, data_dir, overrides=None):
    """Fail with a clear message when the study data is not prepared.

    A no-op on a read-only staged mount, prepared by construction.
    """
    if not os.access(data_dir, os.W_OK):
        return

    cfg = _task_data_config(modality, task, data_dir, overrides)
    study = dict(cfg["study"]["source"])
    if not (Path(study["path"]) / study["name"]).exists():
        raise SystemExit(
            f"{modality}/{task} data is not prepared under {data_dir}. "
            "Run `benchopt prepare benchmark -d EMG2Pose` first."
        )


class _NBWindows(torch.utils.data.Dataset):
    """Adapt a prepared neuralset ``SegmentDataset`` to ``(X, y, info)``.

    ``seg_ds[i]`` yields a neuralset ``Batch`` whose ``data`` holds the
    ``neuro`` signal, the ``target`` and the ``subject_id``.
    """

    def __init__(self, seg_ds):
        self.seg_ds = seg_ds

    def __len__(self):
        return len(self.seg_ds)

    def __getitem__(self, i):
        data = self.seg_ds[i].data
        X = torch.as_tensor(to_numpy(data["neuro"]), dtype=torch.float32)
        if X.ndim == 3:                 # (1, C, T) -> (C, T)
            X = X[0]
        y = torch.as_tensor(to_numpy(data["target"]), dtype=torch.float32)
        if y.ndim and y.shape[0] == 1:  # (1, J, T) -> (J, T)
            y = y[0]
        subject = data.get("subject_id")
        subject = (-1 if subject is None
                   else int(np.asarray(to_numpy(subject)).reshape(-1)[0]))
        return X, y, {"subject_id": subject}


def load_task(modality, task, *, data_dir, seed=0, overrides=None):
    """Build the competition datasets + meta from a neuralbench task config.

    ``overrides`` are extra ``data:``-section overrides, as dotted keys,
    applied last. Returns ``datasets`` (``train``, ``val``, ``test``) and
    ``meta`` (``sfreq, n_chans, n_times, n_joints``).
    """
    from neuralbench.data import Data
    _quiet_neuro_logs()

    # Only neuralbench's datasets are kept, batching happens in make_loader.
    # ``num_workers=0`` stops the task defaults from injecting N_CPUS.
    cfg = _task_data_config(
        modality, task, data_dir,
        {"seed": seed, "num_workers": 0, "pin_memory": False,
         "persistent_workers": False, **(overrides or {})},
    )
    nb_loaders = Data(**cfg).prepare()
    datasets = {split: _NBWindows(loader.dataset)
                for split, loader in nb_loaders.items()}

    # Peek one window (lazy) for the shapes.
    X0, y0, _ = datasets["test"][0]
    meta = dict(
        sfreq=float(nb_loaders["test"].dataset.extractors["neuro"].frequency),
        n_chans=int(X0.shape[-2]),
        n_times=int(X0.shape[-1]),
        n_joints=int(y0.shape[0]),
    )
    return datasets, meta
