"""Submission contract.

A submission is a **valid benchopt solver**: a folder with a ``submission.py``
defining ``class Solver(CompetSolver)``. All its Python code lives in that
single file (benchopt loads it by path, so sibling ``.py`` files cannot be
imported); the folder may also hold non-code files (weights, configs) read
from ``meta["submission_dir"]``. Participants implement:

- ``get_model(self, meta)`` -> model (required): build the untrained model
  and place it on ``meta["device"]``.
- ``fit(self, model, train_set)`` (optional): train it. Defaults to
  ``model.fit(self.make_loader(train_set))``. It runs on the platform, on a
  GPU.

``train_set`` is a torch ``Dataset`` of ``(X, y, info)`` windows on the CPU:
``X`` is ``(n_chans, n_times)`` EMG, ``y`` is ``(n_joints, n_times)`` joint
angles in radians (see ``benchmark_utils.data``). ``self.make_loader(
train_set, batch_size=256, shuffle=True)`` batches it onto ``meta["device"]``
with the platform's dataloader workers. ``meta`` holds ``n_chans, n_times,
n_joints, sfreq, device, num_workers, submission_dir``.

The trained model must expose ``model.predict(X) -> (B, n_joints, T')`` in
radians, with ``X`` a torch batch ``(B, n_chans, n_times)``. A coarser time
axis ``T'`` is nearest-resampled to ``n_times``. Batches are independent
windows: do not rely on their order or keep state across ``predict`` calls.
"""

import inspect
from pathlib import Path

from benchopt import BaseSolver

from benchmark_utils.data import make_loader


class CompetSolver(BaseSolver):
    """Base class for all submissions (and the built-in baselines).

    The competition image fixes the available packages: submissions cannot
    install extra ones.
    """

    requirements = []

    sampling_strategy = "run_once"

    def set_objective(self, train_set, **meta):
        self.train_set = train_set
        self.device = meta["device"]
        submission_dir = Path(inspect.getfile(type(self))).parent
        self.meta = {**meta, "submission_dir": submission_dir}
        self.model = self.get_model(self.meta)

    def run(self, _):
        self.fit(self.model, self.train_set)

    def get_result(self):
        return dict(model=self.model)

    def make_loader(self, dataset, batch_size=256, shuffle=True):
        """Batch ``dataset`` onto the device with the platform's workers."""
        return make_loader(
            dataset, batch_size=batch_size, shuffle=shuffle, device=self.device,
            num_workers=self.meta["num_workers"],
        )

    # --- to be implemented by the submission ---------------------------

    def get_model(self, meta):
        """Build and return the (untrained) model."""
        raise NotImplementedError

    def fit(self, model, train_set):
        """Train the model in place."""
        model.fit(self.make_loader(train_set))
