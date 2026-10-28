"""Submission contract.

A submission is a **valid benchopt solver**: a folder with a ``submission.py``
defining ``class Solver(CompetSolver)``. All its Python code lives in that
single file (benchopt loads it by path, so sibling ``.py`` files cannot be
imported); the folder also holds the trained weights, read from
``meta["submission_dir"]``. The platform only evaluates: it never trains.
Participants implement:

- ``load_model(self, meta)`` -> model (required): build the model, load the
  shipped weights when they exist, and place it on ``meta["device"]``.
- ``fit(self, model, train_set)``: train it. Defaults to
  ``model.fit(self.make_loader(train_set))``. It runs locally only, when the
  objective's ``training`` parameter is on (the default), and must reproduce
  the shipped weights: the organizers retrain the top submissions with it.
- ``save_model(self, model, path)``: write the trained weights into the
  submission folder ``path``, after ``fit``.

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
        self.model = self.load_model(self.meta)

    def run(self, _):
        # No training set on the platform: the model is the loaded one.
        if self.train_set is not None:
            self.fit(self.model, self.train_set)
            self.save_model(self.model, self.meta["submission_dir"])

    def get_result(self):
        return dict(model=self.model)

    def make_loader(self, dataset, batch_size=256, shuffle=True):
        """Batch ``dataset`` onto the device with the platform's workers."""
        return make_loader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            device=self.device, num_workers=self.meta["num_workers"],
        )

    # --- to be implemented by the submission ---------------------------

    def load_model(self, meta):
        """Build and return the model, with its trained weights if any."""
        raise NotImplementedError

    def fit(self, model, train_set):
        """Train the model in place."""
        model.fit(self.make_loader(train_set))

    def save_model(self, model, path):
        """Write the trained weights into the folder ``path``."""
