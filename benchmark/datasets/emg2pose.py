"""EMG2Pose: wrist surface EMG to hand pose (Salter et al. 2024).

Wraps the neuralbench ``emg/pose`` task (see ``benchmark_utils.nb_task``):
5-s windows of 16-channel wrist EMG at 2 kHz, target = 20 hand joint-angle
trajectories in radians, with the corpus' predefined split.

    benchopt prepare benchmark -d EMG2Pose

- **Training set**: the train split of the first ``n_train_subjects``
  subjects. The full corpus is hundreds of GB; the subset sizes the training
  for the platform's time budget (to be timed on the GPU worker).
- **public** test set: the corpus' ``user_stage`` test rows (movement stages
  held out for users seen elsewhere in the data).
- **private** test set: a sealed cohort of new users, staged on the worker
  only, under ``get_data_path("emg2pose_sealed")`` as ``X.npy`` (N, 16, T),
  ``y.npy`` (N, 20, T) in radians and ``subject.npy`` (N,). Absent locally,
  so a local run scores the public set only.

``prepare`` downloads the study and runs the extraction once: it caches next
to the data, so runs only read warm caches. Both steps are idempotent.
"""

import numpy as np
from benchopt import BaseDataset
from benchopt.config import get_data_path

# Hard requirement, imported at module level so benchopt reports the
# dataset as not installed when it is missing.
import neuralbench  # noqa: F401

from benchmark_utils.data import get_device, make_loader
from benchmark_utils.nb_task import download_study, load_task, require_prepared


class Dataset(BaseDataset):

    name = "EMG2Pose"

    requirements = [
        "pip::neuralset==0.4.0",
        "pip::neuralfetch==0.4.0",
        "pip::neuraltrain==0.4.0",
        "pip::neuralbench==0.4.0",
        "pip::mne", "scikit-learn",
    ]

    parameters = {
        "n_train_subjects": [8],
        "batch_size": [64],
        # Dataloader workers; raise it from the phase config on a worker
        # with spare cores.
        "num_workers": [0],
    }

    # Loader settings do not change the prepared data.
    prepare_cache_ignore = ("batch_size", "num_workers")

    def prepare(self):
        download_study("emg", "pose", self._data_dir())
        self._load()

    def _data_dir(self):
        path = get_data_path("emg2pose")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load(self, device="cpu"):
        cache = str(self._data_dir() / "cache")
        return load_task(
            "emg", "pose",
            data_dir=self._data_dir(),
            device=device,
            batch_size=self.batch_size,
            seed=self.get_seed(),
            num_workers=self.num_workers,
            overrides={
                "study.keep_train_subjects": {
                    "name": "QueryEvents",
                    "query": ("split == 'test' or "
                              f"subject_index < {self.n_train_subjects}"),
                },
                # Cache the target extraction next to the data, not in the
                # container's /tmp.
                "target.infra.cluster": None,
                "target.infra.folder": cache,
                "target.infra.permissions": None,
                # Inline timeline cache: concurrent writers corrupt it.
                "study.source.timelines.infra.backend": "Cached",
                "study.source.timelines.infra.folder": cache,
            },
        )

    def get_data(self):
        require_prepared("emg", "pose", self._data_dir())
        device = get_device()
        loaders, meta = self._load(device=device)
        test_sets = {"public": loaders["test"]}
        sealed = get_data_path("emg2pose_sealed")
        if (sealed / "X.npy").exists():
            test_sets["private"] = make_loader(
                np.load(sealed / "X.npy", mmap_mode="r"),
                np.load(sealed / "y.npy", mmap_mode="r"),
                np.load(sealed / "subject.npy"),
                batch_size=self.batch_size, device=device,
            )
        return dict(train_loader=loaders["train"], test_sets=test_sets,
                    **meta)
