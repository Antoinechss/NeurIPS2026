"""EMG2Pose: wrist surface EMG to hand pose (Salter et al. 2024).

Wraps the neuralbench ``emg/pose`` task (see ``benchmark_utils.nb_task``):
5-s windows of 16-channel wrist EMG at 2 kHz, target = 20 hand joint-angle
trajectories in radians, with the corpus' predefined split.

    benchopt prepare benchmark -d EMG2Pose

- **Training set**: the train split of the first ``n_train_subjects`` of
  the 158 training users. The full corpus is ~340 GB; the subset sizes the
  download and the training for the platform's time budget.
- **public** test set: the corpus' ``user_stage`` test rows, from 20 users
  absent from training, on held-out movement stages.
- **private** test set: a sealed cohort of new users, staged on the worker
  only, under ``get_data_path("emg2pose_sealed")`` as ``X.npy`` (N, 16, T),
  ``y.npy`` (N, 20, T) in radians and ``subject.npy`` (N,). Absent locally,
  so a local run scores the public set only.

``prepare`` downloads these subjects only and runs the extraction once: it
caches next to the data, so runs only read warm caches. Both steps are
idempotent. The download opens one connection per CPU, and NEMAR throttles
each connection, so prepare on a machine with many cores.
"""

import numpy as np
from benchopt import BaseDataset
from benchopt.config import get_data_path

# Hard requirement, imported at module level so benchopt reports the
# dataset as not installed when it is missing.
import neuralbench  # noqa: F401

from benchmark_utils.data import get_device, make_loader
from benchmark_utils.nb_task import download_study, load_task, require_prepared

# BIDS labels from the corpus metadata (emg2pose_metadata.csv): the users of
# the ``user_stage`` test rows, and the users with training recordings.
TEST_SUBJECTS = [
    '02', '03', '05', '08', '20', '28', '44', '49', '55', '56', '84', '89',
    '102', '111', '134', '137', '141', '180', '183', '193',
]
TRAIN_SUBJECTS = [
    '01', '04', '06', '07', '09', '10', '11', '12', '13', '14', '15', '16',
    '17', '18', '19', '21', '22', '23', '24', '25', '26', '27', '30', '31',
    '32', '33', '35', '36', '37', '38', '39', '40', '41', '42', '43', '46',
    '48', '50', '51', '53', '54', '57', '58', '59', '60', '61', '62', '63',
    '64', '65', '66', '67', '69', '70', '71', '72', '73', '74', '75', '76',
    '77', '78', '79', '80', '81', '82', '83', '85', '86', '87', '88', '90',
    '91', '92', '93', '94', '96', '97', '99', '100', '101', '103', '104',
    '105', '106', '107', '108', '109', '110', '112', '113', '114', '115',
    '116', '117', '118', '119', '120', '121', '122', '123', '124', '125',
    '126', '127', '128', '129', '130', '131', '133', '135', '136', '138',
    '139', '140', '142', '143', '146', '147', '148', '149', '150', '151',
    '152', '153', '154', '155', '156', '157', '158', '159', '160', '161',
    '162', '163', '164', '165', '167', '168', '169', '170', '171', '172',
    '174', '175', '176', '177', '179', '181', '182', '184', '185', '186',
    '188', '189', '190', '191', '192',
]


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
        download_study("emg", "pose", self._data_dir(), self._query())
        self._load()

    def _data_dir(self):
        path = get_data_path("emg2pose")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _query(self):
        subjects = TEST_SUBJECTS + TRAIN_SUBJECTS[:self.n_train_subjects]
        labels = [f"Salter2024Emg2pose/{s}" for s in subjects]
        return {"study.source.query": f"subject in {labels!r}"}

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
                **self._query(),
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
