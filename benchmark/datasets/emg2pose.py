"""EMG2Pose: wrist surface EMG to hand pose (Salter et al. 2024).

Wraps the neuralbench ``emg/pose`` task (see ``benchmark_utils.nb_task``):
5-s windows of 16-channel wrist EMG at 2 kHz, target = 20 hand joint-angle
trajectories in radians, with the corpus' predefined split.

    benchopt prepare benchmark -d EMG2Pose

- **Training set**: the train split of the first ``n_train_subjects`` of
  the 158 training users. The full corpus is ~340 GB; the subset sizes the
  download and the training. The platform does not train, so it stages a
  single training user.
- **public** test set: the corpus' ``user_stage`` test rows, from 20 users
  absent from training, on held-out movement stages.
- **private** test set, ``mode="full"`` only: the sealed 2026 cohort of new
  users, a BIDS tree in the public corpus' layout. Its zip sits in
  ``get_data_path("emg2pose_sealed_raw")`` on the staging worker only, and
  goes through the same task pipeline as the public set. Participants run
  ``mode="public"``, which scores the public set alone.

``prepare`` downloads these subjects only and runs the extraction once: it
caches next to the data, so runs only read warm caches. Both steps are
idempotent. The download opens one connection per CPU, and NEMAR throttles
each connection, so prepare on a machine with many cores.
"""

import json
import typing as tp
import zipfile

import numpy as np
from pathlib import Path

from benchopt import BaseDataset
from benchopt.config import get_data_path

# Hard requirement, imported at module level so benchopt reports the
# dataset as not installed when it is missing.
import neuralbench  # noqa: F401
from neuralfetch.studies.salter2024emg2pose import Salter2024Emg2pose

from benchmark_utils.data import get_device, make_loader
from benchmark_utils.nb_task import download_study, load_task, require_prepared

_nemar_load_raw = Salter2024Emg2pose._load_raw


# Named like the method it replaces: neuralset stores loaders by name.
def _load_raw(self, timeline):
    """Read a recording with its joint angles in radians.

    NEMAR nm000281 >= 1.0.4 relabelled most BDF joint channels from uV to rad
    without rescaling the samples, so MNE reads them as radians x 1e6.
    """
    raw = _nemar_load_raw(self, timeline)
    if np.nanmax(np.abs(raw.get_data(picks="misc"))) > 1e3:
        raw.apply_function(lambda data: data * 1e-6, picks="misc")
    return raw


Salter2024Emg2pose._load_raw = _load_raw

SEALED_ZIP = "secret-test-set-bids.zip"
SEALED_ROOT = "secret-test-set-bids"


class Emg2poseSealed2026(Salter2024Emg2pose):
    """The sealed 2026 cohort: the public corpus' BIDS layout, whose
    ``scans.tsv`` already marks every recording as a ``user_stage`` test."""

    aliases: tp.ClassVar[tuple[str, ...]] = ()
    description: tp.ClassVar[str] = (
        "The sealed 2026 emg2pose cohort, staged on the worker.")
    # The study checks the BIDS tree holds this many recordings: the 2887 of
    # the cohort, plus the two copies below.
    _info: tp.ClassVar = Salter2024Emg2pose._info.model_copy(
        update=dict(num_timelines=2889, num_subjects=57))

    def _download(self, overwrite=False):
        raise FileNotFoundError(
            f"The sealed cohort is not downloaded: unzip {SEALED_ZIP} into "
            f"{self.path}.")

    @property
    def bids_root(self):
        return self.path / SEALED_ROOT

    def iter_timelines(self):
        """The cohort, all test, plus its longest recording copied into
        train and val: neuralbench cannot build an empty split."""
        timelines = list(super().iter_timelines())
        longest = max(timelines, key=lambda t: json.loads(
            Path(t["path"]).with_suffix(".json").read_text())[
                "RecordingDuration"])
        yield from timelines
        for split in ("train", "val"):
            yield {**longest, "split": split}


# BIDS labels from the corpus metadata (emg2pose_metadata.csv): the users of
# the ``user_stage`` test rows, and the users with training recordings.
TEST_SUBJECTS = [
    '02', '03', '05', '08', '20', '28', '44', '49', '55', '56', '84', '89',
    '102', '111', '134', '137', '141', '180', '183', '193',
]
# neuralbench builds a val loader and fails on an empty split: keep the
# smallest validation user.
VAL_SUBJECTS = ['166']
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
        # Evaluation batch size; solvers batch the training set themselves.
        "batch_size": [256],
        # Dataloader workers, for evaluation and passed on to solvers; raise
        # it from the phase config on a worker with spare cores.
        "num_workers": [0],
        # "full" adds the sealed private test set, on the staging worker.
        "mode": ["public"],
    }

    # Loader settings do not change the prepared data.
    prepare_cache_ignore = ("batch_size", "num_workers")

    def prepare(self):
        download_study("emg", "pose", self._data_dir(), self._query())
        self._load()
        if self.mode == "full":
            study = self._sealed_dir() / Emg2poseSealed2026.__name__
            if not (study / SEALED_ROOT).is_dir():
                archive = get_data_path("emg2pose_sealed_raw") / SEALED_ZIP
                if not archive.is_file():
                    raise SystemExit(
                        f"[EMG2Pose] mode=full needs the sealed cohort at "
                        f"{archive}: stage.sh fetches it on the worker.")
                with zipfile.ZipFile(archive) as z:
                    z.extractall(study, [n for n in z.namelist()
                                         if n.startswith(f"{SEALED_ROOT}/")])
            self._load_sealed()

    def _data_dir(self):
        path = get_data_path("emg2pose")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _sealed_dir(self):
        path = get_data_path("emg2pose_sealed")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _query(self):
        subjects = (TEST_SUBJECTS + VAL_SUBJECTS
                    + TRAIN_SUBJECTS[:self.n_train_subjects])
        labels = [f"Salter2024Emg2pose/{s}" for s in subjects]
        return {"study.source.query": f"subject in {labels!r}"}

    def _load(self):
        cache = str(self._data_dir() / "cache")
        return load_task(
            "emg", "pose",
            data_dir=self._data_dir(),
            seed=self.get_seed(),
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
        datasets, meta = self._load()
        # A cache built without the _load_raw fix serves radians x 1e6.
        y = np.asarray(datasets["test"][0][1], dtype=float)
        if np.nanmax(np.abs(y)) > 2 * np.pi:
            raise RuntimeError(
                f"Joint angles up to {np.nanmax(np.abs(y)):.3g}, not radians: "
                f"delete {self._data_dir() / 'cache'} and prepare again."
            )
        test_sets = {"public": datasets["test"]}
        if self.mode == "full":
            require_prepared("emg", "pose", self._sealed_dir(),
                             self._sealed_overrides())
            test_sets["private"] = self._load_sealed()
        test_sets = {
            name: make_loader(ds, batch_size=self.batch_size, device=device,
                              num_workers=self.num_workers)
            for name, ds in test_sets.items()
        }
        return dict(train_set=datasets["train"], test_sets=test_sets,
                    device=device, num_workers=self.num_workers, **meta)

    def _sealed_overrides(self):
        cache = str(self._sealed_dir() / "cache")
        return {
            "study.source.name": Emg2poseSealed2026.__name__,
            "target.infra.cluster": None,
            "target.infra.folder": cache,
            "target.infra.permissions": None,
            "study.source.timelines.infra.backend": "Cached",
            "study.source.timelines.infra.folder": cache,
        }

    def _load_sealed(self):
        datasets, _ = load_task("emg", "pose", data_dir=self._sealed_dir(),
                                seed=self.get_seed(),
                                overrides=self._sealed_overrides())
        return datasets["test"]
