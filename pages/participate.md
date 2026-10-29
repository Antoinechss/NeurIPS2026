# How to participate

## The task

A wristband records **16 channels of surface EMG** (the electrical activity
of the forearm muscles) at 2 kHz, while motion capture tracks the hand.
Given a 5-second EMG window, predict the trajectories of the **20 joint
angles** of the hand over the same window. Decoding hand pose from the wrist
is a building block of neural interfaces for AR/VR and assistive devices.

The data is the public [emg2pose](https://github.com/facebookresearch/emg2pose)
corpus (Salter et al., 2024).

## The metric

**Mean angular MAE**, in degrees (lower is better), averaged over joints,
time steps and windows. Your model predicts angles in **radians**; the
leaderboard converts the error to degrees. The standard deviation of the
per-subject MAE and the R2 are also shown.

## The split

- **Training set**: the emg2pose recordings of up to 158 training users, the
  ones the `EMG2Pose` dataset lists for training. Choose how many you use
  with its `n_train_subjects` parameter (8 by default).
- **Public column** (live feedback): held-out movement stages of 20 users
  absent from the training set.
- **Private / final ranking** (hidden): a private cohort of **new users**.
  Generalizing across people (anatomy, band placement) is the challenge: do
  not overfit the public column.

## What you submit

You train **on your own machine** and submit the trained model. The platform
only evaluates it, on a GPU.

A submission is a folder, zipped: a **`submission.py`** defining
`class Solver(CompetSolver)` and the weight files it loads. All your Python
code must live in `submission.py`: the platform loads it on its own, so it
cannot import other `.py` files you ship.

```python
from benchmark_utils.base_solver import CompetSolver


class Solver(CompetSolver):
    name = "submission"

    def load_model(self, meta):
        # build your torch model, load the weights shipped next to this file
        # from meta["submission_dir"] if they exist, put it on meta["device"]
        ...

    def fit(self, model, train_set):
        # train it (local runs only)
        for X, y, info in self.make_loader(train_set, batch_size=128):
            ...  # X: (B, 16, T) EMG, y: (B, 20, T) angles in radians

    def save_model(self, model, path):
        # write the trained weights into the folder `path`
        ...
```

`train_set` is a torch `Dataset` of single windows. `self.make_loader`
batches it, shuffled, onto the GPU, so you choose the batch size (256 by
default). Evaluation calls `predict` on batches of 256 windows.

Your model must expose `predict(X) -> (B, 20, T')`, angles in radians (a
coarser time axis `T'` is resampled to `T`). Windows are independent: do not
rely on their order. See `solution/submission.py` and `benchmark/solvers/`
for working baselines.

**`fit` must reproduce your weights.** After the competition, the organizers
retrain the top-ranked submissions from their `submission.py` (see the
terms). Keep all the training in `fit`, and write at the top of the file the
`n_train_subjects` you used and how long training took on which GPU.

## The compute

Each submission runs on its own AWS g4dn.2xlarge machine: one NVIDIA T4 GPU
(16 GB), 8 vCPUs and 32 GB of RAM, in the competition's Docker image
(`tommoral/emg2pose-compet:v1`, CUDA 12.6). The platform only evaluates your
trained model, and this must fit in **1 hour**, or the submission fails. The
three Datacamp challenges share these machines, so a submission may wait in
the queue at busy times.

## Run it locally, same code as the server

```bash
pip install -r requirements.txt
benchopt run benchmark -d Simulated -s solution/submission.py   # smoke test
benchopt prepare benchmark -d "EMG2Pose[n_train_subjects=8]"    # large download
# train, score, and write the submission
benchopt run benchmark -d "EMG2Pose[n_train_subjects=8]" -s solution/submission.py
# evaluate it exactly as the server does: load the weights, no training
benchopt run benchmark -d "EMG2Pose[n_train_subjects=8]" \
    -s benchmark/outputs/submission/submission.py -o "EMG-pose[training=False]"
```

A training run writes `benchmark/outputs/<name>.zip`, your `submission.py`
and the weights `save_model` wrote: upload this zip. It also leaves the
folder it zips, `benchmark/outputs/<name>/`. Each training run overwrites
them, so train on `EMG2Pose` last.

The number you get locally on the public split is the number the leaderboard
shows. The benchmark you run *is* the benchmark the server runs.
