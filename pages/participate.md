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

- **Training set**: emg2pose recordings of a subset of the users.
- **Public column** (live feedback): held-out movement stages of users seen
  elsewhere in the data.
- **Private / final ranking** (hidden): a private cohort of **new users**.
  Generalizing across people (anatomy, band placement) is the challenge: do
  not overfit the public column.

## What you submit

A single **`submission.py`** defining `class Solver(CompetSolver)`. All your
Python code must live in this one file: the platform loads it on its own, so
it cannot import other `.py` files you ship. You may add non-code files next
to it (configs, pretrained weights) and read them from
`meta["submission_dir"]`.

```python
from benchmark_utils.base_solver import CompetSolver


class Solver(CompetSolver):
    name = "submission"

    def get_model(self, meta):
        # build your (untrained) torch model on meta["device"]
        ...

    def fit(self, model, train_loader):
        # optional: train it (default: model.fit(train_loader))
        for X, y, info in train_loader:
            ...  # X: (B, 16, T) EMG, y: (B, 20, T) angles in radians
```

Your trained model must expose `predict(X) -> (B, 20, T')`, angles in
radians (a coarser time axis `T'` is resampled to `T`). Windows are
independent: do not rely on their order. The platform **trains your code on
a GPU** within the time limit (1 hour), then evaluates it. See
`solution/submission.py` and `benchmark/solvers/` for working baselines.

## Run it locally, same code as the server

```bash
pip install -r requirements.txt
benchopt run benchmark -d Simulated -s solution/submission.py   # smoke test
benchopt prepare benchmark -d EMG2Pose                          # large download
benchopt run benchmark -d EMG2Pose -s solution/submission.py
```

The number you get locally on the public split is the number the leaderboard
shows. The benchmark you run *is* the benchmark the server runs.
