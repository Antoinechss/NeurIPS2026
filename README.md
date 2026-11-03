# EMG2Pose: hand pose from wrist EMG

A DataCamp challenge on the public emg2pose corpus (Salter et al., 2024).
Given a 5-second window of 16-channel wrist surface EMG, predict the
trajectories of the 20 joint angles of the hand. Participants train
locally and submit trained weights, which the platform evaluates on a GPU.

The evaluation is a [benchopt](https://benchopt.github.io) benchmark run on
Codabench: participants run locally the same code the server runs.

## Layout

| Path | Content |
|---|---|
| `benchmark/` | the benchopt benchmark: objective, datasets (`EMG2Pose`, `Simulated`), baseline solvers (`MeanPose`, `Ridge`, `Torch-Linear`, `EEGNet`) |
| `solution/` | sample submission (small temporal CNN) and its weights |
| `ingestion_program/`, `scoring_program/` | Codabench programs: run `benchopt run`, parse the results |
| `phases/main/input_data/config.yaml` | phase run config and leaderboard column map |
| `competition.yaml`, `pages/` | Codabench competition and pages |
| `tools/` | bundle builder, Docker image and local Docker run |

## Quick start

```bash
pip install -r requirements.txt
benchopt run benchmark -d Simulated -s solution/submission.py -s Ridge
```

## Get the data

The public data comes from the emg2pose corpus on NEMAR. One command
downloads the public test set and all 158 training users, about 340 GB
(prefer a compute node with many cores):

```bash
benchopt prepare benchmark -d "EMG2Pose[n_train_subjects=158]"
```

The default `EMG2Pose` takes the first 8 training users only. The data goes
to `benchmark/data/`, or to `$BENCHOPT_DATA_HOME`. The private test set, a
sealed cohort of new users, exists on the worker only: the phase runs
`EMG2Pose[mode=full]`, whose prepare reads the cohort's zip from
`<data>/emg2pose_sealed_raw/`. Locally, the default `mode=public` scores the
public set only.

## Build the Codabench bundle

Run `python tools/create_bundle.py`.
