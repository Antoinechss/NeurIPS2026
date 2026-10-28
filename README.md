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

Real data (large download, prefer a compute node):
`BENCHOPT_DATA_HOME=<data> benchopt prepare benchmark -d EMG2Pose`. The
private test set is staged on the worker only, in `<data>/emg2pose_sealed/`
(`X.npy`, `y.npy`, `subject.npy`); without it, a run scores the public set
only.

Build the Codabench bundle with `python tools/create_bundle.py`.
