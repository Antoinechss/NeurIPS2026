# Experiment log

Public MAE = `objective_angular_mae_public` (degrees, lower is better); equals the leaderboard public column on real data.
Simulated scores are random-data sanity checks only: never compare them with real-data numbers.

| # | Date | File + params | Data / n_train_subjects | Hardware | Train time | Public MAE (°) | Subject std | Inference time | Takeaway |
|---|---|---|---|---|---|---|---|---|---|
| 0a | 2026-11-04 | `solution/vemg2pose_plus.py[max_steps=30,warmup_steps=5]` | Simulated | Apple M1 CPU | ~1 min | 18.947 | 0.562 | – | Smoke test: trains, exports `vemg2pose_plus.zip` (24 MB weights). |
| 0b | 2026-11-04 | `benchmark/outputs/vemg2pose_plus/submission.py`, `-o "EMG-pose[training=False]"` | Simulated | Apple M1 CPU | – (no training) | 18.947 | 0.562 | – | Re-eval matches 0a to every digit: save/load works. |
