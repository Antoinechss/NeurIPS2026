# Experiment log

Public MAE = `objective_angular_mae_public` (degrees, lower is better); equals the leaderboard public column on real data.
Simulated scores are random-data sanity checks only: never compare them with real-data numbers.

| # | Date | File + params | Data / n_train_subjects | Hardware | Train time | Public MAE (°) | Subject std | Inference time | Takeaway |
|---|---|---|---|---|---|---|---|---|---|
| 0a | 2026-11-04 | `solution/vemg2pose_plus.py[max_steps=30,warmup_steps=5]` | Simulated | Apple M1 CPU | ~1 min | 18.947 | 0.562 | – | Smoke test: trains, exports `vemg2pose_plus.zip` (24 MB weights). |
| 0b | 2026-11-04 | `benchmark/outputs/vemg2pose_plus/submission.py`, `-o "EMG-pose[training=False]"` | Simulated | Apple M1 CPU | – (no training) | 18.947 | 0.562 | – | Re-eval matches 0a to every digit: save/load works. |
| 1a | 2026-11-05 | `MeanPose` | EMG2Pose / 8 | RTX 6000 Ada (RunPod) | seconds | 19.273 | 2.021 | – | Floor on real data. |
| 1b | 2026-11-05 | `Ridge` | EMG2Pose / 8 | RTX 6000 Ada (RunPod) | seconds | 19.273 | 2.021 | – | Equals the floor: raw EMG sample by sample carries no linear pose information. |
| 1c | 2026-11-05 | `solution/vemg2pose_plus.py[max_steps=1000,warmup_steps=100]` | EMG2Pose / 8 | RTX 6000 Ada (RunPod) | ~9 min (~125 steps/min after compile) | 19.957 | 2.437 | – | **Worse than the floor**: train L1 0.24 rad (13.7°), so it fits the 8 users and does not transfer. Check at 158 users before the long run. |
| 1d | 2026-11-05 | export of 1c, `-o "EMG-pose[training=False]"` | EMG2Pose / 8 | RTX 6000 Ada (RunPod) | – | 19.957 | 2.437 | – | Identical to 1c: save/load works on real data. |
| 1e | 2026-11-05 | export of 1c with `tta=False`, `-o "EMG-pose[training=False]"` | EMG2Pose / 8 | RTX 6000 Ada (RunPod) | – | 20.414 | 2.466 | – | TTA over ±1 rotations is worth ~0.46° (19.96 with it). Keep `tta=True`. |
| 2a | 2026-11-05 | `solution/vemg2pose_plus.py[max_steps=3000,warmup_steps=300]` | EMG2Pose / 158 | RTX 6000 Ada (RunPod) | 24 min train (~128 steps/min, no I/O slowdown) + ~25 min data loading | 17.109 | 1.510 | – | **Beats the floor by 2.2°** and halves nothing but narrows the subject spread (1.51 vs 2.02): users, not steps, were the 8-user problem. Go for the long run. |
