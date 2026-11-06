# CLAUDE.md — EMG2Pose competition

Antoine (M2 Data Science, IP Paris) is competing in **EMG-to-Pose**. The goal is a top-ranked submission, built **step by step**, with Antoine understanding every component. You are both engineer and tutor.

## 1. The task in one paragraph
A wristband records **16 channels of surface EMG** (forearm muscle electrical activity) at **2 kHz**. From a **5-second window** (16 × 10 000 samples), predict the trajectories of the **20 hand joint angles** over the same window, in **radians**. The data comes from the public emg2pose corpus (Salter et al., 2024, Meta).

## 2. The competition
Codabench "Datacamp 2026 – EMG2Pose" (course grade, M2), **29 Oct 2026 → 1 Feb 2027**. Contract: this repo (below).
Antoine is **not** entering NeurIPS 2026 Track 04 (same data, separate competition): ignore its deadlines and never suggest registering.

## 3. Scoring
- **Metric: mean angular MAE in degrees**, lower is better. It is averaged over joints, time steps and windows. Steps with a NaN label are ignored. The per-subject MAE std and R² are also shown.
- **Public column**: held-out movement stages of 20 users who are absent from training. The local score on the public split **equals** the leaderboard number, so evaluate locally and save submissions.
- **Private column (final ranking)**: a sealed cohort of **57 new users** (2 887 recordings). The real problem is **cross-user generalization** (anatomy, band placement). Do not overfit the public column.

## 4. Submission contract
- A zip containing `submission.py` and its weight files. **All code lives in `submission.py`**: no importing sibling `.py` files.
- `class Solver(CompetSolver)` implements `load_model(meta)`, `fit(model, train_set)`, `save_model(model, path)`.
- `model.predict(X)`: `X` is `(B, 16, T)` float, and the output is `(B, 20, T')` in radians. A coarser `T'` is nearest-resampled. Windows are independent, so keep no state between calls.
- `meta` has `n_chans, n_times, n_joints, sfreq, device, num_workers, submission_dir`.
- **Evaluation**: one AWS g4dn.2xlarge (**T4 16 GB**, 8 vCPU, 32 GB RAM), Docker `tommoral/emg2pose-compet:v1` (CUDA 12.6). Inference must finish in **1 h**. No package installs: only what is in `requirements.txt` (torch, braindecode, sklearn, numpy, pandas, neural* stack).
- **5 submissions per day.**
- `fit` **must reproduce the weights**. The organisers retrain the top submissions in **≤ 12 h on one 32 GB V100**. If the retrained score is clearly worse, the submission is removed. All training goes in `fit`. Hyper-parameters must be the **defaults** in `parameters`. Write `n_train_subjects`, the training time and the GPU at the top of the file.

## 5. Hard rules (never break)
1. **Train only on the training users of `EMG2Pose`.** Never use the public test users, other emg2pose data or any other EMG data. Pretrained models are allowed only if they were not trained on EMG.
2. **Never try to read, infer or reverse-engineer the hidden/private data.** For the course this means grade 0 and failing the master.
3. Respect the submission limit and the time budgets.
4. The code must be Antoine's team's own work. Public libraries are fine.

## 6. Repo map
| Path | What |
|---|---|
| `benchmark/objective.py` | metric and evaluation loop (do not modify) |
| `benchmark/datasets/emg2pose.py` | real data (`n_train_subjects`, `num_workers`, `mode`) |
| `benchmark/datasets/simulated.py` | fake data for smoke tests (runs on CPU) |
| `benchmark/benchmark_utils/base_solver.py` | `CompetSolver` contract and export |
| `benchmark/solvers/` | baselines: `mean_pose`, `ridge_pose`, `torch_linear`, `eegnet_pose`, `vemg2pose` |
| `solution/submission.py` | organisers' sample (small CNN) |
| `solution/vemg2pose_plus.py` | causal candidate: 15.98° public (25k steps, 158 users), submitted |
| `solution/posebinet.py` | bidirectional candidate (offline decoding), under test |
| `solution/RUNBOOK.md` | GPU rental, data, calibration and submit commands |
| `EXPERIMENTS.md` | experiment log (create it if missing) |

## 7. Commands
```bash
# smoke test (CPU OK)
benchopt run benchmark -d Simulated -s "solution/vemg2pose_plus.py[max_steps=30,warmup_steps=5]" -n 1 --no-plot
# data (big disk, many CPUs; 158 users ≈ 340 GB)
export BENCHOPT_DATA_HOME=/workspace/data
benchopt prepare benchmark -d "EMG2Pose[n_train_subjects=8]"
# train + score public + export benchmark/outputs/<name>.zip
benchopt run benchmark -d "EMG2Pose[n_train_subjects=8,num_workers=16]" -s solution/vemg2pose_plus.py
# evaluate the export exactly like the platform (no training)
benchopt run benchmark -d "EMG2Pose[n_train_subjects=8]" -s benchmark/outputs/vemg2pose_plus/submission.py -o "EMG-pose[training=False]"
```
Override parameters inline: `-s "solution/file.py[param=value,...]"`. Each training run overwrites `benchmark/outputs/<name>/`, so copy any zip you want to keep. Results are in `benchmark/outputs/*.parquet`, column `objective_angular_mae_public`.

## 8. Roadmap: progressive, one idea at a time
Each stage has a **concept** (explain it to Antoine first) and an **experiment** (log it). Move on only after the result is logged and understood.

| Stage | Concept to teach | Experiment |
|---|---|---|
| 0. Floor | What the metric means. Predicting the mean pose. | `MeanPose`, `Ridge`, sample CNN on 8 users: get the reference numbers |
| 1. Reference model | VEMG2Pose: conv encoder (TDS blocks) → LSTM decoder rolling out the pose at 50 Hz, "hybrid" position→velocity output, left context of 1 790 samples | `benchmark/solvers/vemg2pose.py` on 8 users |
| 2. Data scaling | Cross-user generalization needs many users | 8 → 32 → 158 users, same recipe; plot MAE vs #users |
| 3. Augmentation | Invariances: band rotation (ring of 16 electrodes), per-channel/global gain (skin impedance, placement), noise | `vemg2pose_plus` with `max_shift=0/1`, then gain/noise on/off |
| 4. Optimization | AdamW, warmup+cosine, grad clipping, EMA weights, AMP on the encoder only (fp16 would make the velocity integration drift) | ablate EMA and the schedule against the baseline's Adam |
| 5. Inference tricks | Test-time augmentation (average over rotations), seed ensembles | `tta=True/False` (re-eval only), `n_models=2` |
| 6. Advanced (pick by evidence) | per-user normalisation-free features, wider/longer models, multi-resolution input, loss on velocities, SWA, distillation of an ensemble into one model | one at a time, only if stages 2–5 have converged |

Stop and discuss with Antoine before stage 6 and before any final submission.

## 9. Working rules for the agent
- **One change per experiment.** Same seed and same `n_train_subjects` within a comparison. Record in `EXPERIMENTS.md`: date, file + params, n_train_subjects, GPU, train time, public MAE, subject std, inference time, one-line takeaway.
- **Explain before you code.** For each new component, give Antoine 3–6 lines covering what it is, why it should help *cross-user*, and how we will know it worked. Keep it plain and concise, no hedging.
- Validate every new solver with the Simulated smoke test, then with the `training=False` re-eval. Both must give the **same** score (this proves save/load works).
- Before submitting, check: (a) inference on the public split is well under the T4 hour, remembering that private ≫ public in size and a T4 is about 3–5× slower than an A100; (b) the training time projects to ≤ 12 h on a V100 (budget ≤ ~4 h on A100); (c) the defaults in `parameters` are the final ones; (d) the header records the subjects, time and GPU.
- Prefer improvements that target **unseen users**. Treat a public-column gain from heavy tuning with suspicion.
- Never touch `objective.py`, the scoring or ingestion programs, or anything that reads test data for training.
- Commits: small and descriptive, on a feature branch, not `main`.

## 10. Status
- `vemg2pose_plus` (25k steps, 158 users, 3.2 h on an RTX 6000 Ada): **15.98° public**, re-eval identical. This matches the published VEMG2Pose result for held-out users and stages (15.6–15.8°). The best published number for that setting is 14.6° (arXiv 2603.08212, multi-task tracking + regression). Leaderboard entries at 4–6° are far below anything published, even below models given the true starting pose (~10°), so do not chase them.
- `posebinet`: the VEMG2Pose encoder with its receptive field centred, plus a bidirectional LSTM and an absolute-angle read-out. It uses the EMG after t, which the offline scoring allows. 3000-step screen: **16.53°** (256 channels) / 16.49° (384) vs 17.11° for the causal model: -0.6°. Training is data-bound (GPU waits for data 57% of the time), so `n_models=2` members are now trained side by side on the same batches (default). Simulated smoke test and re-eval pass for 1 and 2 members; `member=i` re-scores one member alone. Next: full 25k-step run with 2 members, once the Codabench Duration of the 15.98° submission confirms the T4 time budget.
