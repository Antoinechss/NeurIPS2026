# VEMG2Pose+ runbook (rented GPU)

## 0. Machine
- 1x A100 80GB or H100 (cheap per-hour on RunPod / Lambda / vast.ai), CUDA 12.x driver
- **>= 32 vCPUs** (the NEMAR download opens one throttled connection per CPU; extraction and loading also use them)
- >= 128 GB RAM, **>= 1 TB disk** (340 GB raw + neuralbench caches)
- Python 3.12 (neuralset requires >= 3.12)

## 1. Setup
The code lives in `codabench_emg2pose`, branch `feat/vemg2pose-plus`, pushed to the remote `mine`
(github.com/Antoinechss/NeurIPS2026; its `main` holds unrelated Meta code, so always clone the branch).
```bash
git clone -b feat/vemg2pose-plus https://github.com/Antoinechss/NeurIPS2026.git codabench && cd codabench
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
export BENCHOPT_DATA_HOME=/workspace/data   # on the big disk
```

## 2. Smoke test (1 min)
```bash
benchopt run benchmark -d Simulated -s "solution/vemg2pose_plus.py[max_steps=30,warmup_steps=5]" -n 1 --no-plot
# then the platform-style eval of the export (must give the same score):
benchopt run benchmark -d Simulated -s benchmark/outputs/vemg2pose_plus/submission.py -o "EMG-pose[training=False]" -n 1 --no-plot
```

## 3. Data (hours; run in tmux)
```bash
benchopt prepare benchmark -d "EMG2Pose[n_train_subjects=158]"
```
Start with `n_train_subjects=8` in parallel if you want to iterate before the full download ends.

## 4. Calibrate the step budget
The organisers retrain top submissions with `fit` in **<= 12 h on one 32 GB V100**.
Measure throughput on your GPU:
```bash
benchopt run benchmark -d "EMG2Pose[n_train_subjects=158,num_workers=16]" \
  -s "solution/vemg2pose_plus.py[max_steps=1000]"
```
Read min/step from the log. Pick `max_steps` so the run takes **<= 4 h on A100 / <= 3 h on H100**
(conservative V100 margin; `max_hours=11` is the hard safety stop). Set it as the **default** in
`parameters` — the retrain uses the file's defaults.

## 5. Experiments (local score on the public split == leaderboard public column)
In order of expected gain:
1. 158 subjects vs 8 (baseline config)          -> expect the largest drop in MAE
2. `max_shift=0` vs `1` (rotation aug on/off)
3. `tta=False` vs `True` (free at train time; just re-evaluate)
4. `n_models=2` if the budget allows (halves steps per member)
Do not tune many knobs on the public column: the private set is 57 new users.

## 6. Check inference time like the platform (T4, 1 h limit)
```bash
benchopt run benchmark -d "EMG2Pose[n_train_subjects=158]" \
  -s benchmark/outputs/vemg2pose_plus/submission.py -o "EMG-pose[training=False]"
```
A T4 is ~3-5x slower than an A100 at inference. If the eval on public takes more than ~8 min on
your GPU, set `tta=False` (TTA triples the batch) — the private cohort (2887 recordings) is much
larger than the public set.

## 7. Submit
Upload `benchmark/outputs/vemg2pose_plus.zip`. Fill the training record at the top of the file
first (n_train_subjects, hours, GPU) and re-run export, or edit the copy in the zip.
