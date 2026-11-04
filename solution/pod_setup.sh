#!/usr/bin/env bash
# One-time setup of a RunPod pod. Run it once, on the first pod (CPU or GPU):
#   cd /workspace && git clone -b feat/vemg2pose-plus https://github.com/Antoinechss/NeurIPS2026.git codabench
#   bash /workspace/codabench/solution/pod_setup.sh
# Everything (Python, venv, caches, data) lives on the network volume /workspace,
# so later pods reuse it: there, only run `source /workspace/env.sh`.
set -euo pipefail

W=/workspace
[ -d "$W/codabench" ] || { echo "Clone the repo into $W/codabench first"; exit 1; }

# tmux keeps long runs alive if the connection drops.
command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; }

# Environment for every session. uv's Python and cache go on the volume too:
# by default they sit in the pod's home, which is lost with the pod.
cat > "$W/env.sh" <<'EOF'
export UV_PYTHON_INSTALL_DIR=/workspace/uv-python
export UV_CACHE_DIR=/workspace/.uv-cache
export BENCHOPT_DATA_HOME=/workspace/data
# Same as the organisers' Docker image: parallel extraction corrupts the
# data index on many-core machines.
export NEURALBENCH_EXTRACT_CPUS=1
export PATH="/workspace/bin:$PATH"
[ -f /workspace/venv/bin/activate ] && . /workspace/venv/bin/activate
cd /workspace/codabench
EOF
. "$W/env.sh"

# uv: fast installer that also downloads Python 3.12 (neuralset needs >= 3.12).
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR="$W/bin" INSTALLER_NO_MODIFY_PATH=1 sh
fi
[ -d "$W/venv" ] || uv venv -p 3.12 "$W/venv"
. "$W/venv/bin/activate"

# CUDA 12.6 build of torch, like the evaluation Docker image. torchaudio
# (braindecode) and torchvision (torchmetrics) must come from the same index
# as torch: PyPI builds of them fail to load ("torchvision::nms does not exist").
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements.txt
TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])")
uv pip install --reinstall-package torchvision --reinstall-package torchaudio \
    "torch==$TORCH" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

python -c "import torch, torchvision.ops, braindecode, neuralset, neuraltrain; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
echo "Setup OK. In every new terminal: source /workspace/env.sh"
