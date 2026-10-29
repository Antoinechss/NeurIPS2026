# One image for the competition: it carries the environment and the two
# programs. The benchopt benchmark comes with the phase input_data mounted on
# /app/input_data, so a benchmark change needs no rebuild. Build from the repo
# root:
#
#     docker build -f tools/Dockerfile -t <dockerhub-user>/emg2pose-compet:v1 .
#
# The same image serves the Codabench workers and the participants:
#
#     # stage the data into a host folder (bind-mounted read-only as
#     # /app/data for the runs)
#     # from the phase input_data, the cache off its read-only benchmark
#     docker run -e BENCHOPT_CACHE=/tmp/benchopt \
#         -v <host-data>:/app/data \
#         -v <bundle>/main_phase/input_data:/app/input_data:ro <image> \
#         benchopt prepare /app/input_data/benchmark \
#         --config /app/input_data/config.yaml
#
#     # reproduce the platform evaluation on a submission folder
#     docker run --gpus all -v <host-data>:/app/data \
#         -v <bundle>/main_phase/input_data:/app/input_data \
#         -v <submission>:/app/ingested_program -v <results>:/app/output \
#         <image> python3 /compet/ingestion_program/ingestion.py
#
# Python 3.12 base (neuralset needs >= 3.12); torch comes from the PyTorch
# wheel index, whose wheels bundle the CUDA runtime, so GPU works through the
# nvidia container runtime without a CUDA base image.
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# torch with an explicit CUDA channel: cu126 runs on any R525+ driver (cu128+
# for Blackwell GPUs, cpu for CI). Installed first so requirements.txt keeps
# this build. torchvision and torchaudio come from the same channel: the
# neuro stack pulls them, and PyPI builds do not match a channel torch.
ARG TORCH_CHANNEL=cu126
RUN pip install -U pip && pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/${TORCH_CHANNEL}

# All Python dependencies a submission may import: nothing is installed at
# submission time.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# neuralbench reads its settings from this file instead of a user config.
COPY tools/neuralbench_config.json /compet/neuralbench_config.json

COPY ingestion_program /compet/ingestion_program
COPY scoring_program /compet/scoring_program

# Data is bind-mounted read-only at /app/data by the compute worker; the
# phase config and benchmark land on /app/input_data. Extraction runs serially: parallel
# extractors corrupt the study's timeline cache on many-core hosts.
ENV BENCHOPT_DATA_HOME=/app/data \
    NEURALBENCH_CONFIG=/compet/neuralbench_config.json \
    NEURALBENCH_EXTRACT_CPUS=1
VOLUME /app/data

# Keep the participant trace to what their submission does.
ENV PYTHONWARNINGS=ignore::FutureWarning \
    MNE_LOGGING_LEVEL=ERROR \
    TQDM_DISABLE=1

# Fail the build if the stack is incoherent.
RUN python -c "import benchopt, pandas, sklearn, torch, braindecode, neuralbench"
