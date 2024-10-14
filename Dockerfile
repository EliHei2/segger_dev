FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git \
    wget \
    tmux \
    vim \
    htop \
    zip \
    unzip \
    build-essential \
    python3 \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir debugpy

WORKDIR /workspace

RUN git clone https://github.com/EliHei2/segger_dev.git /workspace/segger_dev && \
    pip install -e "/workspace/segger_dev[cuda12,rapids12,cupy12,faiss]"

EXPOSE 5678

ENV PYTHONPATH=/workspace/segger_dev/src:$PYTHONPATH
