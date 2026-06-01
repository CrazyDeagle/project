# syntax=docker/dockerfile:1.7

# Reproducible CUDA-enabled image for SilexCode training and development.
# The base image ships CUDA 12.1 + cuDNN 9 on Ubuntu 22.04, which matches the
# PyTorch CUDA wheels used in development.
ARG CUDA_IMAGE=nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        ninja-build \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/silexcode

# Install PyTorch with CUDA support before copying the source so the layer is
# cacheable across iterations on the project sources.
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
        "torch>=2.1" \
    && python -m pip install "numpy>=1.24" "pytest>=7.4" "ruff>=0.5.0"

COPY . .

# Build the CUDA extension in-place. This is the slow step; cache it on the
# layer above when iterating locally.
RUN pip install -e . --no-build-isolation

CMD ["python", "-m", "pytest", "-q", "-m", "not cuda"]
