# Use the latest Ubuntu image
FROM ubuntu:latest

# Install essential packages
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y --no-install-recommends gnupg && \
    apt-key update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update  && \
    apt-get install -y  \
    python3 \
    python3-pip \
    python3-venv \
    git \
    git-lfs \
    vim \
    curl \
    python3-dev \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    tcl8.6-dev \
    tk8.6-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libxcb1-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Git LFS and initialize it
RUN git lfs install

# Clone your GitHub repository using an ARG for the personal access token
ARG GITHUB_TOKEN
WORKDIR /home/ubuntu
RUN git clone -b chris-gpnpu-mode https://${GITHUB_TOKEN}@github.com/quadric-io/onnxruntime.git

# Set the working directory
WORKDIR /home/ubuntu

# Create a virtual environment and install dependencies and create fp32 conv graph
RUN python3 -m venv .venv && \
    . /home/ubuntu/.venv/bin/activate && \
    pip3 install --upgrade pip && \
    pip3 install -r /home/ubuntu/fp16_exploration/fp16_ort/requirements.txt

# onnxruntime
WORKDIR /home/ubuntu
RUN git clone -b  https://${GITHUB_TOKEN}@github.com/quadric-io/onnxruntime.git
WORKDIR /home/ubuntu/onnxruntime
RUN . /home/ubuntu/.venv/bin/activate && \
./build.sh \
--config RelWithDebInfo \
--build_shared_lib \
--parallel \
--compile_no_warning_as_error \
--allow_running_as_root \
--build_wheel

RUN . /home/ubuntu/.venv/bin/activate && \
 pip3 install $(find . -name "*.whl")

RUN sysctl -w kernel.yama.ptrace_scope=0

SHELL ["/bin/bash", "-c"]
