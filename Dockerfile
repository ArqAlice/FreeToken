FROM nvidia/cuda:13.0.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:/root/.local/bin:/opt/venv/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    build-essential \
    ninja-build \
    cmake \
    git \
    curl \
    ca-certificates \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /opt/freetoken

COPY . /opt/freetoken

RUN uv venv /opt/venv --python python3.12

RUN uv pip install \
    --python /opt/venv/bin/python \
    -e ".[accel]"

RUN /opt/venv/bin/ft --version

EXPOSE 1919

ENTRYPOINT ["/opt/venv/bin/ft"]