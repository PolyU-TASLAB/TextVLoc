# =============================================================================
# TextVLoc reproducibility environment, based on the official PyTorch CUDA image.
# =============================================================================
# This image installs the Python dependencies from requirements-textvloc.txt and
# compiles a pinned CUDA-enabled COLMAP build into /usr/local.
#


ARG PYTORCH_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
FROM ${PYTORCH_IMAGE}

ARG COLMAP_COMMIT=3e3ecb16
ARG COLMAP_CUDA_ARCHITECTURES="75;80;86;89;90"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1 \
    PYTHONPATH=/app

# COLMAP 源码编译依赖 + 常见 CV / Jupyter 前端的系统库
# libgomp1: 部分数值库
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ca-certificates \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    libboost-test-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    libsuitesparse-dev \
    libcurl4-openssl-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 固定到当前环境使用的 COLMAP 开发版提交，并安装到 /usr/local/bin/colmap。
RUN git clone https://github.com/colmap/colmap.git /tmp/colmap && \
    cd /tmp/colmap && \
    git checkout "${COLMAP_COMMIT}" && \
    cmake -S . -B build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_CUDA_ARCHITECTURES="${COLMAP_CUDA_ARCHITECTURES}" \
        -DCUDA_ENABLED=ON \
        -DGUI_ENABLED=OFF && \
    cmake --build build --target install && \
    /usr/local/bin/colmap -h && \
    rm -rf /tmp/colmap

WORKDIR /app

# 仅复制依赖，便于缓存层
COPY requirements-textvloc.txt ./

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-textvloc.txt && \
    pip install --no-cache-dir pycolmap==3.11.1


# 工程代码；PYTHONPATH=/app 已可让 import hloc 生效
COPY . /app

WORKDIR /app
CMD ["bash"]
