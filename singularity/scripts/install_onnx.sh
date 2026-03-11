#!/bin/bash
set -e

ONNXRUNTIME_VERSION=1.21.0
CMAKE_VERSION=4.0.0
EIGEN_COMMIT="1d8b82b0740839c0de7f1242a3585e3390ff5f33"

export DEBIAN_FRONTEND=noninteractive

# apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     autoconf \
#     automake \
#     libtool \
#     pkg-config \
#     ca-certificates \
#     locales \
#     locales-all \
#     python3 \
#     python3-pip \
#     python3-dev \
#     unzip \
#     wget \
#     git \
#     curl \
#     g++ \
#     cmake \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
    
apt-get update && apt-get install -y --no-install-recommends \
build-essential \
software-properties-common \
autoconf \
automake \
libtool \
pkg-config \
ca-certificates \
locales \
locales-all \
python3-full \
unzip \
wget \
git

apt-get clean

# System locale
# Important for UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

cd /opt

# ONNX Runtime requires CMake 4.0 or higher
wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh
bash cmake-${CMAKE_VERSION}-linux-x86_64.sh \
    --prefix=/usr/local \
    --exclude-subdir \
    --skip-license

rm cmake-${CMAKE_VERSION}-linux-x86_64.sh

# Install ONNX Runtime
git clone --recursive --branch v${ONNXRUNTIME_VERSION} https://github.com/Microsoft/onnxruntime
cd onnxruntime

# Pre install Eigen to avoid gitlab haskey mismatch issue
wget -O eigen.zip "https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_COMMIT}/eigen-${EIGEN_COMMIT}.zip"
unzip eigen.zip -d /opt/eigen-source
rm eigen.zip

./build.sh \
    --allow_running_as_root \
    --cuda_home /usr/local/cuda \
    --cudnn_home /usr/lib/x86_64-linux-gnu/ \
    --use_cuda \
    --config RelWithDebInfo \
    --build_shared_lib \
    --skip_tests \
    --parallel $(nproc) \
    --cmake_extra_defines "FETCHCONTENT_SOURCE_DIR_EIGEN=/opt/eigen-source/eigen-${EIGEN_COMMIT}"

cd build/Linux/RelWithDebInfo
make install

rm -rf /opt/onnxruntime
rm -rf /opt/eigen-source
