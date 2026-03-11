#!/bin/bash

export JANA_HOME=/opt/JANA2
export PATH=$JANA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$JANA_HOME/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$JANA_HOME/cmake/lib/JANA:$CMAKE_PREFIX_PATH

apt-get update -y && apt-get upgrade -y
apt install -y --no-install-recommends \
    build-essential \
    ca-certificates cmake wget unzip git \
    python3-dev \
    libx11-dev libxpm-dev libxft-dev libxext-dev

apt-get autoremove -y && apt-get clean -y

git clone https://github.com/JeffersonLab/JANA2 /opt/JANA2
cd /opt/JANA2
mkdir build
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=`pwd`
cmake --build build --target install -j$(nproc)

