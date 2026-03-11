#!/bin/bash

set -e

export LANG="C.UTF-8"

ROOT_BIN="root_v6.34.00.Linux-ubuntu24.04-x86_64-gcc13.2.tar.gz"

package=(
    ca-certificates cmake curl davix-dev dcap-dev fonts-freefont-ttf g++ gcc gfortran git libafterimage-dev libcfitsio-dev libfcgi-dev libfftw3-dev libfreetype6-dev libftgl-dev libgfal2-dev libgif-dev libgl2ps-dev libglew-dev libglu-dev libgraphviz-dev libgsl-dev libjpeg-dev liblz4-dev liblzma-dev libmysqlclient-dev libpcre2-dev libpcre3-dev libpng-dev libpq-dev libsqlite3-dev libssl-dev libtbb-dev libtiff-dev libx11-dev libxext-dev libxft-dev libxml2-dev libxpm-dev libxxhash-dev libvdt-dev libz-dev libzstd-dev locales make nlohmann-json3-dev protobuf-compiler python3-dev python3-numpy srm-ifce-dev unixodbc-dev python-is-python3 xrootd-plugins wget unzip
)
apt-get update -y && apt-get upgrade -y
apt install -y --no-install-recommends ${package[@]} \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*


mkdir -p /opt/root
cd /opt/root
ln -sf /usr/share/zoneinfo/UTC /etc/localtime
wget https://root.cern/download/${ROOT_BIN}
tar --strip-components=1 -xzvf ${ROOT_BIN}
rm -f ${ROOT_BIN}
echo /opt/root/lib >> /etc/ld.so.conf
ldconfig
yes | unminimize


export ROOTSYS=/opt/root
export PATH=$ROOTSYS/bin:$PATH
export PYTHONPATH=$ROOTSYS/lib:$PYTHONPATH
export CLING_STANDARD_PCH=none
