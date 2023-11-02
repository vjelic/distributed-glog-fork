#!/bin/bash

set -ex

function install_ucx() {
  set -ex
  git clone --recursive https://github.com/openucx/ucx.git
  pushd ucx
  git checkout ${UCX_COMMIT}
  git submodule update --init --recursive

  ./autogen.sh
  ./configure --prefix=/opt/ucx       \
      --enable-mt                     \
      --with-rocm=/opt/rocm           \
      --enable-profiling              \
      --enable-stats
  time make -j
  sudo make install

  popd
  rm -rf ucx
}


install_ucx
# install_ucc