#!/bin/bash

# fail at first error
set -e

# clone source repos
# pushd dask_src
# git clone git@github.com:AMD-AI/RMM.git
# git clone --recurse-submodules --branch cuda_array_interface_dev git@github.com:AMD-AI/cupy.git
# git clone --branch dev-rocm git@github.com:ROCm/distributed.git
# git clone --branch rocm-dev-pyamdsmi-with-cupy git@github.com:AMD-AI/ucx-py.git
# popd

# set up config parameters for the Docker image here
# DASK_ROCM_ARCH is used for building cupy-rocm package
# CMAKE_VERSION is for building RMM
# UCX_COMMIT is for building ucx for rocm: common/install_ucx.sh
# GITHUB_USER/_PASS are temporary measure for RMM package that requires access to private repo.

# create image tag name using build environment
UBUNTU_VERSION=20.04
ANACONDA_PYTHON_VERSION=3.9
ROCM_VERSION=5.6
BUILD_ENVIRONMENT=rocm${ROCM_VERSION}_ubu${UBUNTU_VERSION}_py${ANACONDA_PYTHON_VERSION}

DASK_ROCM_ARCH=gfx90a
CMAKE_VERSION=3.24.1
UCX_COMMIT=v1.14.x

podman build \
    --security-opt label=disable \
    --build-arg UBUNTU_VERSION=${UBUNTU_VERSION} \
    --build-arg ANACONDA_PYTHON_VERSION=${ANACONDA_PYTHON_VERSION} \
    --build-arg ROCM_VERSION=${ROCM_VERSION} \
    --build-arg BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT} \
    --build-arg DASK_ROCM_ARCH=${DASK_ROCM_ARCH} \
    --build-arg CMAKE_VERSION=${CMAKE_VERSION} \
    --build-arg UCX_COMMIT=${UCX_COMMIT} \
    --build-arg github_user=${GITHUB_USER} \
    --build-arg github_pass=${GITHUB_PASS} \
    -t dask:${BUILD_ENVIRONMENT} \
    -f Dockerfile .
