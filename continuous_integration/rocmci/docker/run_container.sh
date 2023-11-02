#!/bin/bash

# Create docker container from base Dask docker image plus dependencies
# Uses podman, which can be replaced with docker, if docker is available.

DOCKER_TAG=rocm5.6_ubu20.04_py3.9

podman run -it --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video \
  --name dask-release -p 11088:8788 -p 11087:8787 \
  -v $HOME:/myhome localhost/dask:${DOCKER_TAG} /bin/bash
