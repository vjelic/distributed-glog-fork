#!/bin/bash

# Create Docker container for Dask cluster consistent of multi-node

CONTAINER_NAME=dask-multinode
IMAGE_NAME=localhost/dask
TAG_NAME=rocm5.6_ubu20.04_py3.9


IMAGE_NAME=
docker run -it --network=host \
  --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video \
  --name ${CONTAINER_NAME} ${IMAGE_NAME}:${TAG_NAME} /bin/bash
