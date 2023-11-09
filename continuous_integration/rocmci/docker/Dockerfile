ARG UBUNTU_VERSION

FROM ubuntu:${UBUNTU_VERSION}

ARG UBUNTU_VERSION

ENV DEBIAN_FRONTEND noninteractive

# Set AMD gpu targets to build for
ARG DASK_ROCM_ARCH
ENV DASK_ROCM_ARCH ${DASK_ROCM_ARCH}

# Install common dependencies (so that this step can be cached separately)
COPY ./common/install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh

# Install user
COPY ./common/install_user.sh install_user.sh
RUN bash ./install_user.sh && rm install_user.sh

# Install conda and other packages (e.g., numpy, pytest)
ARG ANACONDA_PYTHON_VERSION
ARG CONDA_CMAKE
ENV ANACONDA_PYTHON_VERSION=$ANACONDA_PYTHON_VERSION
ENV PATH /opt/conda/envs/py_$ANACONDA_PYTHON_VERSION/bin:/opt/conda/bin:$PATH
COPY requirements-ci.txt /opt/conda/requirements-ci.txt
COPY ./common/install_conda.sh install_conda.sh
COPY ./common/common_utils.sh common_utils.sh
RUN bash ./install_conda.sh && rm install_conda.sh common_utils.sh /opt/conda/requirements-ci.txt

# Install rocm
ARG ROCM_VERSION
COPY ./common/install_rocm.sh install_rocm.sh
RUN bash ./install_rocm.sh
RUN rm install_rocm.sh
ENV ROCM_PATH /opt/rocm
ENV PATH /opt/rocm/bin:$PATH
ENV PATH /opt/rocm/hcc/bin:$PATH
ENV PATH /opt/rocm/hip/bin:$PATH
ENV PATH /opt/rocm/opencl/bin:$PATH
ENV PATH /opt/rocm/llvm/bin:$PATH
ENV MAGMA_HOME /opt/rocm/magma
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# (optional) Install non-default CMake version
ARG CMAKE_VERSION
COPY ./common/install_cmake.sh install_cmake.sh
RUN if [ -n "${CMAKE_VERSION}" ]; then bash ./install_cmake.sh; fi
RUN rm install_cmake.sh

# (optional) Install UCX
ARG UCX_COMMIT
ENV UCX_COMMIT $UCX_COMMIT
ADD ./common/install_ucx.sh install_ucx.sh
RUN if [ -n "${UCX_COMMIT}" ]; then bash ./install_ucx.sh; fi
RUN rm install_ucx.sh

# Include BUILD_ENVIRONMENT environment variable in image
ARG BUILD_ENVIRONMENT
ENV BUILD_ENVIRONMENT ${BUILD_ENVIRONMENT}


#### ======= Adding Dask and Dependencies ========

# Temporary use of GITHUB access keys for RMM repo which is private until release
ARG github_user
ARG github_pass

# create a workspace directory inside docker to perform all the tasks
USER root
ENV WORKSPACE_DIR=/workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

# copy package sources to build
# current sub-directory:
#    cupy  RMM  distributed  ucx-py
COPY ./dask_src/. $WORKSPACE_DIR

# explicitly add video group in the docker environment
RUN echo 'ADD_EXTRA_GROUPS=1' | tee -a /etc/adduser.conf && \
echo 'EXTRA_GROUPS=video' | tee -a /etc/adduser.conf

RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

# install dependent packages: pyamdsmi and others for building python packages
COPY requirements.txt $WORKSPACE_DIR
RUN pip install --index-url https://test.pypi.org/simple pyamdsmi \
    && pip install -r requirements.txt

# install hip-python
# RUN wget https://test-files.pythonhosted.org/packages/c3/04/a130b957b1f5750194e8d19e411e42a89594d6293375779f527b44928ddd/hip_python-5.6.0.228.19-cp38-cp38-manylinux_2_17_x86_64.whl#sha256=1e748b4041f781ad85e019134be89f198aee51a58c03aa6320db57854899cbec \
#     && pip install hip_python*.whl
RUN pip install --index-url https://test.pypi.org/simple hip-python

# build and install cupy-rocm, can choose architecture, e.g. gfx90a (MI200)
WORKDIR $WORKSPACE_DIR/cupy
RUN CUPY_INSTALL_USE_HIP=1 ROCM_HOME=/opt/rocm HCC_AMDGPU_TARGET=$DASK_ROCM_ARCH pip install -e . --no-cache-dir

# build and install RMM
WORKDIR $WORKSPACE_DIR/RMM
RUN pip install scikit-build \
    && export GITHUB_USER=${github_user} && export GITHUB_PASS=${github_pass} \
    && CXX=hipcc bash build.sh

# install dask-hip
RUN pip install --index-url https://test.pypi.org/simple dask-hip==0.2.0 \
    && pip uninstall -y distributed

# build and install distributed, should be after dask-hip
WORKDIR $WORKSPACE_DIR/distributed
RUN python -m build \
    && pip install dist/*.whl

# build and install ucx-py
WORKDIR $WORKSPACE_DIR/ucx-py
RUN UCX_HOME=/opt/ucx pip install -e . --no-cache-dir

# prepare test
COPY ./run_dask_cluster_simple.py $WORKSPACE_DIR
COPY ./*.ipynb $WORKSPACE_DIR
# COPY ./run_dask_cluster_multinode.py $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

#after executing all of this the new docker image will be ready

RUN echo "Dask container build completed"

# CMD ["bash"]
