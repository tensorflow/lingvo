# Run the following commands in order:
#
# LINGVO_DIR="/tmp/lingvo"  # (change to the cloned lingvo directory, e.g. "$HOME/lingvo")
# LINGVO_DEVICE="gpu"  # (Leave empty to build and run CPU only docker)
# sudo docker build --tag tensorflow:lingvo $(test "$LINGVO_DEVICE" = "gpu" && echo "--build-arg base_image=nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04") - < lingvo/docker/dev.dockerfile
# sudo docker run --rm $(test "$LINGVO_DEVICE" = "gpu" && echo "--runtime=nvidia") -it -v ${LINGVO_DIR}:/tmp/lingvo -v ${HOME}/.gitconfig:/home/${USER}/.gitconfig:ro -p 6006:6006 -p 8888:8888 --name lingvo tensorflow:lingvo bash

# TODO(drpng): upgrade to latest (18.04)
ARG cpu_base_image="ubuntu:16.04"
ARG base_image=$cpu_base_image
FROM $base_image

LABEL maintainer="Patrick Nguyen <drpng@google.com>"

# Re-declare args because the args declared before FROM can't be used in any
# instruction after a FROM.
ARG cpu_base_image="ubuntu:16.04"
ARG base_image=$cpu_base_image

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        aria2 \
        build-essential \
        curl \
        gcc-4.8 g++-4.8 gcc-4.8-base \
        git \
        less \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        lsof \
        pkg-config \
        python \
        python-dev \
        python-tk \
        rsync \
        software-properties-common \
        sox \
        unzip \
        vim \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 100 &&\
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 100

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        jupyter_http_over_ws \
        matplotlib \
        numpy \
        pandas \
        recommonmark \
        scikit-learn==0.20.3 \
        scipy \
        sklearn \
        sphinx \
        sphinx_rtd_theme \
	sympy \
        && \
    python -m ipykernel.kernelspec

RUN jupyter serverextension enable --py jupyter_http_over_ws

# The latest tf-nightly-gpu requires CUDA 10 compatible nvidia drivers (410.xx).
# If you are unable to update your drivers, an alternative is to compile
# TensorFlow from source instead of installing from pip. 
RUN pip --no-cache-dir install tf-nightly$(test "$base_image" != "$cpu_base_image" && echo "-gpu")==1.15.0.dev20190730

ARG bazel_version=0.24.1
# This is to install bazel, for development purposes.
ENV BAZEL_VERSION ${bazel_version}
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh


# TensorBoard
EXPOSE 6006

# Jupyter
EXPOSE 8888

WORKDIR "/tmp/lingvo"

CMD ["/bin/bash"]
