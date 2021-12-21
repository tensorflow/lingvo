# CPU only:
# docker build --tag tensorflow:lingvo_lib - < docker/lib.dockerfile
# docker run --rm -it -p 6006:6006 -p 8888:8888 --name lingvo tensorflow:lingvo_lib bash
#
# With GPU support:
# docker build --tag tensorflow:lingvo_lib_gpu --build-arg base_image=nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04 - < docker/lib.dockerfile
# docker run --runtime=nvidia --rm -it -p 6006:6006 -p 8888:8888 --name lingvo tensorflow:lingvo_lib_gpu bash

ARG cpu_base_image="ubuntu:18.04"
ARG base_image=$cpu_base_image
FROM $base_image

LABEL maintainer="Lingvo Bot <lingvo-bot@google.com>"

# Re-declare args because the args declared before FROM can't be used in any
# instruction after a FROM.
ARG cpu_base_image="ubuntu:18.04"
ARG base_image=$cpu_base_image

# Pick up some TF dependencies
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends software-properties-common
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        gpg-agent \
        less \
        lsof \
        pkg-config \
        python3-distutils \
        rsync \
        sox \
        unzip \
        vim \
        dirmngr \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install python 3.9
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776
RUN echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu bionic main" > /etc/apt/sources.list.d/deadsnakes-ppa-bionic.list
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y python3.9 python3.9-distutils
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1000

RUN curl -O https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm get-pip.py

RUN python3 -m pip --no-cache-dir install lingvo

RUN python3 -m ipykernel.kernelspec
RUN jupyter serverextension enable --py jupyter_http_over_ws

# TensorBoard
EXPOSE 6006

# Jupyter
EXPOSE 8888

CMD ["/bin/bash"]
