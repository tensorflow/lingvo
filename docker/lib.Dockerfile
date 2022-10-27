# CPU only:
# cd /tmp/lingvo
# docker build --tag tensorflow:lingvo_lib -f docker/lib.dockerfile .
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

ENV DEBIAN_FRONTEND=noninteractive

# Pick up some TF dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    apt update && \
    apt install -yqq --no-install-recommends \
        software-properties-common && \
    apt install -yqq --no-install-recommends \
        build-essential \
        curl \
        dirmngr \
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
        tensorflow-text==2.9

# Install python 3.9
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776
RUN echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu bionic main" > /etc/apt/sources.list.d/deadsnakes-ppa-bionic.list
RUN apt update && apt install -y python3.9 python3.9-distutils
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1000

RUN curl -O https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm get-pip.py

RUN python3 -m pip --no-cache-dir install lingvo

RUN python3 -m ipykernel.kernelspec
RUN jupyter serverextension enable --py jupyter_http_over_ws

# Expose TensorBoard and Jupyter ports
EXPOSE 6006
EXPOSE 8888

CMD ["/bin/bash"]
