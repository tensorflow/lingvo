# For CPU only:
# docker build --tag tensorflow:lingvo_lib - < lingvo/docker/lib.dockerfile

# For GPU support:
# docker build --tag tensorflow:lingvo_lib --build-arg base_image=nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04" - < lingvo/docker/lib.dockerfile

ARG cpu_base_image="ubuntu:18.04"
ARG base_image=$cpu_base_image
FROM $base_image

LABEL maintainer="Lingvo Bot <lingvo-bot@google.com>"

# Re-declare args because the args declared before FROM can't be used in any
# instruction after a FROM.
ARG cpu_base_image="ubuntu:18.04"
ARG base_image=$cpu_base_image

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        less \
        lsof \
        pkg-config \
        python3-distutils \
        rsync \
        sox \
        unzip \
        vim \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm get-pip.py

RUN pip3 --no-cache-dir install lingvo$(test "$base_image" != "$cpu_base_image" && echo "_gpu")

RUN python3 -m ipykernel.kernelspec
RUN jupyter serverextension enable --py jupyter_http_over_ws

# TensorBoard
EXPOSE 6006

# Jupyter
EXPOSE 8888

CMD ["/bin/bash"]
