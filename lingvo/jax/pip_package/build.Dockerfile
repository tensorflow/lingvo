FROM tensorflow/tensorflow:custom-op-ubuntu16

ENV GITHUB_BRANCH="master"
ENV PYTHON_VERSION="3"
ENV PYTHON_MINOR_VERSION="9"
ENV PIP_MANYLINUX2010="1"

# There are some problems with the python3 installation from custom-op-ubuntu16.
# Remove it and install new ones.
RUN apt-get remove --purge -y python3.5 python3.6
# Delete buggy preinstalled python3.7 interpreter (`import bz2` fails).
RUN rm -R -f /usr/local/lib/python3.7* /usr/local/bin/python3.7*
RUN rm -f /etc/apt/sources.list.d/jonathonf-ubuntu-python-3_6-xenial.list
RUN apt-key del F06FC659

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776
RUN echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu xenial main" > /etc/apt/sources.list.d/deadsnakes-ppa-xenial.list
RUN apt-get update && apt-get install -y python3.7 python3.7-distutils python3.8 python3.8-distutils python3.9 python3.9-distutils
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

# Download and install bazel.
RUN wget https://github.com/bazelbuild/bazel/releases/download/4.0.0/bazel-4.0.0-installer-linux-x86_64.sh > /dev/null
RUN bash bazel-4.0.0-installer-linux-x86_64.sh

RUN for python in python3.7 python3.8 python3.9; do \
      $python get-pip.py && \
      $python -m pip install --upgrade pip setuptools auditwheel && \
      $python -m pip install --upgrade \
        absl-py \
        attrs \
        clu \
        flax \
        graph-compression-google-research \
        grpcio \
        jax \
        matplotlib \
        mock \
        model-pruning-google-research \
        numpy \
        optax \
        optax-shampoo \
        scipy \
        sentencepiece \
        sympy \
        twine && \
      $python -m pip install tensorflow tensorflow-datasets tensorflow-text; \
    done

WORKDIR "/tmp/lingvo"
