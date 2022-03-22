FROM tensorflow/tensorflow:nightly-custom-op-ubuntu16

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

# Deadsnakes PPA no longer supports 16.04
# https://github.com/deadsnakes/issues/issues/195
# We build the supported python versions here
RUN mkdir /tmp/python
RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libssl-dev zlib1g-dev openssl libffi-dev

RUN for v in 3.7.13 3.8.13 3.9.11; do \
    wget "https://www.python.org/ftp/python/$v/Python-${v}.tar.xz" && \
    tar xvf "Python-${v}.tar.xz" -C /tmp/python && \
    cd "/tmp/python/Python-${v}" && \
    ./configure && \
    make -j8 altinstall && \
    ln -s "/usr/local/bin/python${v%.*}" "/usr/bin/python${v%.*}"; \
  done

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
        einops \
        flax \
        graph-compression-google-research \
        grpcio \
        jax \
        jax-bitempered-loss \
        matplotlib \
        mock \
        model-pruning-google-research \
        numpy \
        optax \
        optax-shampoo \
        scipy \
        sentencepiece \
        sympy \
        tensorstore \
        twine && \
      $python -m pip install tensorflow tensorflow-datasets tensorflow-text; \
    done

WORKDIR "/tmp/lingvo"
