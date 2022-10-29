# Constructs the environment within which we will build the lingvo pip wheels.
#
# From /tmp/lingvo,
# ❯ docker build --tag tensorflow:lingvo_wheelhouse --progress plain \
#     -f pip_package/build.Dockerfile .
# ❯ docker run --rm -it -v /tmp/lingvo:/tmp/lingvo -w /tmp/lingvo \
#      tensorflow:lingvo_wheelhouse bash
#   ❯ ./pip_package/invoke_build_per_interpreter.sh


ARG base_image="tensorflow/build:2.10-python3.9"
FROM $base_image
LABEL maintainer="Lingvo team <lingvo-bot@google.com>"

ENV DEBIAN_FRONTEND=noninteractive

# Install supplementary Python interpreters
RUN mkdir /tmp/python
RUN --mount=type=cache,target=/var/cache/apt \
  apt update && \
  apt install -yqq \
    apt-utils \
    bat \
    build-essential \
    checkinstall \
    libbz2-dev \
    libc6-dev \
    libffi-dev \
    libgdbm-dev \
    libncursesw5-dev \
    libreadline-gplv2-dev \
    libsqlite3-dev \
    libssl-dev \
    neovim \
    openssl \
    tk-dev \
    zlib1g-dev


# 3.9 is the built-in interpreter version in this image.
RUN for v in 3.8.15 3.10.0; do \
    wget "https://www.python.org/ftp/python/$v/Python-${v}.tar.xz" && \
    rm -rf "/tmp/python${v}" && mkdir -p "/tmp/python${v}" && \
    tar xvf "Python-${v}.tar.xz" -C "/tmp/python${v}" && \
    cd "/tmp/python${v}/Python-${v}" && \
    ./configure 2>&1 >/dev/null && \
    make -j8 altinstall 2>&1 >/dev/null && \
    ln -sf "/usr/local/bin/python${v%.*}" "/usr/bin/python${v%.*}"; \
  done

# For each python interpreter, install pip dependencies needed for lingvo
# TF version is fixed at 2.9.
RUN --mount=type=cache,target=/root/.cache \
  for p in 3.8 3.9 3.10; do \
    python${p} -m pip install -U pip && \
    python${p} -m pip install -U \
      attrs \
      auditwheel \
      graph-compression-google-research \
      grpcio \
      matplotlib \
      mock \
      model-pruning-google-research \
      numpy \
      scipy \
      sentencepiece \
      setuptools \
      sympy \
      twine \
      tensorflow~=2.9.2 tensorflow-text~=2.9.0 tensorflow-datasets; \
  done

COPY pip_package/devel.bashrc /root/devel.bashrc
RUN echo 'source /root/devel.bashrc' >> /root/.bashrc

WORKDIR "/tmp/lingvo"
