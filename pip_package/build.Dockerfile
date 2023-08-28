# Constructs the environment within which we will build the lingvo pip wheels.
#
# From /tmp/lingvo,
# ❯ docker build --tag tensorflow:lingvo_wheelhouse --progress plain \
#     -f pip_package/build.Dockerfile .
# ❯ docker run --rm -it -v /tmp/lingvo:/tmp/lingvo -w /tmp/lingvo \
#      tensorflow:lingvo_wheelhouse bash
#   ❯ ./pip_package/invoke_build_per_interpreter.sh


ARG base_image="tensorflow/build:2.13-python3.9"
FROM $base_image
LABEL maintainer="Lingvo team <lingvo-bot@google.com>"

ENV DEBIAN_FRONTEND=noninteractive

# Install supplementary Python interpreters
RUN mkdir /tmp/python
# Temporarily remote nvidia as their apt repo seems broken.
#   See: https://forums.developer.nvidia.com/t/unable-to-add-apt-repo-mirror-sync-in-progress/263504
RUN mv /etc/apt/sources.list.d/cuda.list ~/cuda.list.bak
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
    lzma \
    liblzma-dev \
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
# TF version is fixed at 2.13.
RUN --mount=type=cache,target=/root/.cache \
  for p in 3.8 3.9 3.10; do \
    python${p} -m pip install -U pip && \
    python${p} -m pip install -U \
      attrs \
      apache-beam \
  	  backports.lzma \
      contextlib2 \
      dataclasses \
      google-api-python-client \
      auditwheel \
      dill \
      freezegun \
      graph-compression-google-research \
      h5py \
      ipykernel \
      jupyter \
      jupyter_http_over_ws \
      grpcio \
      matplotlib \
      mock \
      model-pruning-google-research \
      numpy \
      oauth2client \
      pandas \
      Pillow \
      pyyaml \
      recommonmark \
      scikit-learn \
      scipy \
      sentencepiece \
      sphinx \
      sphinx_rtd_theme \
      sympy \
      setuptools \
      sympy \
      twine \
      tensorflow~=2.13.0 tensorflow-text~=2.13.0 tensorflow-datasets tensorflow-probability; \
  done

COPY pip_package/devel.bashrc /root/devel.bashrc
RUN echo 'source /root/devel.bashrc' >> /root/.bashrc

WORKDIR "/tmp/lingvo"
