FROM tensorflow/tensorflow:custom-op-ubuntu16

ENV GITHUB_BRANCH="master"
ENV PYTHON_VERSION="3"
ENV PYTHON_MINOR_VERSION=""
ENV PIP_MANYLINUX2010="1"

RUN apt-get update && apt-get install -y && \
    echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
    curl https://bazel.build/bazel-release.pub.gpg | apt-key add - && \
    apt-get update && apt-get install -y bazel && \
    rm -rf /usr/local/bin/bazel && hash -r

RUN apt-get install python3.6

RUN pip3 install --upgrade setuptools auditwheel

# Add dependent packages to the image.
RUN pip3 install matplotlib \
    mock \
    numpy>=1.16.0 \
    sympy
    
RUN pip3 install tf-nightly==1.15.0.dev20190814

WORKDIR "/tmp/lingvo"
						