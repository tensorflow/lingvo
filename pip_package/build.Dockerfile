FROM tensorflow/tensorflow:custom-op-ubuntu16

ENV GITHUB_BRANCH="master"
ENV PYTHON_VERSION="3"
ENV PYTHON_MINOR_VERSION="6"
ENV PIP_MANYLINUX2010="1"

RUN wget https://github.com/bazelbuild/bazel/releases/download/0.29.0/bazel-0.29.0-installer-linux-x86_64.sh > /dev/null
RUN bash bazel-0.29.0-installer-linux-x86_64.sh

RUN apt-get install -y python3.6

RUN pip3 install --upgrade setuptools auditwheel

# Add dependent packages to the image.
RUN pip3 install matplotlib \
    mock \
    model-pruning-google-research \
    numpy>=1.16.0 \
    sympy \
    twine
    
RUN pip3 install tensorflow-gpu

WORKDIR "/tmp/lingvo"
						
