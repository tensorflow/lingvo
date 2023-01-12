# Run the following commands in order:
#
# LINGVO_DIR="/tmp/lingvo"  # (change to the cloned lingvo directory, e.g. "$HOME/lingvo")
# LINGVO_DEVICE="gpu"  # (Leave empty to build and run CPU only docker)
# docker build --tag tensorflow:lingvo $(test "$LINGVO_DEVICE" = "gpu" && echo "--build-arg base_image=nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04") - < "$LINGVO_DIR/docker/dev.Dockerfile"
# docker run --rm $(test "$LINGVO_DEVICE" = "gpu" && echo "--runtime=nvidia") -it -v ${LINGVO_DIR}:/tmp/lingvo -v ${HOME}/.gitconfig:/home/${USER}/.gitconfig:ro -p 6006:6006 -p 8888:8888 --name lingvo tensorflow:lingvo bash
#
# Other labels present at https://hub.docker.com/r/tensorflow/build/tags include
# tensorflow/build:{TF}-python{PY} with TF in {latest,2.8,2.9,2.10,2.11} and PY
# in {2.7–2.10}.
# 
# Test that everything worked:
#
# bazel test -c opt --test_output=streamed //lingvo:trainer_test //lingvo:models_test

ARG cpu_base_image="tensorflow/build:2.9-python3.9"
ARG base_image=$cpu_base_image
FROM $base_image

LABEL maintainer="Lingvo team <lingvo-bot@google.com>"

# Re-declare args because the args declared before FROM can't be used in any
# instruction after a FROM.
ARG cpu_base_image="tensorflow/build:2.9-python3.9"
ARG base_image=$cpu_base_image

COPY docker /docker

# The latest tensorflow requires CUDA 10 compatible nvidia drivers (410.xx).
# If you are unable to update your drivers, an alternative is to compile
# tensorflow from source instead of installing from pip.
# Note: removed waymo-open-dataset due to a new incompatibility.
RUN --mount=type=cache,target=/root/.cache \
  python3 -m pip install -U pip
RUN --mount=type=cache,target=/root/.cache \
  python3 -m pip install -U -r /docker/dev.requirements.txt
RUN python3 -m ipykernel.kernelspec
RUN jupyter serverextension enable --py jupyter_http_over_ws

COPY docker/devel.bashrc /root/devel.bashrc
RUN echo 'source /root/devel.bashrc' >> /root/.bashrc

# Expose TensorBoard and Jupyter ports
EXPOSE 6006
EXPOSE 8888

WORKDIR "/tmp/lingvo"

CMD ["/bin/bash"]
