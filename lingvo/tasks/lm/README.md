# Lingvo Language Models.
This directory contains libraries and models for training language models.

# Guilde to build GShard Transformer-Based Giant Language Model.

The guide below is meant to illustrate how to train a 128B parameters GShard language model on GCP using
CloudTPUs v3-128 using 128-way model parallelism.

Reference to GShard: https://arxiv.org/abs/2006.16668

### Setup

First, you will need to setup a GCP account.

Some useful resources:

*   https://cloud.google.com/tpu/docs/how-to
*   https://cloud.google.com/compute/docs/instances/create-start-instance
*   https://cloud.google.com/compute/docs/instances/connecting-to-instance
*   https://cloud.google.com/tpu/docs/tutorials/amoebanet

You may also consider reading the
[TensorFlow Minigo Cloud TPU](https://github.com/tensorflow/minigo/tree/master/cluster)
script repository for managing GCP resources.


### Cloud TPU Setup

Please follow this [guide](https://cloud.google.com/tpu/docs/creating-deleting-tpus?hl=en#ctpu_4)
to create a Compute Engine VM and a TPU node with name ${TPU_NAME}.

#### Docker setup
It is required to use Docker to be able to push lingvo Docker images for GCP to load.

On the Compute Engine VM terminal:

    sudo apt-get update
    sudo apt-get install git
    sudo apt-get install python3-pip
    git clone  https://github.com/tensorflow/lingvo.git
    # Get docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    chmod +x get-docker.sh
    ./get-docker.sh
    gcloud auth configure-docker
    sudo usermod -aG docker $(whoami)
    sudo systemctl restart docker
    # Lingvo docker installation
    sudo docker build --tag tensorflow:lingvo_dev_lm - < lingvo/docker/dev.dockerfile

### Data Prep
The current setup assumed a synthetic input. We will provide more examples using real data soon.

## Launching the training job on a 8x8 Cloud TPU V3.
First launch lingvo on docker:

    sudo docker run --rm -it -v /home/$(whoami)/lingvo:/tmp/lingvo -e TPU_NAME=${TPU_NAME} --name lingvo tensorflow:lingvo_dev_lm bash

Inside the docker bash, launch the 128B parameter model with logdir ${LOGDIR}:

    bazel run -c opt //lingvo:trainer -- --mode=sync --alsologtostderr --model=lm.synthetic_packed_input.DenseLm128B8x8 --logdir=${LOGDIR} --tpu=${TPU_NAME} --worker_split_size=128 --ps_replicas=16 --cluster_placer_in_executor=true --job=executor_tpu

