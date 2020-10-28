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

## GShard under the hood.
Everything in GShard starts with a registered model class. We bundle the model hyperparameters in a python class, for example, synthetic_packed_input.DenseLm128B8x8. The Task() function defines hyperparameters for model architecture as well as training parameters like learning rates et al. The Train() and Test() functions specify the input configs.

GShard provides flexibility to divide a tensor or a variable along multiple axises. In DenseLm128B8x8, there are two sharding hyperparameters defining how tensors and variables are sharded across devices. First, NUM_DEVICES_PER_SPLIT defines the number of TPUV3 cores used for this model. That also defines the maximum number of partitions. DEVICE_MESH_SHAPE is a tuple of two integeres (s0, s1), each of which defines the number of partitions along some dimension of a tensor or variable. The product of s0 and s1 should be equal to NUM_DEVICES_PER_SPLIT. In transformer, tensors/variables are partitioned across NUM_DEVICES_PER_SPLIT devices:

* The projection weight in feedforward layer with shape (M, H). M is the model dimension and H is the hidden dimension. The projection weigth will be divided into s0 partitions along the M axis and s1 partitions along the H axis.
* The projection weight in the attention layer with shape (M, D, N). N is the number heads and D is key-value projection dimension per head. The weight matrix will be divided into s0 partitions along the M axis and s1 partitions along the N axis.
* Activation Tensors with shape (B, S, M) where B is the batch size and S is the sequence length. Those activation tensors will be devided into s0 partitions along the B axis and s1 partitions along the M axis.
