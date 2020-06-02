# Lingvo 3D pointcloud object detection

This directory contains libraries and models for training 3D classification and
detection datasets using primarily pointcloud inputs.

The repo contains the code needed to reproduce the results at
https://arxiv.org/abs/1908.11069.

## Guide to reproduce StarNet on GCP

The guide below is meant to illustrate how to train a StarNet model on GCP using
CloudTPUs for training and GPUs for evaluation, using GKE.

### Setup

First, you will need to setup a GCP account.

Some useful resources:

*   https://cloud.google.com/kubernetes-engine/docs/tutorials/hello-app
*   https://cloud.google.com/tpu/docs/kubernetes-engine-setup
*   https://cloud.google.com/kubernetes-engine/docs/troubleshooting

You may also consider reading the
[TensorFlow Minigo Cloud TPU](https://github.com/tensorflow/minigo/tree/master/cluster)
script repository for managing GCP resources.

#### Cloud TPU cluster setup

__IMPORTANT: These commands will result in VMs and other GCP resources being
created and will result in charges to your GCP account! Proceed with care!__

To properly create the CloudTPU cluster, one should follow the TPU documentation
above. An example might be:

    gcloud container clusters create tpu_v3_cluster --cluster-version=1.13 --scopes=cloud-platform,gke-default --enable-ip-alias --enable-tpu --zone=europe-west4-a

where we create a TPU-compatible cluster in a zone that contains V3 TPU pods. We
only need to specify the size of the TPU pod we want when we create the job.

#### GPU cluster setup

__IMPORTANT: These commands will result in VMs and other GCP resources being
created and will result in charges to your GCP account! Proceed with care!__

First, create the cluster. You may need to get
[GPU quota](https://cloud.google.com/compute/quotas#gpus) in the zone you want
for the GPU type you want.

For example, here we create a P100 cluster of size 1 using high-memory CPU
instances (which are useful for evaluating / decoding):

    gcloud container clusters create p100-europe-west4-a-nh16 --accelerator type=nvidia-tesla-p100,count=1 --num-nodes=1 --zone europe-west4-a --scopes=cloud-platform,gke-default --enable-ip-alias --cluster-version=1.13 --machine-type=n1-highmem-16

To be able to launch GPU jobs on this cluster, one must then install the
drivers.

    kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

To check that the GPUs have been fully initialized with these drivers, you can
run:

    kubectl get no -w -o yaml | grep -E 'hostname:|nvidia.com/gpu'

#### Docker setup

It is required to use Docker to be able to push Docker images for GKE to load.

One should set up docker following instructions online.

We have provided a lingvo-compatible dockerfile that comes with the lingvo pip
package pre-installed. One should build the docker with the GPU nvidia-docker
base for running on GPUs. For example, from the current repo:

    docker build --tag tensorflow:lingvo_lib_gpu --build-arg base_image=nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04 - < lingvo/docker/lib.dockerfile

This will be the base docker image you will use to run lingvo jobs on GKE.

#### Data prep

To train our existing models without modification, one must first upload
TF.Example versions of the datasets to GCS.

We have provided scripts to process the raw data in the tools/ and waymo/tools
subdirectories. For example, one can run tools/kitti_exporter.py to create the
TFRecord TFExample files from the raw KITTI data; one can then upload the
results to GCS under a bucket you own.

## Launching the training job on a 4x4 Cloud TPU V3.

We have provided a python script that launches / manages GKE jobs at
lingvo/tools/gke_launch.py. Let's say you want to reproduce the
model used for training the StarNet Pedestrian/Cyclist model on KITTI named
StarNetPedCycModel0704 in params/kitti.py.

To launch training on TPU, decoding for the validation set on GPU, and a
TensorBoard to monitor it all, one can run:

    # Name of the registered model in params/kitti.py to run.
    export MODEL=params.kitti.StarNetPedCycModel0704

    # Environment variable containing path to input dataset.
    export DATA_ENV="KITTI_DIR=gs://my-bucket/kitti"

    # For waymo, one can use the version generated via generate_waymo_tf.py
    # graciously hosted by Waymo; you must have registered for access to
    # the Waymo dataset to be able to access the following bucket.
    # export DATA_ENV="WAYMO_DIR=gs://waymo_open_dataset_v_1_0_0_tf_example_lingvo/v.1.2.0/"

    export SPLIT=dev  # Use 'minidev' for smaller Waymo validation set.

    # Base docker image name to use for packaging the
    # code to run.
    export DOCKER_IMAGE=gcr.io/my-project/lingvo-code

    # Location of where the code to be packaged into
    # the docker file is located.
    export CODE_DIR=/path/to/base/of/lingvo_repo

    # Log directory to write checkpoints to.
    export LOGDIR=gs://my-bucket/logs/starnet.kitti.v0

    # Name (or filter) of the TPU cluster you created above.
    export TPU_CLUSTER_NAME=tpu_v3_cluster

    # Specify using a 4x4 Cloud TPU v3.
    export TPU_TYPE=v3-32

    # Name (or filter)  of the GPU cluster you created above.
    export GPU_CLUSTER_NAME=p100

    # Specify using a p100 GPU.
    export GPU_TYPE=p100

    # The prefix name of the jobs to launch on GKE.
    export EXP_NAME=kitti.starnet.pedcyc.v0

    python3 lingvo/tools/gke_launch.py \
    --model=$MODEL \
    --base_image=tensorflow:lingvo_lib_gpu \
    --image=$DOCKER_IMAGE \
    --logdir=$LOGDIR \
    --tpu_type=$TPU_TYPE \
    --trainer_cell=$TPU_CLUSTER_NAME \
    --decoder_cell=$GPU_CLUSTER_NAME \
    --decoder_gpus=1 \
    --gpu_type=$GPU_TYPE \
    --decoder=$SPLIT \
    --extra_envs=$DATA_ENV \
    --name=$EXP_NAME \
    --build=$CODE_DIR/lingvo/tasks/car \
    reload all

As the model trains, the TPU and GPU jobs will output events to the log
directory, which the TensorBoard job will visualize.

To bring down the jobs, you can run the above command with "down" instead of
"reload".
