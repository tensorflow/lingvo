# Lingvo

## What is it?

Lingvo is a framework for building neural networks in Tensorflow, particularly
sequence models.

A list of publications using Lingvo can be found [here](PUBLICATIONS.md).

## Quick start

### Docker

The docker files in the `docker` directory provide a blueprint of how to install
and run the software in multiple configurations.

### Installation

The prerequisites are:

*   a TensorFlow [installation](https://www.tensorflow.org/install/) (for now
    tf-nightly is required),
*   a `C++` compiler (only g++ 4.8 is officially supported),
*   the bazel build system, and
*   the protobuf package.

### Running the MNIST image model

#### Preparing the input data

```shell
mkdir -p /tmp/mnist
bazel run -c opt //lingvo/tools:keras2ckpt -- --dataset=mnist --out=/tmp/mnist/mnist
```

You will get the following files in `/tmp/mnist`:

*   `mnist.data-00000-of-00001`: 53MB.
*   `mnist.index`: 241 bytes.

#### Running the model

To run the trainer in single-machine mode, use

```shell
bazel build -c opt //lingvo:trainer
bazel-bin/lingvo/trainer --run_locally=cpu --mode=sync --model=image.mnist.LeNet5 --logdir=/tmp/mnist/log --logtostderr
```

After a few seconds, the training accuracy should reach `85%` at step 100, as
seen in the following line.

```
INFO:tensorflow:step:   100 accuracy:0.85546875 log_pplx:0.46025506 loss:0.46025506 num_preds:256 num_samples_in_batch:256
```

The artifacts will be produced in `/tmp/mnist/log/control`:

*   `params.txt`: hyper-parameters.
*   `model_analysis.txt`: model sizes for each layer.
*   `train.pbtxt`: the training `tf.GraphDef`.
*   `events.*`: a tensorboard events file.

In the `/tmp/mnist/log/train` directory, one will obtain:

*   `ckpt-*`: the checkpoint files.
*   `checkpoint`: a text file containing information about the checkpoint files.

### Running the machine translation model

To run a more elaborate model, you'll need a cluster with GPUs. Please refer to
`lingvo/tasks/mt/README.md` for more information.
