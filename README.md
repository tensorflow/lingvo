# Lingvo

## What is it?

Lingvo is a toolkit which is suited to build neural networks, particulary
sequence models.

It is designed to be a flexible research tool.

This is an implementation of the models in the following papers:

* [The Best of Both Worlds: Combining Recent Advances in Neural Machine
Translation](http://aclweb.org/anthology/P18-1008). Mia Chen, Orhan Firat,
Ankur Bapna, Melvin Johnson, Wolfgang Macherey, George Foster, Llion Jones,
Niki Parmar, Mike Schuster, Zhifeng Chen, Yonghui Wu and Macduff Hughes.
ACL 2018.

* Training Deeper Neural Machine Translation Models with Transparent Attention.
Ankur Bapna, Mia Chen, Orhan Firat, Yuan Cao and Yonghui Wu. EMNLP 2018.
TODO(lingvo-team): update with arxiv link.

## Quick start

### Docker
The docker files in the `docker` directory. They provide a blueprint of how
to install and run the software in multiple configuration.

### Installation

To install the toolkit, you need:
* a TensorFlow [installation](https://www.tensorflow.org/install/),
* a `C++` compiler,
* the bazel build system, and
* the protobuf package.

To build, use:

```shell
bazel build -c opt //lingvo:trainer
```

### Running the MNIST image model

#### Preparing the input data

```shell
mkdir -p /tmp/mnist
bazel run -c opt //lingvo/tools:keras2ckpt -- --dataset=mnist --out=/tmp/mnist/mnist
```

You will get the following files in `/tmp/mnist`:

* `mnist.data-00000-of-00001`: 53MB.
* `mnist.index`: 241 bytes.

#### Running the model

To run the trainer in single-machine mode, use

```shell
bazel build -c opt //lingvo:trainer
bazel-bin/lingvo/trainer --run_locally=cpu --mode=sync --model=image.mnist.LeNet5 --logdir=/tmp/mnist/log --logtostderr
```

After a few seconds, the training accuracy should reach `85%` at step 100,
as seen in the following line.

```
INFO:tensorflow:step:   100 accuracy:0.85546875 log_pplx:0.46025506 loss:0.46025506 num_preds:256 num_samples_in_batch:256
```

The artifacts will be produced in `/tmp/mnist/log/control`:

* `params.txt`: hyper-parameters.
* `model_analysis.txt`: model sizes for each layer.
* `train.pbtxt`: the training `tf.GraphDef`.
* `events.*`: a tensorboard events file.

In the `/tmp/mnist/log/train` directory, one will obtain:

* `ckpt-*`: the checkpoint files.
* `checkpoint`: a text file containing information about the checkpoint files.


### Running the machine translation model

To run a more elaborate model, you'll need a cluster with GPUs. Please
refer to `lingvo/tasks/mt/README.md` for more information.
