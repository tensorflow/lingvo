# Lingvo

## What is it?

Lingvo is a toolkit which is suited to build neural networks, particulary
sequence models.

It is designed to be a flexible research tool.

This is an implementation of the models in the following
[paper](https://arxiv.org/abs/1804.09849):
"The Best of Both Worlds: Combining Recent Advances in Neural Machine
Translation", Mia Xu Chen, Orhan Firat, Ankur Bapna, Melvin Johnson,
Wolfgang Macherey, George Foster, Llion Jones, Niki Parmar, Mike Schuster,
Zhifeng Chen, Yonghui Wu, Macduff Hughes.

## Quick start

### Docker
The docker files in the `docker` directory. They provide a blueprint of how
to install and run the software in multiple configuration.

### Installation

To install the toolkit, you need:
* a TensorFlow [installation](https://www.tensorflow.org/install/),
* a `C++` compiler (for now, gcc-4.8.x is required),
* the bazel build system, and
* the protobuf package.

To build, use:

```shell
bazel build -c opt //lingvo:trainer
```

### Running models

TODO(lingvo-team): describe Single machine, CloudTPU, multi-GPUs.

Debug environment.

### Code overview

TODO(lingvo-team): fill out.
