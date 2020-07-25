# Lingvo

[![PyPI](https://badge.fury.io/py/lingvo.svg)](https://badge.fury.io/py/lingvo)
[![Python](https://img.shields.io/pypi/pyversions/lingvo)](https://badge.fury.io/py/tensorflow)

[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://tensorflow.github.io/lingvo)

[![License](https://img.shields.io/github/license/tensorflow/lingvo)](LICENSE)

## What is it?

Lingvo is a framework for building neural networks in Tensorflow, particularly
sequence models.

A list of publications using Lingvo can be found [here](PUBLICATIONS.md).

## Table of Contents

*   [Releases](#releases)
    *   [Major breaking changes](#major-breaking-changes)
*   [Quick start](#quick-start)
    *   [Installation](#installation)
    *   [Running the MNIST image model](#running-the-mnist-image-model)
    *   [Running the machine translation model](#running-the-machine-translation-model)
    *   [Running the 3d object detection model](#running-the-3d-object-detection-model)
*   [Models](#models)
    *   [Automatic Speech Recognition](#automatic-speech-recognition)
    *   [Car](#car)
    *   [Image](#image)
    *   [Language Modelling](#language-modelling)
    *   [Machine Translation](#machine-translation)
*   [References](#references)
*   [License](#license)

## Releases

PyPI Version | Commit
------------ | ----------------------------------------
0.7.2        | b05642fe386ee79e0d88aa083565c9a93428519e

<details><summary>
<b>Older releases</b>
</summary><p>

PyPI Version | Commit
------------ | ------

Details for older releases are unavailable.

</p></details>

### Major breaking changes

#### HEAD

*   **General**
    *   NestedMap Flatten/Pack/Transform/Filter etc now expand descendent dicts
        as well.
    *   Subclasses of BaseLayer extending from `abc.ABCMeta` should now extend
        `base_layer.ABCLayerMeta` instead.
    *   Trying to call self.CreateChild outside of `__init__` now raises an
        error.
    *   `base_layer.initializer` has been removed. Subclasses no longer need to
        decorate their `__init__` function.
    *   Trying to call self.CreateVariable outside of `__init__` or
        `_CreateLayerVariables` now raises an error.
    *   It is no longer possible to access self.vars or self.theta inside of
        `__init__`. Refactor by moving the variable creation and access to
        `_CreateLayerVariables`. The variable scope is set automatically according to
        the layer name in `_CreateLayerVariables`.

<details><summary>
<b>Older releases</b>
</summary><p>

Details for older releases are available.

</p></details>

## Quick start

### Installation

There are two ways to set up Lingvo: installing a fixed version through pip, or
cloning the repository and building it with bazel. Docker configurations are
provided for each case.

If you would just like to use the framework as-is, it is easiest to just install
it through pip. This makes it possible to develop and train custom models using
a frozen version of the Lingvo framework. However, it is difficult to modify the
framework code or implement new custom ops.

If you would like to develop the framework further and potentially contribute
pull requests, you should avoid using pip and clone the repository instead.

**pip:**

The [Lingvo pip package](https://pypi.org/project/lingvo) can be installed with
`pip3 install lingvo`.

See the
[codelab](https://colab.research.google.com/github/tensorflow/lingvo/blob/master/codelabs/introduction.ipynb)
for how to get started with the pip package.

**From sources:**

The prerequisites are:

*   a TensorFlow 2.2 [installation](https://www.tensorflow.org/install/),
*   a `C++` compiler (only g++ 7.3 is officially supported), and
*   the bazel build system.

Refer to [docker/dev.dockerfile](docker/dev.dockerfile) for a set of working
requirements.

`git clone` the repository, then use bazel to build and run targets directly.
The `python -m module` commands in the codelab need to be mapped onto `bazel
run` commands.

**docker:**

Docker configurations are available for both situations. Instructions can be
found in the comments on the top of each file.

*   [lib.dockerfile](docker/lib.dockerfile) has the Lingvo pip package
    preinstalled.
*   [dev.dockerfile](docker/dev.dockerfile) can be used to build Lingvo from
    sources.

[How to install docker.](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

### Running the MNIST image model

#### Preparing the input data

**pip:**

```shell
mkdir -p /tmp/mnist
python3 -m lingvo.tools.keras2ckpt --dataset=mnist
```

**bazel:**

```shell
mkdir -p /tmp/mnist
bazel run -c opt //lingvo/tools:keras2ckpt -- --dataset=mnist
```

The following files will be created in `/tmp/mnist`:

*   `mnist.data-00000-of-00001`: 53MB.
*   `mnist.index`: 241 bytes.

#### Running the model

**pip:**

```shell
cd /tmp/mnist
curl -O https://raw.githubusercontent.com/tensorflow/lingvo/master/lingvo/tasks/image/params/mnist.py
python3 -m lingvo.trainer --run_locally=cpu --mode=sync --model=mnist.LeNet5 --logdir=/tmp/mnist/log
```

**bazel:**

```shell
(cpu) bazel build -c opt //lingvo:trainer
(gpu) bazel build -c opt --config=cuda //lingvo:trainer
bazel-bin/lingvo/trainer --run_locally=cpu --mode=sync --model=image.mnist.LeNet5 --logdir=/tmp/mnist/log --logtostderr
```

After about 20 seconds, the loss should drop below 0.3 and a checkpoint will be
saved, like below. Kill the trainer with Ctrl+C.

```
trainer.py:518] step:   205, steps/sec: 11.64 ... loss:0.25747201 ...
checkpointer.py:115] Save checkpoint
checkpointer.py:117] Save checkpoint done: /tmp/mnist/log/train/ckpt-00000205
```

Some artifacts will be produced in `/tmp/mnist/log/control`:

*   `params.txt`: hyper-parameters.
*   `model_analysis.txt`: model sizes for each layer.
*   `train.pbtxt`: the training `tf.GraphDef`.
*   `events.*`: a tensorboard events file.

As well as in `/tmp/mnist/log/train`:

*   `checkpoint`: a text file containing information about the checkpoint files.
*   `ckpt-*`: the checkpoint files.

Now, let's evaluate the model on the "Test" dataset. In the normal training
setup the trainer and evaler should be run at the same time as two separate
processes.

**pip:**

```shell
python3 -m lingvo.trainer --job=evaler_test --run_locally=cpu --mode=sync --model=mnist.LeNet5 --logdir=/tmp/mnist/log
```

**bazel:**

```shell
bazel-bin/lingvo/trainer --job=evaler_test --run_locally=cpu --mode=sync --model=image.mnist.LeNet5 --logdir=/tmp/mnist/log --logtostderr
```

Kill the job with Ctrl+C when it starts waiting for a new checkpoint.

```
base_runner.py:177] No new check point is found: /tmp/mnist/log/train/ckpt-00000205
```

The evaluation accuracy can be found slightly earlier in the logs.

```
base_runner.py:111] eval_test: step:   205, acc5: 0.99775392, accuracy: 0.94150388, ..., loss: 0.20770954, ...
```

### Running the machine translation model

To run a more elaborate model, you'll need a cluster with GPUs. Please refer to
[`third_party/py/lingvo/tasks/mt/README.md`](https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/mt/README.md)
for more information.

### Running the 3d object detection model

To run the StarNet model using CloudTPUs on GCP, please refer to
[`third_party/py/lingvo/tasks/car/README.md`](https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/car/README.md).

## Models

### Automatic Speech Recogition

*   [asr.librispeech.Librispeech960Grapheme](https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/asr/params/librispeech.py)<sup>1,2</sup>
*   [asr.librispeech.Librispeech960Wpm](https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/asr/params/librispeech.py)<sup>1,2</sup>

### Car

*   [car.kitti.StarNetCarModel0701](https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/car/params/kitti.py)<sup>3</sup>
*   [car.kitti.StarNetPedCycModel0704](https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/car/params/kitti.py)<sup>3</sup>
*   [car.waymo.StarNetVehicle](https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/car/params/waymo.py)<sup>3</sup>
*   [car.waymo.StarNetPed](https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/car/params/waymo.py)<sup>3</sup>

### Image

*   [image.mnist.LeNet5](https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/image/params/mnist.py)<sup>4</sup>

### Language Modelling

*   [lm.one_billion_wds.WordLevelOneBwdsSimpleSampledSoftmax](https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/lm/params/one_billion_wds.py)<sup>5</sup>

### Machine Translation

*   [mt.wmt14_en_de.WmtEnDeTransformerBase](https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/mt/params/wmt14_en_de.py)<sup>6</sup>
*   [mt.wmt14_en_de.WmtEnDeRNMT](https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/mt/params/wmt14_en_de.py)<sup>6</sup>
*   [mt.wmtm16_en_de.WmtCaptionEnDeTransformer](https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/mt/params/wmtm16_en_de.py)<sup>6</sup>

<font size="-1">

\[1]: [Listen, Attend and Spell](https://arxiv.org/pdf/1508.01211.pdf). William
Chan, Navdeep Jaitly, Quoc V. Le, and Oriol Vinyals. ICASSP 2016.

\[2]: [End-to-end Continuous Speech Recognition using Attention-based Recurrent
NN: First Results](https://arxiv.org/pdf/1412.1602.pdf). Jan Chorowski, Dzmitry
Bahdanau, Kyunghyun Cho, and Yoshua Bengio. arXiv 2014.

\[3]:
[StarNet: Targeted Computation for Object Detection in Point Clouds](https://arxiv.org/pdf/1908.11069.pdf).
Jiquan Ngiam, Benjamin Caine, Wei Han, Brandon Yang, Yuning Chai, Pei Sun, Yin
Zhou, Xi Yi, Ouais Alsharif, Patrick Nguyen, Zhifeng Chen, Jonathon Shlens, and
Vijay Vasudevan. arXiv 2019.

\[4]:
[Gradient-based learning applied to document recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf).
Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner. IEEE 1998.

\[5]:
[Exploring the Limits of Language Modeling](https://arxiv.org/pdf/1602.02410.pdf).
Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu.
arXiv, 2016.

\[6]: [The Best of Both Worlds: Combining Recent Advances in Neural Machine
Translation](http://aclweb.org/anthology/P18-1008). Mia X. Chen, Orhan Firat,
Ankur Bapna, Melvin Johnson, Wolfgang Macherey, George Foster, Llion Jones, Mike
Schuster, Noam Shazeer, Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz
Kaiser, Zhifeng Chen, Yonghui Wu, and Macduff Hughes. ACL 2018.

</font>

## References

*   [API Docs](https://tensorflow.github.io/lingvo/)
*   [Codelab](https://colab.research.google.com/github/tensorflow/lingvo/blob/master/codelabs/introduction.ipynb)

Please cite this [paper](https://arxiv.org/abs/1902.08295) when referencing
Lingvo.

```
@misc{shen2019lingvo,
    title={Lingvo: a Modular and Scalable Framework for Sequence-to-Sequence Modeling},
    author={Jonathan Shen and Patrick Nguyen and Yonghui Wu and Zhifeng Chen and others},
    year={2019},
    eprint={1902.08295},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## License

[Apache License 2.0](LICENSE)
