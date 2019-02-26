# Lingvo

## What is it?

Lingvo is a framework for building neural networks in Tensorflow, particularly
sequence models.

A list of publications using Lingvo can be found [here](PUBLICATIONS.md).

## Quick start

### Docker

The easiest way to get started is to use the provided
[Docker script](docker/dev.dockerfile). If instead you want to install it
directly on your machine, skip to the section below.

First,
[install docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/). Then,
the following commands should give you a working shell with Lingvo installed.

```shell
LINGVO_DIR="/tmp/lingvo"  # (change to the cloned lingvo directory, e.g. "$HOME/lingvo")
LINGVO_DEVICE="gpu"  # (Leave empty to build and run CPU only docker)
sudo docker build --tag tensorflow:lingvo $(test "$LINGVO_DEVICE" = "gpu" && echo "--build-arg base_image=nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04") - < ${LINGVO_DIR}/docker/dev.dockerfile
sudo docker run --rm $(test "$LINGVO_DEVICE" = "gpu" && echo "--runtime=nvidia") -it -v ${LINGVO_DIR}:/tmp/lingvo -v ${HOME}/.gitconfig:/home/${USER}/.gitconfig:ro -p 6006:6006 -p 8888:8888 --name lingvo tensorflow:lingvo bash
bazel test -c opt //lingvo:trainer_test //lingvo:models_test
```

### Installing directly

This is an alternative to using Docker as described in the section above.

The prerequisites are:

*   a TensorFlow [installation](https://www.tensorflow.org/install/) (for now
    tf-nightly is required),
*   a `C++` compiler (only g++ 4.8 is officially supported), and
*   the bazel build system.

Refer to [docker/dev.dockerfile](docker/dev.dockerfile) for more specific
details.

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

## Current models

### Automatic Speech Recogition

*   [asr.librispeech.Librispeech960Grapheme](https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/asr/params/librispeech.py)<sup>1,2</sup>
*   [asr.librispeech.Librispeech960Wpm](https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/asr/params/librispeech.py)<sup>1,2</sup>

### Image

*   [image.mnist.LeNet5](https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/image/params/mnist.py)<sup>3</sup>

### Language Modelling

*   [lm.one_billion_wds.WordLevelOneBwdsSimpleSampledSoftmax](https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/lm/params/one_billion_wds.py)<sup>4</sup>

### Machine Translation

*   [mt.wmt14_en_de.WmtEnDeTransformerBase](https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/mt/params/wmt14_en_de.py)<sup>5</sup>
*   [mt.wmt14_en_de.WmtEnDeRNMT](https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/mt/params/wmt14_en_de.py)<sup>5</sup>
*   [mt.wmtm16_en_de.WmtCaptionEnDeTransformer](https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/mt/params/wmtm16_en_de.py)<sup>5</sup>

<font size="-1">

\[1]: [Listen, Attend and Spell](https://arxiv.org/pdf/1508.01211.pdf). William
Chan, Navdeep Jaitly, Quoc V. Le, and Oriol Vinyals. ICASSP 2016.

\[2]: [End-to-end Continuous Speech Recognition using Attention-based Recurrent
NN: First Results](https://arxiv.org/pdf/1412.1602.pdf). Jan Chorowski, Dzmitry
Bahdanau, Kyunghyun Cho, and Yoshua Bengio. arXiv 2014.

\[3]:
[Gradient-based learning applied to document recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf).
Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner. IEEE 1998.

\[4]:
[Exploring the Limits of Language Modeling](https://arxiv.org/pdf/1602.02410.pdf).
Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu.
arXiv, 2016.

\[5]: [The Best of Both Worlds: Combining Recent Advances in Neural Machine
Translation](http://aclweb.org/anthology/P18-1008). Mia X. Chen, Orhan Firat,
Ankur Bapna, Melvin Johnson, Wolfgang Macherey, George Foster, Llion Jones, Mike
Schuster, Noam Shazeer, Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz
Kaiser, Zhifeng Chen, Yonghui Wu, and Macduff Hughes. ACL 2018.

</font>

## References

*   [API Docs](https://tensorflow.github.io/lingvo/)

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
