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
