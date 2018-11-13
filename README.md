# Lingvo

## What is it?

Lingvo is a toolkit which is suited to build neural networks, particulary
sequence models.

It was used for the following research papers.

*   Translation:

    *   [The Best of Both Worlds: Combining Recent Advances in Neural Machine
        Translation](http://aclweb.org/anthology/P18-1008). Mia X. Chen, Orhan
        Firat, Ankur Bapna, Melvin Johnson, Wolfgang Macherey, George Foster,
        Llion Jones, Mike Schuster, Noam Shazeer, Niki Parmar, Ashish Vaswani,
        Jakob Uszkoreit, Lukasz Kaiser, Zhifeng Chen, Yonghui Wu, and Macduff
        Hughes. ACL 2018.

    *   [Revisiting Character-Based Neural Machine Translation with Capacity and
        Compression](https://arxiv.org/abs/1808.09943). Colin Cherry, George
        Foster, Ankur Bapna, Orhan Firat, and Wolfgang Macherey. EMNLP 2018.

    *   [Training Deeper Neural Machine Translation Models with Transparent
        Attention](https://arxiv.org/abs/1808.07561). Ankur Bapna, Mia X. Chen,
        Orhan Firat, Yuan Cao and Yonghui Wu. EMNLP 2018.

    *   [Sequence-to-Sequence Models Can Directly Translate Foreign Speech](https://arxiv.org/abs/1703.08581).
        Ron J. Weiss, Jan Chorowski, Navdeep Jaitly, Yonghui Wu, and Zhifeng
        Chen. Interspeech 2017.

    *   [Google's Neural Machine Translation System: Bridging the Gap between
        Human and Machine Translation](https://arxiv.org/abs/1609.08144).
        Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi,
        Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff
        Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Łukasz Kaiser,
        Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens,
        George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason
        Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, and
        Jeffrey Dean. Technical Report, 2016.

*   Speech Recognition:

    *   [A comparison of techniques for language model integration in
        encoder-decoder speech
        recognition](https://arxiv.org/pdf/1807.10857.pdf). Shubham Toshniwal,
        Anjuli Kannan, Chung-Cheng Chiu, Yonghui Wu, Tara N. Sainath, Karen
        Livescu. IEEE SLT 2018.

    *   [Deep Context: End-to-End Contextual Speech Recognition](https://arxiv.org/pdf/1808.02480.pdf).
        Golan Pundak, Tara N. Sainath, Rohit Prabhavalkar, Anjuli Kannan, Ding
        Zhao. IEEE SLT 2018.

    *   [Speech recognition for medical conversations](https://arxiv.org/abs/1711.07274).
        Chung-Cheng Chiu, Anshuman Tripathi, Katherine Chou, Chris Co, Navdeep
        Jaitly, Diana Jaunzeikare, Anjuli Kannan, Patrick Nguyen, Hasim Sak,
        Ananth Sankar, Justin Tansuwan, Nathan Wan, Yonghui Wu, and Xuedong
        Zhang. Interspeech 2018.

    *   [Compression of End-to-End Models](https://www.isca-speech.org/archive/Interspeech_2018/abstracts/1025.html).
        Ruoming Pang, Tara Sainath, Rohit Prabhavalkar, Suyog Gupta, Yonghui Wu,
        Shuyuan Zhang, and Chung-Cheng Chiu. Interspeech 2018.

    *   [Contextual Speech Recognition in End-to-End Neural Network Systems
        using Beam
        Search](https://www.isca-speech.org/archive/Interspeech_2018/abstracts/2416.html).
        Ian Williams, Anjuli Kannan, Petar Aleksic, David Rybach, and Tara N.
        Sainath. Interspeech 2018.

    *   [State-of-the-art Speech Recognition With Sequence-to-Sequence Models](https://arxiv.org/abs/1712.01769).
        Chung-Cheng Chiu, Tara N. Sainath, Yonghui Wu, Rohit Prabhavalkar,
        Patrick Nguyen, Zhifeng Chen, Anjuli Kannan, Ron J. Weiss, Kanishka Rao,
        Ekaterina Gonina, Navdeep Jaitly, Bo Li, Jan Chorowski, and Michiel
        Bacchiani. ICASSP 2018.

    *   [End-to-End Multilingual Speech Recognition using Encoder-Decoder Models](https://arxiv.org/abs/1711.01694).
        Shubham Toshniwal, Tara N. Sainath, Ron J. Weiss, Bo Li, Pedro Moreno,
        Eugene Weinstein, and Kanishka Rao. ICASSP 2018.

    *   [Multi-Dialect Speech Recognition With a Single Sequence-to-Sequence
        Model](https://arxiv.org/abs/1712.01541). Bo Li, Tara N. Sainath, Khe
        Chai Sim, Michiel Bacchiani, Eugene Weinstein, Patrick Nguyen, Zhifeng
        Chen, Yonghui Wu, and Kanishka Rao. ICASSP 2018.

    *   [Improving the Performance of Online Neural Transducer Models](https://arxiv.org/abs/1712.01807).
        Tara N. Sainath, Chung-Cheng Chiu, Rohit Prabhavalkar, Anjuli Kannan,
        Yonghui Wu, Patrick Nguyen, and Zhifeng Chen. ICASSP 2018.

    *   [Minimum Word Error Rate Training for Attention-based
        Sequence-to-Sequence Models](https://arxiv.org/abs/1712.01818). Rohit
        Prabhavalkar, Tara N. Sainath, Yonghui Wu, Patrick Nguyen, Zhifeng Chen,
        Chung-Cheng Chiu, and Anjuli Kannan. ICASSP 2018.

    *   [No Need for a Lexicon? Evaluating the Value of the Pronunciation Lexica
        in End-to-End Models](https://arxiv.org/abs/1712.01864). Tara N.
        Sainath, Rohit Prabhavalkar, Shankar Kumar, Seungji Lee, Anjuli Kannan,
        David Rybach, Vlad Schogol, Patrick Nguyen, Bo Li, Yonghui Wu, Zhifeng
        Chen, and Chung-Cheng Chiu. ICASSP 2018.

    *   [Learning hard alignments with variational inference](https://arxiv.org/abs/1705.05524).
        Dieterich Lawson, Chung-Cheng Chiu, George Tucker, Colin Raffel, Kevin
        Swersky, and Navdeep Jaitly. ICASSP 2018.

    *   [Monotonic Chunkwise Attention](https://arxiv.org/abs/1712.05382).
        Chung-Cheng Chiu, and Colin Raffel. ICLR 2018.

    *   [An Analysis of Incorporating an External Language Model into a
        Sequence-to-Sequence Model](https://arxiv.org/abs/1712.01996). Anjuli
        Kannan, Yonghui Wu, Patrick Nguyen, Tara N. Sainath, Zhifeng Chen, and
        Rohit Prabhavalkar. ICASSP 2018.

*   Language understanding

    *   [Semi-Supervised Learning for Information Extraction from Dialogue](https://www.isca-speech.org/archive/Interspeech_2018/abstracts/1318.html).
        Anjuli Kannan, Kai Chen, Diana Jaunzeikare, and Alvin Rajkomar.
        Interspeech 2018.

    *   [CaLcs: Continuously Approximating Longest Common Subsequence for
        Sequence Level Optimization] (http://aclweb.org/anthology/D18-1406).
        Semih Yavuz, Chung-Cheng Chiu, Patrick Nguyen, and Yonghui Wu.
        EMNLP 2018.

*   Speech synthesis

    *   [Hierarchical Generative Modeling for Controllable Speech Synthesis](https://arxiv.org/abs/1810.07217).
        Wei-Ning Hsu, Yu Zhang, Ron J. Weiss, Heiga Zen, Yonghui Wu, Yuxuan
        Wang, Yuan Cao, Ye Jia, Zhifeng Chen, Jonathan Shen, Patrick Nguyen,
        Ruoming Pang. Submitted to ICLR 2019.

    *   [Transfer Learning from Speaker Verification to Multispeaker
        Text-To-Speech Synthesis](https://arxiv.org/abs/1806.04558). Ye Jia, Yu
        Zhang, Ron J. Weiss, Quan Wang, Jonathan Shen, Fei Ren, Zhifeng Chen,
        Patrick Nguyen, Ruoming Pang, Ignacio Lopez Moreno, Yonghui Wu.
        NIPS 2018.

    *   [Natural TTS Synthesis By Conditioning WaveNet On Mel Spectrogram
        Predictions](https://arxiv.org/abs/1712.05884). Jonathan Shen, Ruoming
        Pang, Ron J. Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang,
        Zhifeng Chen, Yu Zhang, Yuxuan Wang, RJ Skerry-Ryan, Rif A. Saurous,
        Yannis Agiomyrgiannakis, Yonghui Wu. ICASSP 2018.

    *   [On Using Backpropagation for Speech Texture Generation and Voice
        Conversion](https://arxiv.org/abs/1712.08363). Jan Chorowski, Ron J.
        Weiss, Rif A. Saurous, Samy Bengio. ICASSP 2018.

## Quick start

### Docker

The docker files in the `docker` directory. They provide a blueprint of how to
install and run the software in multiple configuration.

### Installation

The prerequisites of the toolkit are:

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
