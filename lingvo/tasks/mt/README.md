# Local machine translation quick start

To build a local translation model, we provide the tools to run a WMT16
Multimodal EN-DE experiment. This dataset consists of translated image captions
(we are ignoring the associated images). The vocabulary is limited and sentences
are short, meaning you can build a working system quickly, using only CPUs.

## Download and prepare the data

We provide scripts to download the data. The script requires such tools as
`git`, `perl`, `wget`, and standard bash utilities to run properly.

First, build the wordpiece model encoding tool:

```python
bazel build -c opt //lingvo/tools:wpm_encode_file
```

Then, run the data ingestion scripts:

```shell
bash lingvo/tasks/mt/tools/wmtm16_get_data.sh
```

The master script above calls scripts for individual steps. If any step fails,
it can be re-run by hand. Similar steps are described in individual detail under
the Full Machine translation task below. The entire process should complete in
fewer than 10 minutes.

In total, `250MB` of free space are required. The location is configured in
`lingvo/tasks/mt/tools/wmtm16_lib.sh`, defaulting to
`/tmp/wmtm16`. The final output is in
`/tmp/wmtm16/wpm`, stored in `tfrecord` format, which is described
in detail under the Full Machine translation task.

## Training a toy system

MT training is most naturally run on a cluster with many parallel processes, but
for this small task, we can run locally, taking advantage of multiple CPUs on a
reasonable development machine. First, build the training executable:

```shell
bazel build -c opt //lingvo:trainer
```

Then launch:

```shell
mkdir /tmp/wmtm16/log

bazel-bin/lingvo/trainer \
  --run_locally=cpu --mode=sync --saver_max_to_keep=3 \
  --logdir=/tmp/wmtm16/log \
  --job=controller,trainer_client \
  --model=mt.wmtm16_en_de.WmtCaptionEnDeTransformer \
  --logtostderr >& /tmp/wmtm16/log/train.log
```

This will begin training a model, providing statistics on training perplexity
(log_pplx) as training runs. Typically, you will also want to monitor your
progress on a held-out set. Background the above process or open a new terminal
to start two other processes to monitor our progress, one to track the
perplexity assigned to reference translations:

```shell
bazel-bin/lingvo/trainer \
  --run_locally=cpu --mode=sync --saver_max_to_keep=3 \
  --logdir=/tmp/wmtm16/log \
  --job=evaler_Dev \
  --model=mt.wmtm16_en_de.WmtCaptionEnDeTransformer \
  --logtostderr >& /tmp/wmtm16/log/eval_dev.log
```

And one to report the accuracy of the decoder's output when compared to those
reference translations:

```shell
bazel-bin/lingvo/trainer \
  --run_locally=cpu --mode=sync --saver_max_to_keep=3 \
  --logdir=/tmp/wmtm16/log \
  --job=decoder_Dev \
  --model=mt.wmtm16_en_de.WmtCaptionEnDeTransformer \
  --logtostderr >& /tmp/wmtm16/log/decode_dev.log
```

Note that we give the same log directory for both of these processes. The
monitors use the trainer's checkpoints, saved in the log directory, to load
models and compute scores.

Example translation outputs appear in
`/tmp/wmtm16/log/decode_dev.log`. To track development BLEU and
perplexity, run tensorboard on `/tmp/wmtm16/log`. Training should
achieve well over 30 BLEU in fewer than 10,000 steps.

# Full Machine translation task

We provide the tools to run the WMT14 EN-DE setup. The system is described in
[The Best of Both Worlds: Combining Recent Advances in Neural Machine
Translation](http://aclweb.org/anthology/P18-1008)

## Downloading the data

We provide scripts to download the data. The script requires such tools as
`git`, `perl`, `wget`, and standard bash utilities to run properly.

First, build the wordpiece model encoding tool:

```python
bazel build -c opt lingvo/tools:wpm_encode_file
```

Then, run the data ingestion scripts. The master script,
`lingvo/tasks/mt/tools/wmt14_get_data.sh`. If any step
fails, it can be re-run by hand.

In total, `13GB` of free space are required. The location is configured in
`lingvo/tasks/mt/tools/wmt14_lib.sh`, defaulting to
`/tmp/wmt14`.

The final output is in `/tmp/wmt14/wpm`. It comes in the form of
`tf.Example` protos in `tfrecord` format. The examples contain the following
features:

*   `source_id`: An `int64` tensor of length `S`, with wordpiece IDs, always
    terminated with `2`, the ID for `</s>`.
*   `source_padding`: A `float` tensor of zeros, of length `S`.
*   `source_word`: A `string` tensor of wordpieces, always ending in `</s>`, but
    not starting with `<s>`.
*   `target_id`: An `int64` tensor of length `T` with wordpiece IDs, always
    starting with `1`, the ID for `<s>`, but not terminated with `</s>`.
*   `target_padding`: A `float` tensor of zeros, of length `T`.
*   `target_word`: A `string` tensor of wordpiece, always starting with `<s>`
    but not ending in `</s>`.
*   `target_label`: An `int64` tensor of length `T`, with wordpiece IDs, like
    `target_id`, but, instead of starting with `<s>`, is terminated with `</s>`.
*   `target_weight`: A `float` tensor of ones, of length `T`.
*   `natural_order`: An `int64` which is always `1`.

Above, `S` is the length of the source (EN), in word pieces, including the
trailing `</s>`. Similarly, `T` is the length of the target (DE), in word
pieces, including either the beginning `<s>`, or the trailing `</s>`.

Both source and target share the same wordpiece inventory.

### Downloading the Moses scripts

The first step, `wmt14.01.download_moses_scripts.sh`, obtains a copy of the
moses decoder from github. In particular, the tokenizer, cleaning, and SGM
conversion scripts will be used.

### Downloading the training data

On a typical 100Mbit connection, the second step, `wmt14.02.download_train.sh`,
will take roughly half an hour to complete. It will download the following into
the `${ROOT}/raw` directory:

*   `training-parallel-europarl-v7.tgz`: `628M`
*   `training-parallel-commoncrawl.tgz`: `876M`
*   `training-parallel-nc-v9.tgz`: `77M`

### Downloading the dev set

The step `wmt14.03.download_devtest.sh` will result in `${ROOT}/raw`:

*   `dev.tgz`: `17M`
*   `test-filtered.tgz`: `3.2M`

### Unpacking the data

The steps `wmt14.04.unpack_train.sh` and `wmt14.05.unpack_devtest.sh` unpack the
train data into `${ROOT}/unpacked`. Note that we will add SGM-converted test
files into the `${ROOT}/unpacked` directory later.

### Tokenizing the data

The steps `wmt14.06.tokenize_train.sh` and `wmt14.07.tokenize_devtest.sh`
tokenize the training data, into `${ROOT}/tokenized/{train,dev,test}`.

For test, an additional step of converting the SGM is done prior to
tokenization. For train data, a cleaning step is done after tokenization. The
following files are going to be used as input to the next stage:

*   dev/newstest2013.de: `391K`, 3000 lines
*   dev/newstest2013.en: `342K`, 3000 lines
*   test/newstest2014.de: `369K`, 2737 lines
*   test/newstest2014.en: `337K`, 2737 lines
*   train/commoncrawl.clean.de: `329M`, `2.4M` lines
*   train/commoncrawl.clean.en: `305M`, `2.4M` lines
*   train/europarl-v7.clean.de: `315M`, `1.9M` lines
*   train/europarl-v7.clean.en: `276M`, `1.9M` lines
*   train/news-commentary-v9.clean.de: `33M`, `2.4M` lines
*   train/news-commentary-v9.clean.en: `28M`, `2.4M` lines

### Wordpiece model encoding

The next steps, `wmt14.08.wpm_encode_train.sh` and
`wmt14.09.wpm_encode_devtest.sh`, encode the traiing data with the wordpiece
model, as well as transcoding to the final format, to wit `tf.Example` in
`tfrecord` format.

Conversion of the training data will take about one hour on a relatively recent
machine.

## Setting up your cluster

We provide a script running a docker fleet as an debugging example
(`lingvo/docker/run_distributed.py`). Please take a look
at the script to see the cluster configuration.

### Shared filesystem

You will need a cluster with a distributed filesystem, such as `HDFS`, `NFS`,
`sshfs`, or Google [filestore](https://cloud.google.com/filestore/). The
filesystem will be used to store the checkpoints and the trainer binary.

The shared filesystem must mount the data and the log directory, possibly in two
different directories.

### Cluster configuration

For each cluster node, you will need to designate a role and a port number. It
is assumed that nodes in the cluster may occasionally crash or be replaced at
any time. However, when they come back, they must remain under the same
hostname, answering to the same port.

We will train in synchronous mode. For training, you will need the following
nodes:

*   `trainer_client`: (singleton) controls the distributed workers.
*   `controller`: (singleton) saves the checkpoints.
*   `worker`: (8 replicas) runs the actual training.

To continuous evaluation, you will need to add:

*   `decoder_<set>`: runs the beam search on the dev or eval set.
*   `evaler_<set>`: runs the BLEU scoring.

#### Cluster spec

For training, processes need to be aware of each other. This is done by
specifying a cluster spec string. Let us assume the following cluster spec:

```python
{
    "worker": [
        "wk0.example.com:43222",
        "wk1.example.com:43222",
        "wk2.example.com:43222",
    ],
    "controller": ["cont.example.com:43214",],
    "trainer_client": ["tc.example.com:24601"],
}
```

The `trainer_client` is running on a machine under hostname `tc.example.com`,
with an available port `24601`, and each worker is on a machine with hostname
`wk0.example.com`, `wk1.example.com`, and `wk2.example.com` respectively. There
is no naming convention required for the hostnames. The cluster spec is
specified as follows:

```
cluster_spec := <role_spec> ( '@', <role_spec> )*
role_spec := (<role>, '=', <machine_spec>, (',' <machine_spec>)* )
machine_spec := (<host>, ':', <port>)
```

In the example above, the role spec for the `worker` role would be:

```
worker=wk0.example.com:43222,wk1.example.com:43222,wk2.example.com:43222
```

and the final cluster spec would be:

```
worker=wk0.example.com:43222,wk1.example.com:43222,wk2.example.com:43222@\
controler=cont.example.com:43214@\
trainer_client=tc.example.com:24601
```

For decoding, each process has its own cluster spec, for instance,
`decoder_test=dec.example.com:3984`.

### Uploading the binary

There is a single binary which can be obtained as follows:

```shell
bazel build -c opt lingvo:trainer.par
```

You may then copy the binary to the shared filesystem. Henceforth, the trainer
will be invoked using `trainer.par --...` from the unpacked directory. The
`trainer.par` binary is a self-contained executable in a zip format.

All cluster nodes must have tensorflow installed, with the same version as the
one that was used to build lingvo.

### Uploading the data

You will need to upload the data files to the shared filesystem. The location
must match the one specified in
`lingvo/tasks/mt/params/wmt14_en_de.py` as `DATADIR`.

## Running the model

You are now ready to run the model.

For each of the roles (`trainer_client`, `controller`, `decoder_dev`, etc), you
must run the trainer with the proper cluster spec:

```shell
trainer.par --cluster_spec=<cluster_spec> \
  --model=mt.wmt14_en_de.WmtEnDeTransformerSmall \
  --job=<role> --task=<task_id> --mode=sync --logtostderr \
  --logdir=<shared_log_dir>
```

as described above.

## Examining results

To examine the results, it is best to run a tensorboard process on the log
directory.

<!-- TODO(drpng): complete this -->
