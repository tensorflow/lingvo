# Multimodal Image and Language Networks (Milan)

Milan is a set of libraries and models for multimodal grounded language learning
built on top of [Lingvo](https://github.com/tensorflow/lingvo).

This directory defines models used in [Parekh et al. 2021](https://arxiv.org/abs/2004.15020). If you use this code, please cite:

```
@inproceedings{parekh-etal-2021-cxc,
    title = "Crisscrossed Captions: Extended Intramodal and Intermodal Semantic Similarity Judgments for MS-COCO",
    author = "Parekh, Zarana and  Baldridge, Jason and Cer, Daniel and Waters, Austin and Yang, Yinfei",
    booktitle = "Proceedings of the 16th Conference of the {E}uropean Chapter of the Association for Computational Linguistics (EACL)",
    year = "2021",
    publisher = "Association for Computational Linguistics",
}
```

## Crisscrossed Captions

`params/cxc.py` defines a model called `EfficientNetB4BertAdapter`, which
corresponds to the `DE_I2T` image-text dual encoder in the Crisscrossed Captions
paper.

### Download and prepare the MS-COCO (COCO captions) dataset.

For convenience, we provide a tool to download MS-COCO and augment it with
precomputed BERT embeddings of the image captions. These features are the inputs
for the text side of the `EfficientNetB4BertAdapter` model.

To generate the dataset, run

```shell
python3 -m lingvo.tasks.milan.tools.prepare_coco \
  --output_dir /your/data/dir/here \
  --num_workers 4 \  # Or higher, up to the number of available CPU cores
  --alsologtostderr
```

This will produce a set of TFRecord files in the specified `--output_dir`.

NOTE: This tool downloads data and models from the web (via
`tensorflow_datasets` and `tensorflow_hub`) and altogether may consume 50+ GB of
storage. Users can set the `TFDS_DATA_DIR` and `TFHUB_CACHE_DIR` environment
variables to control where the intermediate data is stored and/or to reuse
cached copies.

[Parekh et al. 2021](https://arxiv.org/abs/2004.15020) show that this model
performs significantly better on MS-COCO if it is first pretrained on the
(much larger) Conceptual Captions dataset. Please see
[Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/) for information about obtaining this dataset.

### Training

In order to train on this prepared data, `EfficientNetB4BertAdapter` needs to
its location on disk. For convenience, the input generator for this model
through an environment variable:

```shell
MILAN_DATASET_CONFIG_JSON="{'data_dir': '/your/data/dir/here'}"
```

Local training runs can then be launched with the usual commands, e.g.

```shell
MILAN_DATASET_CONFIG_JSON=...
TRAINER_DIR='/your/trainer/dir'
mkdir -p "${TRAINER_DIR}"
python3 -m lingvo.trainer \
  --run_locally=cpu --mode=sync --saver_max_to_keep=3 \
  --logdir="${TRAINER_DIR}" \
  --job=trainer_client \
  --model=milan.cxc.EfficientNetB4BertAdapter \
  --logtostderr >& "${TRAINER_DIR}/train.log"
```

To train on a real TPU cluster, we recommend following the instructions outlined
in the ["car" task README](../car/README.md). Note that for TPU training the
prepared TFRecords must be uploaded to Google Cloud Storage. Their location can
be forwarded to the training jobs through `gke_launch` as follows:

```shell
python3 lingvo/tools/gke_launch.py \
  --model=milan.cxc.EfficientNetB4BertAdapter \
  --extra_envs=MILAN_DATASET_CONFIG_JSON='{"data_dir": "gs://your/storage/bucket"}' \
  [other flags...] reload all
```
