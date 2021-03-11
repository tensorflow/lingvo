# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Helpers for reading general image-text data and labeling example pairs."""

import json
import os

import lingvo.compat as tf
from lingvo.core import hyperparams
from lingvo.tasks.milan import common_schema
from lingvo.tasks.milan import dataset_spec
from lingvo.tasks.milan import labels as label_lib
from lingvo.tasks.milan import utils


def _SimpleImageCaptionDatasetLabeler(
    image_id_feature) -> label_lib.LabelFnType:
  """Creates a simple labeler that works with (many) image-caption datasets.

  This labeler assumes each example in the dataset contains one image. If it
  sees two distinct in-batch examples with same image (as identified by the
  `image_id_feature`), the labeler marks the pair to be dropped from the loss
  calculation.

  This has two high-level effects:
    - Duplicate-example pairs are dropped. This is good; it means duplicates
      aren't incorrectly assigned negative labels.
    - Any co-caption pairs (examples with the same image but different captions)
      are also dropped. This is non-optimal but fine in most cases. (Ideally
      image -> caption pairs from these examples should get positive labels,
      but doing so requires a more specialized labeling function. Dropping them
      is easier and not wrong.)

  Args:
    image_id_feature: Name of a scalar feature that uniquely identifies the
      image in each example.

  Returns:
    The instantiated label function (callable).
  """
  return label_lib.ExamplePairLabeler(drop_pairs_that_match=image_id_feature)


class ImageTextTFRecords(dataset_spec.TFRecordDatasetSpec):
  """Generic dataset of image-text examples stored in TFRecords.

  See `common_shema` for the assumed format of the on-disk examples.
  """

  @classmethod
  def Params(cls):
    p = hyperparams.InstantiableParams(cls)
    p.Define('split_paths', dict(train='train-*', dev='dev-*', test='test-*'),
             'Paths of each split of the dataset (split name => filepattern).')
    p.Define('data_dir', '',
             'Optional; base directory of any relative paths in split_paths.')
    p.Define('bert_max_length', 48,
             'Max token length of the precomputed BERT caption embeddings.')
    return p

  @classmethod
  def ParamsFromEnv(cls,
                    environment_variable='MILAN_DATASET_CONFIG_JSON',
                    die_if_unset: bool = False):
    params_json = os.getenv(environment_variable)
    if params_json is not None:
      params = cls.Params().Set(**json.loads(params_json))
      tf.logging.info('%s: Loaded params from %s: %s', cls.__name__,
                      environment_variable, params.ToText())
      return params

    message = (f'{cls.__name__}; Must set environment variable '
               f'"{environment_variable}" to configure dataset.')
    if die_if_unset:
      raise ValueError(message)
    else:
      tf.logging.warning(message)
      return cls.Params().Set(
          data_dir=f'/please-set-{environment_variable}-to-configure-dataset')

  def __init__(self, params: hyperparams.Params):
    self.params = params

    # Resolve any relative data paths.
    split_paths = {
        name: os.path.join(params.data_dir, path)
        for name, path in params.split_paths.items()
    }

    # The input TFRecords are assumed to hold `tf.train.Example`s with (at
    # least) the features in this schema. (The schema specifies what we'll
    # extract from them, via `tf.io.parse_example()`.)
    tf_example_schema = common_schema.ImageFeatures()
    tf_example_schema.update(
        common_schema.TextFeatures(bert_embeddings_shape=(None, 768)))

    def _Transform(batch):
      batch = batch.copy()
      # Crop or pad BERT sequences to the fixed `bert_max_length`. Fully defined
      # static shapes are necessary for TPU training.
      batch['text/bert/token_features'] = utils.PadOrTrimDimension(
          batch['text/bert/token_features'], params.bert_max_length, axis=-2)
      return batch

    super().__init__(
        split_paths=split_paths,
        schema=tf_example_schema,
        transform=_Transform,
        label_fn=_SimpleImageCaptionDatasetLabeler(image_id_feature='image/id'))
