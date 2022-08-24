# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""A DataSource for Tensorflow Datasets (TFDS).

These two classes were isolated from datasources.py for build efficiency.
"""

import functools

from lingvo.core import datasource
import tensorflow_datasets as tfds


class TFDSInput(datasource.TFDatasetSource):
  """Load tfds datasets."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('dataset', None, 'The tfds dataset to load.')
    p.Define(
        'split', None,
        'The split to load. See https://www.tensorflow.org/datasets/splits.')
    p.Define(
        'load_fn', '',
        'An input_generator method name to call to load data. It must accept '
        '(info, features_dict) and return a NestedMap. If not specified, a '
        'dataset containing features_dict is returned.')
    p.Define('shuffle_buffer_size', 10000,
             'Number of records buffered for random shuffling.')
    return p

  def GetDataset(self):
    p = self.params
    if not p.dataset or not p.split:
      raise ValueError('A dataset and split must be specified.')

    dataset, info = tfds.load(
        p.dataset,
        split=p.split,
        download=False,  # no need to download internally.
        shuffle_files=not self.cluster.require_sequential_input_order,
        with_info=True)

    if p.load_fn:
      dataset = dataset.map(
          functools.partial(getattr(self._input_generator, p.load_fn), info),
          **self._map_args)

    if not self.cluster.require_sequential_input_order:
      dataset = dataset.shuffle(p.shuffle_buffer_size)
      dataset = dataset.repeat()
    return dataset
