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
"""Milan input generator."""

import functools
import re

from absl import logging
from lingvo import compat as tf
from lingvo.core import base_input_generator
from lingvo.core import py_utils


class MilanInputGenerator(base_input_generator.BaseInputGenerator):
  """Common input generator for Milan.

  This class mostly wraps a user-provided `dataset_fn`, which when called
  returns a `tf.data.Dataset` of batched examples to use as input. The function
  must be callable with a batch_size argument, as

  ```dataset = p.dataset_fn(batch_size=42, **p.dataset_fn_kwargs)```.

  The `preprocessors` param enables features to be transformed through a layer
  before being fed to the model. These are configured as a map of feature name
  to layer params. For example, setting ::

    preprecessors['foo'] = FooPreprocessor.Params()

  causes feature `foo` to be replaced with the output of `FooPreprocessor`.
  """

  @classmethod
  def Params(cls):
    """Returns `Params` object for configuring this input generator.

    Callers must set `dataset_fn` before before instantiating the input
    generator.
    """
    p = super().Params()
    p.Define(
        'dataset_fn', None, 'Function that constructs a tf.data.Dataset '
        'of input examples. Must be callable as: '
        'dataset_fn(batch_size=42, **dataset_fn_kwargs).')
    p.Define(
        'dataset_fn_kwargs', {}, 'Dict of kwargs to pass to dataset_fn(), '
        'e.g. to override default options. May not contain "batch_size".')

    p.Define(
        'features_to_read', [], 'Regular expression(s) of feature names. '
        'If empty, defaults to all features.')
    p.Define('preprocessors', {},
             'Dictionary of input_feature_name => layer_params.')
    p.Define('preprocess_parallelism', tf.data.experimental.AUTOTUNE,
             'Number of batches to preprocess in parallel.')

    # Set reasonable defaults.
    p.name = 'milan_input_generator'
    p.batch_size = 32
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    if 'batch_size' in p.dataset_fn_kwargs:
      raise ValueError('dataset_fn_kwargs may not contain "batch_size".')
    if not isinstance(p.features_to_read, (tuple, list, type(None))):
      raise ValueError(
          'Expected sequence type for "features_to_read"; got {}'.format(
              type(p.features_to_read)))
    if p.preprocessors:
      self._preprocessor_input_names, preprocessor_layer_params = list(
          zip(*list(p.preprocessors.items())))
      self.CreateChildren('_preprocessors', list(preprocessor_layer_params))

  def GetPreprocessedInputBatch(self):
    p = self.params

    # Dataset of parsed examples.
    dataset = p.dataset_fn(
        batch_size=self.InfeedBatchSize(), **p.dataset_fn_kwargs)
    dataset = dataset.map(
        # Force retracing if self.do_eval changes.
        functools.partial(self._PreprocessInputBatch, do_eval=self.do_eval),
        num_parallel_calls=p.preprocess_parallelism)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    iterator = dataset.make_one_shot_iterator()
    input_batch = iterator.get_next()
    return input_batch

  def _FilterFeaturesByName(self, features):
    p = self.params
    if not p.features_to_read:
      return features

    union_regex = re.compile('({})'.format('|'.join(p.features_to_read)))
    return features.FilterKeyVal(lambda k, _: union_regex.match(k))

  @tf.function(experimental_relax_shapes=True)
  def _PreprocessInputBatch(self, input_batch, do_eval: bool):
    del do_eval  # Only exists to force separate train/eval mode traces.
    input_batch = py_utils.NestedMap(input_batch)
    input_batch = self._FilterFeaturesByName(input_batch)

    # Apply preprocessors.
    if self.params.preprocessors:
      for input_name, preprocessor in zip(self._preprocessor_input_names,
                                          self._preprocessors):
        input_batch[input_name] = preprocessor(input_batch[input_name])

    # Remove any string features if training on TPU.
    if py_utils.use_tpu():
      input_batch = input_batch.Filter(lambda t: t.dtype != tf.string)
    logging.info('Final input batch: %s', input_batch)
    return input_batch
