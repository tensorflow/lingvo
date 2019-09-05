# Lint as: python2, python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Base extractor interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lingvo import compat as tf
from lingvo.core import base_input_generator
from lingvo.core import base_layer
from lingvo.core import generic_input
from lingvo.core import hyperparams
from lingvo.core import py_utils


# Items exceeding this value will be dropped and not sent to the trainer.
BUCKET_UPPER_BOUND = 9999


def _ParseSequenceExample(record, feature_map):
  _, features = tf.parse_single_sequence_example(
      serialized=record, context_features=None, sequence_features=feature_map)
  return features


# Supported raw record types and the corresponding parsing functions.
_PARSING_FUNCTIONS = {
    'EXAMPLE': tf.parse_single_example,
    'SEQUENCE_EXAMPLE': _ParseSequenceExample,
}


class _BaseExtractor(base_input_generator.BaseInputGeneratorFromFiles):
  """The base extractor for all V06+-derived Minecraft datasets.

  Subclasses should define and pass in a custom dictionary of extractors to
  select which fields from V06+ datasets to output from an input
  generator.

  Preprocessors are applied to all the extracted outputs jointly, in the
  specified sequence.
  """

  @classmethod
  def Params(cls, extractors):
    """Defaults params.

    Args:
      extractors: An NestedMap of extractor names to Extractors.  A few
        extractor types are *required*:
        'labels': A LabelExtractor.Params().

    Returns:
      A base_layer Params object.
    """
    p = super(_BaseExtractor, cls).Params()
    p.Define('extractors', extractors, 'A NestedMap of FieldsExtractors.')
    p.Define('preprocessors', hyperparams.Params(),
             'A Params() of Preprocessors.')
    p.Define(
        'preprocessors_order', [],
        'A list corresponding to flattened keys in preprocessors '
        'Params(). This specifies the execution order of the '
        'preprocessors.')
    p.Define('record_type', 'EXAMPLE',
             'Raw record format, default to tf.Example.')

    p.batch_size = 64
    p.use_per_host_infeed = True
    p.file_random_seed = 0
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(_BaseExtractor, self).__init__(params)
    p = self.params

    # Instantiate every extractor as a child layer.
    self._extractors = []
    for (name, eparam) in p.extractors.FlattenItems():
      name = name.replace('.', '_')
      self.CreateChild(name, eparam)
      self._extractors += [self.children[name]]
    self._extractors = p.extractors.Pack(self._extractors)

    # Instantiate preprocessors based on their ordering.
    flattened_processors = dict(p.preprocessors.IterParams())

    # Validate that all keys in preprocessors_order appear are valid.
    if not set(p.preprocessors_order).issubset(flattened_processors.keys()):
      raise ValueError(
          'preprocessor_order specifies keys which were not found in '
          'preprocessors. preprocessors_order={} preprocessors keys={}'.format(
              p.preprocessors_order, flattened_processors.keys()))

    preprocessors = [flattened_processors[key] for key in p.preprocessors_order]
    self.CreateChildren('preprocessors', preprocessors)

    dtypes = self.DType()
    shapes = self.Shape()
    dtypes.Pack(zip(dtypes.Flatten(), shapes.Flatten())).VLog(0, 'InpGen: ')

  def Shape(self):
    shapes = self._extractors.Transform(lambda x: x.Shape())
    for preprocessor in self.preprocessors:
      shapes = preprocessor.TransformShapes(shapes)
    return shapes

  def DType(self):
    dtypes = self._extractors.Transform(lambda x: x.DType())
    for preprocessor in self.preprocessors:
      dtypes = preprocessor.TransformDTypes(dtypes)
    return dtypes

  @property
  def class_names(self):
    raise NotImplementedError('Return a list of class names strings.')

  def _DataSourceFromFilePattern(self, file_pattern):

    def Proc(record):
      """Parses a serialized tf.Example record."""
      bucket, outputs = self.ExtractUsingExtractors(record)
      return outputs.Flatten(), bucket

    # Ensure buckets above BUCKET_UPPER_BOUND are dropped.
    args = self.CommonInputOpArgs()
    args['bucket_upper_bound'] = [BUCKET_UPPER_BOUND - 1]
    return generic_input.GenericInput(
        processor=Proc, file_pattern=file_pattern, **args)

  def ExtractUsingExtractors(self, record):
    """Extracts Tensors from a tf.Example record using self.extractors.

    Args:
      record: A tf.Example input to pass to tf.parse_single_example.

    Returns:
      bucket_id: A scalar int Tensor.
      extracted: a NestedMap of Tensors extracted.
    """
    feature_map = {}
    self._extractors.Transform(lambda e: feature_map.update(e.FeatureMap()))

    if self.params.record_type not in _PARSING_FUNCTIONS:
      raise ValueError('Invalid record_type: {}'.format(
          self.params.record_type))
    parsing_fn = _PARSING_FUNCTIONS[self.params.record_type]
    features = parsing_fn(record, feature_map)

    def ExtractAndFilter(e):
      with tf.name_scope(e.params.name):
        with tf.name_scope('extract'):
          extracted = e.Extract(features)
        with tf.name_scope('filter'):
          bucket = e.Filter(extracted)
      return bucket, extracted

    bucket_extracted = self._extractors.Transform(ExtractAndFilter)
    buckets = bucket_extracted.Transform(lambda x: x[0])
    extracted = bucket_extracted.Transform(lambda x: x[1])

    # Return the maximum bucket id so that any extractor can decide whether
    # to filter the entire example.

    for key, preprocessor in zip(self.params.preprocessors_order,
                                 self.preprocessors):
      with tf.name_scope(key), tf.name_scope(preprocessor.params.name):
        extracted = preprocessor.TransformFeatures(extracted)

    return tf.reduce_max(buckets.Flatten()), extracted

  def InputBatch(self):
    batched_outputs, bucket_keys = self._BuildDataSource()
    ret = self._NestedMapFromBatchedOutputs(batched_outputs)
    ret.bucket_keys = bucket_keys
    return ret

  def _NestedMapFromBatchedOutputs(self, outputs):
    """Create a NestedMap from a tuple of outputs from generic_input_op."""
    batch_size = self.InfeedBatchSize()
    shapes = self.Shape()
    shapes.VLog(0, 'input extractor shape: ')
    flatten_shapes = shapes.Flatten()
    dtypes = self.DType()
    assert dtypes.IsCompatible(shapes), '{} vs. {}'.format(
        dtypes.DebugString(), shapes.DebugString())
    flatten_dtypes = dtypes.FlattenItems()
    assert len(flatten_shapes) == len(outputs), '{} vs. {}'.format(
        len(flatten_shapes), len(outputs))
    assert len(flatten_dtypes) == len(outputs), '{} vs. {}'.format(
        len(flatten_dtypes), len(outputs))

    rets = []
    for (output, (name, dtype), shape) in zip(outputs, flatten_dtypes,
                                              flatten_shapes):
      assert dtype == output.dtype, '{}: {} vs. {}'.format(
          name, dtype, output.dtype)
      # Pad every output to make shapes fixed according to the corresponding
      # declared shape, since the shapes of outputs are lost through
      # generic_input_op.
      try:
        shape.assert_is_fully_defined()
      except ValueError as e:
        raise ValueError('Invalid shape for %s: %s' % (name, e))
      padded = py_utils.PadOrTrimTo(output, [batch_size] + shape.as_list())
      rets += [padded]

    rets = shapes.Pack(rets)
    if py_utils.use_tpu():
      # Drops tf.string tensors, which is not supported on TPUs.
      rets = rets.Filter(lambda x: x.dtype != tf.string)
    return rets
