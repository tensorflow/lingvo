# Lint as: python3
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

from lingvo import compat as tf
from lingvo.core import base_input_generator
from lingvo.core import datasource
from lingvo.core import generic_input
from lingvo.core import hyperparams
from lingvo.core import py_utils

# Samples whose buckets are greater or equal to this value will
# be dropped and not sent to the trainer.
BUCKET_UPPER_BOUND = 9999


def _ParseSequenceExample(record, feature_map, context_map):
  """Parse a SequenceExample, adding the context features to the features."""
  context, features = tf.io.parse_single_sequence_example(
      serialized=record,
      context_features=context_map,
      sequence_features=feature_map)
  # Add all keys from context to features. Keys must not overlap.
  common_keys = set(context.keys()) & set(features.keys())
  if common_keys:
    raise ValueError(
        'Keys {} are present in context and features.'.format(common_keys))
  features.update(context)
  return features


def _TextInput(record, feature_map):
  # record is a Tensor containing a string line.
  if feature_map:
    raise ValueError('For PlainText datasets, FeatureMap() must be empty.')
  return {'line': record}


# Supported raw record types and the corresponding parsing functions.
_PARSING_FUNCTIONS = {
    'EXAMPLE': tf.io.parse_single_example,
    'SEQUENCE_EXAMPLE': _ParseSequenceExample,
    'TEXT': _TextInput,
}


class _BaseExtractor(base_input_generator.BaseInputGeneratorFromFiles):
  """The base extractor for all lingvo car task datasets.

  Subclasses should define and pass in a custom dictionary of extractors to
  select which fields from car datasets to output from an input
  generator.

  Preprocessors are applied to all the extracted outputs jointly, in the
  specified sequence.
  """

  @classmethod
  def Params(cls, extractors):
    """Defaults params.

    Args:
      extractors: An hyperparams.Params of extractor names to Extractors. A few
        extractor types are *required*:
        'labels': A LabelExtractor.Params().

    Returns:
      A base_layer Params object.
    """
    p = super().Params()
    p.Define('extractors', extractors,
             'A hyperparams.Params() of FieldsExtractors.')
    p.Define('preprocessors', hyperparams.Params(),
             'A Params() of Preprocessors.')
    p.Define(
        'preprocessors_order', [],
        'A list corresponding to flattened keys in preprocessors '
        'Params(). This specifies the execution order of the '
        'preprocessors.')
    p.Define('record_type', 'EXAMPLE',
             'Raw record format, default to tf.Example.')
    p.Define(
        'batched_input', False,
        'If true, ProcessFeatures() is expected to receive batches of Tensors, '
        'rather than single example Tensors.')

    p.batch_size = 64
    p.use_per_host_infeed = True
    p.file_random_seed = 0

    p.file_datasource = datasource.SimpleDataSource.Params()

    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    # Instantiate every extractor as a child layer.
    self._extractors = py_utils.NestedMap()
    for (name, eparam) in p.extractors.IterParams():
      name = name.replace('.', '_')
      self.CreateChild(name, eparam)
      self._extractors[name] = self.children[name]

    # Instantiate preprocessors based on their ordering.
    flattened_processors = dict(p.preprocessors.IterParams())

    # Validate that all keys in preprocessors_order appear are valid.
    if not set(p.preprocessors_order).issubset(
        list(flattened_processors.keys())):
      raise ValueError(
          'preprocessor_order specifies keys which were not found in '
          'preprocessors. preprocessors_order={} preprocessors keys={}'.format(
              p.preprocessors_order, list(flattened_processors.keys())))

    preprocessors = [flattened_processors[key] for key in p.preprocessors_order]
    self.CreateChildren('preprocessors', preprocessors)

    dtypes = self.DType()
    shapes = self.Shape()
    if not dtypes.IsCompatible(shapes):
      raise ValueError('{} vs. {}'.format(dtypes.DebugString(),
                                          shapes.DebugString()))
    dtypes.Pack(list(zip(dtypes.Flatten(),
                         shapes.Flatten()))).VLog(0, 'InpGen: ')

  def FeatureMap(self):
    """Get a mapping from feature names to feature tensors."""
    feature_map = {}
    self._extractors.Transform(lambda e: feature_map.update(e.FeatureMap()))
    return feature_map

  def ContextMap(self):
    """Get a mapping from context names to context tensors.

    ContextMap() is used for tf.SequenceExample datasets to extract
    context_features.  In that scenario, FeatureMap() is used to extract
    the sequence_features.

    Returns:
      A map from context keys to context features.
    """
    context_map = {}
    self._extractors.Transform(lambda e: context_map.update(e.ContextMap()))
    return context_map

  def Shape(self):
    shapes = self._extractors.Transform(lambda x: x.Shape())
    for preprocessor in self.preprocessors:
      shapes = preprocessor.TransformShapes(shapes)

    if self.params.batched_input:
      shapes = shapes.Transform(lambda x: [self.InfeedBatchSize()] + x)

    return shapes

  def DType(self):
    dtypes = self._extractors.Transform(lambda x: x.DType())
    for preprocessor in self.preprocessors:
      dtypes = preprocessor.TransformDTypes(dtypes)
    return dtypes

  @property
  def class_names(self):
    raise NotImplementedError('Return a list of class names strings.')

  def _DataSourceFromFilePattern(self, file_pattern, input_source_weights=None):

    def Proc(record):
      """Parses a serialized tf.Example record."""
      bucket, outputs = self.ExtractUsingExtractors(record)
      return outputs.Flatten(), bucket

    # Ensure buckets [BUCKET_UPPER_BOUND, inf) are dropped.
    args = self.CommonInputOpArgs()
    args['bucket_upper_bound'] = [BUCKET_UPPER_BOUND - 1]
    batched_outputs, bucket_keys = generic_input.GenericInput(
        processor=Proc,
        file_pattern=file_pattern,
        input_source_weights=input_source_weights,
        **args)
    ret = self._NestedMapFromBatchedOutputs(batched_outputs)
    ret.bucket_keys = bucket_keys
    return ret

  def ProcessFeatures(self, features):
    """Process extracted features.

    Args:
      features: A dict of extracted Tensors from the records.

    Returns:
      A tuple of tensors:

      - bucket_id: A scalar int Tensor.
      - extracted: a NestedMap of Tensors extracted.
    """

    def ExtractAndFilter(e):
      with tf.name_scope(e.params.name):
        with tf.name_scope('extract'):
          # Filter out extracted features from other extractors.
          filtered_features = {}
          if self.params.record_type == 'TEXT':
            # Text extractors only produce {'line': record} and their
            # FeatureMap() is empty, so don't do any filtering.
            filtered_features = features
          else:
            filtered_keys = e.FeatureMap().keys() | e.ContextMap().keys()
            filtered_features = {
                k: v for k, v in features.items() if k in filtered_keys
            }
          try:
            if self.params.batched_input:
              extracted = e.ExtractBatch(filtered_features)
            else:
              extracted = e.Extract(filtered_features)
          except Exception as exc:  # pylint:disable=bare-except
            # Raise exception with context about which extractor failed.
            raise RuntimeError('Failed running extractor '
                               f'{e.params.name}: {repr(exc)}. '
                               'See above exception for details.') from exc
        with tf.name_scope('filter'):
          if self.params.batched_input:
            bucket = e.FilterBatch(extracted)
          else:
            bucket = e.Filter(extracted)
      return bucket, extracted

    bucket_extracted = self._extractors.Transform(ExtractAndFilter)
    buckets = bucket_extracted.Transform(lambda x: x[0])
    extracted = bucket_extracted.Transform(lambda x: x[1])

    # Return the maximum bucket id so that any extractor can decide whether
    # to filter the entire example.
    max_bucket = tf.reduce_max(buckets.Flatten())

    def NullLike():
      """A function to return the same Tensor signature as Preprocess.

      This is necessary for the tf.cond() to avoid executing the preprocessor
      for examples that are going to be dropped because it exceeds the bucket
      limit; tf.cond() requires that the output of both branches yields the same
      structure.

      Returns:
        A structure with the same Tensor dtype as the output of
        Preprocess.
      """
      shapes = self.Shape()
      rets = []
      for dtype, shape in zip(self.DType().Flatten(), shapes.Flatten()):
        if shape.is_fully_defined():
          rets += [tf.zeros(dtype=dtype, shape=shape)]
        else:
          rets += [tf.zeros(dtype=dtype, shape=[])]  # Our best guess.
      return shapes.Pack(rets)

    def Preprocess(extracted):
      for key, preprocessor in zip(self.params.preprocessors_order,
                                   self.preprocessors):
        with tf.name_scope(key), tf.name_scope(preprocessor.params.name):
          if self.params.batched_input:
            extracted = preprocessor.TransformBatchedFeatures(extracted)
          else:
            extracted = preprocessor.TransformFeatures(extracted)
      return extracted

    # If the extractor wants to filter the example, don't run the preprocessor.
    #
    # Preprocessors can then assume that only examples that pass filtering will
    # be executed.
    #
    # Note that the NullLike branch may return tensors with shapes different
    # from self.Shape().
    final_output = tf.cond(
        tf.less(max_bucket, BUCKET_UPPER_BOUND), lambda: Preprocess(extracted),
        NullLike)

    return max_bucket, final_output

  def ExtractUsingExtractors(self, record):
    """Extracts Tensors from a tf.Example record using self.extractors.

    Args:
      record: A tf.Example input to pass to tf.io.parse_single_example.

    Returns:
      A tuple of tensors:

      - bucket_id: A scalar int Tensor.
      - extracted: a NestedMap of Tensors extracted.
    """
    if self.params.record_type not in _PARSING_FUNCTIONS:
      raise ValueError('Invalid record_type: {}'.format(
          self.params.record_type))

    parsing_fn = _PARSING_FUNCTIONS[self.params.record_type]
    if self.params.record_type == 'SEQUENCE_EXAMPLE':
      features = parsing_fn(record, self.FeatureMap(), self.ContextMap())
    else:
      features = parsing_fn(record, self.FeatureMap())

    return self.ProcessFeatures(features)

  def GetCpuPassthroughKeys(self):
    dtypes = self.DType()
    # By default, string types in the input are passthrough types.
    string_dtypes = dtypes.Filter(lambda x: x == tf.string)
    return [v[0] for v in string_dtypes.FlattenItems()]

  # TODO(vrv): Remove once all users are migrated.
  def _NestedMapFromBatchedOutputs(self, outputs):
    return self.NestedMapFromBatchedOutputs(outputs)

  def NestedMapFromBatchedOutputs(self, outputs):
    """Create a NestedMap from a list/tuple of batched outputs.

    Args:
      outputs: A tuple or list of Tensors whose order matches the flattened
        structure of Shape() and DType().

    Returns:
      A NestedMap reconstructing the structure of the output of extractors
        and preprocessors, where each Tensor's shape is statically
        padded/trimmed to match the Shape() specification.

    Raises:
      ValueError: If `outputs` contains a shape that is not fully
        defined.
      AssertionError: If any shape of a Tensor in `outputs` cannot be
        PadOrTrimTo'd by the corresponding Shape() specification.
    """
    batch_size = self.InfeedBatchSize()
    shapes = self.Shape()
    shapes.VLog(0, 'input extractor shape: ')
    flatten_shapes = shapes.Flatten()
    dtypes = self.DType()
    flatten_dtypes = dtypes.FlattenItems()
    assert len(flatten_shapes) == len(outputs), '{} vs. {}'.format(
        len(flatten_shapes), len(outputs))
    assert len(flatten_dtypes) == len(outputs), '{} vs. {}'.format(
        len(flatten_dtypes), len(outputs))

    rets = []
    assertion_errors = []
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
      curr_shape = py_utils.GetShape(output)
      padded_shape = shape.as_list()
      if not self.params.batched_input:
        padded_shape = [batch_size] + padded_shape

      try:
        padded = py_utils.PadOrTrimTo(output, padded_shape)
        rets.append(padded)
      except AssertionError as e:
        assertion_errors += [f'{name}: {e}, ({curr_shape} vs. {padded_shape}']

    if assertion_errors:
      raise AssertionError('Mismatched shapes:\n' + '\n'.join(assertion_errors))

    rets = shapes.Pack(rets)

    # String tensors in rets will be filtered out from being sent to the
    # device automatically, and instead will be present in CPU passthrough.
    return rets
