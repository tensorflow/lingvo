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
"""Generic input."""

import lingvo.compat as tf
from lingvo.core import ops
from lingvo.core import py_utils
from tensorflow.python.util import tf_inspect  # pylint: disable=g-direct-tensorflow-import


# This cache is used to reuse the GenericInputV2 resources.
# key: the DataSource object that calls into `GenericInput`.
# value: a three-element tuple that will be used as inputs to
# `GenericInputV2GetNext`:
# (data_resource_id, batch_types, batch_template)
_GENERIC_CACHE_V2 = {}

# Escape hatch for existing Eager-mode tests that happen to use GenericInput op.
_IS_GENERIC_INPUT_V2_ALLOWED_IN_EAGER = False


def SetAllowGenericInputV2InEager(allowed=True):
  global _IS_GENERIC_INPUT_V2_ALLOWED_IN_EAGER
  _IS_GENERIC_INPUT_V2_ALLOWED_IN_EAGER = allowed


def IsGenericInputV2AllwedInEager():
  return _IS_GENERIC_INPUT_V2_ALLOWED_IN_EAGER


_MISSING_KEY_ERR = (
    'In tf.function or pure Eager, GenericInputV2 ops will be called. '
    'Each GenericInputV2 op requires a unique key for identification. '
    'To provide this key, you can specify the keyword arg '
    '`generic_input_v2_key` when calling this function.')


def _ParseProcessor(processor):
  """Parses python callable `processor` into a TF concrete function."""
  output_tmpl = py_utils.NestedMap()

  @tf.function(autograph=False)
  def _FlatOutputProcessor(source_id, record):
    """Returns a flattened list of 'processor(inputs)'."""
    processor_spec = tf_inspect.getargspec(processor)
    tf.logging.debug('GenericInput.processor.argspec=%s', processor_spec)
    processor_args = set(processor_spec.args) - set(['self'])
    if len(processor_args) == 1:
      output, bucketing_key = processor(record)
    elif processor_args == set(['source_id', 'record']):
      output, bucketing_key = processor(source_id=source_id, record=record)
    else:
      raise ValueError(
          'GenericInput: processor should take either a single arg '
          'or two args named as "source_id" and "record". '
          'Actual: %s' % processor_args)
    if isinstance(output, list):
      assert output
      assert all(isinstance(x, tf.Tensor) for x in output), '{}'.format(output)
    else:
      assert isinstance(output, py_utils.NestedMap), '{}'.format(output)
      assert output
      assert all(
          isinstance(x, tf.Tensor) for x in output.Flatten()), '{}'.format(
              output.DebugString())
    bucketing_key = tf.cast(bucketing_key, tf.int32)
    tf.logging.debug('Processor outputs=%s bucketing_key=%s', output,
                     bucketing_key)
    output_tmpl.out_values = output
    flat_output_tmpl = output_tmpl.Flatten()
    tf.logging.debug('Processor flat outputs=%s', flat_output_tmpl)
    tf.logging.debug('extra_inputs=%s extra_args=%s extra_vars=%s',
                     py_utils.GetExtraInputs(), py_utils.GetExtraArgs(),
                     py_utils.GetExtraVars())
    assert not py_utils.GetExtraArgs(), (
        'fns {} is not pure: extra_args={}'.format(processor,
                                                   py_utils.GetExtraArgs()))
    return flat_output_tmpl + [bucketing_key]

  with py_utils.GlobalStepContext(None):
    # Hide global_step tensor from being captured by _FlatOutputProcessor.
    proc_fn = _FlatOutputProcessor.get_concrete_function(
        tf.TensorSpec([], tf.int32), tf.TensorSpec([], tf.string))

  out_types = [
      tf.DType(a.type) for a in proc_fn.function_def.signature.output_arg
  ]
  assert out_types[-1] == tf.int32, ('%s is not expected.' % out_types[-1])
  return proc_fn, out_types, output_tmpl


def GenericInput(processor, **kwargs):
  """Builds a generic input pipeline.

  Example usage:

    def ParseRecord(record):
      # Given a tf.string record, return a (NestedMap, bucketing key) pair.
      feature_map = ...
      features = tf.io.parse_single_example(record, feature_map)
      # Each example is represented by a NestedMap of tensors (without a
      # batch dimension).
      example = py_utils.NestedMap(field1=..., field2=...)
      # bucketing_key is a scalar convertible to tf.int32.
      # Use 1 if all examples are of the same size.
      bucketing_key = 1
      return example, bucketing_key

    input_batch, bucket_keys = GenericInput(ParseRecord, file_pattern=..., ...)
    # input_batch is a NestedMap of tensors, where dim 0 of each tensor
    # represents the batch dimension.
    input_batch.field1 = ...

  ParseRecord can also take both 'source_id' and 'record' as inputs (the arg
  names must be exactly 'source_id' and 'record'):

    def ParseRecord(source_id, record):
      # Given a tf.int32 source_id and a tf.string record, return a (NestedMap,
      # bucketing key) pair.
      example = py_utils.NestedMap(source_id=source_id, ...)
      ...
      return example, bucketing_key

    input_batch, bucket_keys = GenericInput(ParseRecord, file_pattern=..., ...)

  Args:
    processor: a function that takes either a tf.string record or a
      (source_id: tf.int32, record: tf.string) pair as input and returns a tuple
      (output, bucketing_key). `output` must be a NestedMap or a list of tensors
      representing an example. `bucketing_key` must be a scalar convertible to
      a tf.int32 tensor that represents the bucketing key (e.g., sequence
      length for sequence inputs). If `bucketing_key` is a negative number,
      the record is dropped.
    **kwargs: additional keyword args for x_ops.generic_input.

  Returns:
    A tuple of (outputs, bucket_keys):

    - outputs: a NestedMap or a tuple of tensors, similar to `processor`'s
      return,  except every tensor will have an additional dimension 0 that
      represents the batch dimension.
    - bucket_keys: a tf.int32 vector.

  Raises:
    RuntimeError: If called in pure Eager/tf.function mode without
      `generic_input_v2_key` defined.
  """
  # In TF2 mode, call `GenericInputV2Create` and `GenericInputV2GetNext` for
  # the purpose of migration.
  if py_utils.IsEagerMode():
    if not IsGenericInputV2AllwedInEager() and ('allow_eager' not in kwargs or
                                                not kwargs['allow_eager']):
      raise RuntimeError(
          'GenericInput is called in tf.function or pure Eager mode. This means'
          ' you might be in the process of migrating your code from TF1 to TF2.'
          ' GenericInput is generally not safe for pure Eager mode and a newer '
          'version is introduced (GenericInputV2). To enable that, please '
          'add keyword arg `allow_eager=True` when calling GenericInput. '
          'Also, we recommend that you add extra tests for your own data '
          'pipeline in TF2 mode. Refer to b/223271939 for concrete examples.')

    kwargs.pop('allow_eager', None)
    generic_input_v2_key = kwargs.pop('generic_input_v2_key', None)
    if generic_input_v2_key is None:
      raise RuntimeError(_MISSING_KEY_ERR)

    if generic_input_v2_key in _GENERIC_CACHE_V2:
      resource, out_types, output_tmpl = _GENERIC_CACHE_V2[generic_input_v2_key]
    else:
      with tf.init_scope():
        resource, out_types, output_tmpl = GenericInputV2Create(
            processor, **kwargs)

    _GENERIC_CACHE_V2[generic_input_v2_key] = (resource, out_types, output_tmpl)
    return GenericInputV2GetNext(resource, out_types, output_tmpl)

  proc_fn, out_types, output_tmpl = _ParseProcessor(processor)
  flat_outputs, bucket_keys = ops.gen_x_ops.generic_input(
      processor=proc_fn, out_types=out_types[:-1], **kwargs)
  tf.logging.debug('x_ops.generic_input flat_outputs=%s', flat_outputs)
  # Pack flat_outputs to outputs.
  outputs = output_tmpl.Pack(flat_outputs).out_values
  if isinstance(outputs, list):
    outputs = tuple(outputs)  # b/124336469
  tf.logging.debug('x_ops.generic_input outputs=%s', outputs)
  return outputs, bucket_keys


def GenericInputV2Create(processor, **kwargs):
  # pyformat: disable
  """Builds a generic input pipeline with an explicit resource handle.

  The resource handle uniquely identifies each GenericInputV2 dataset. This
  handle is passsed into method GenericInputV2GetNext to get a batch.

  Example usage:

    def ParseRecord(record):
      # Given a tf.string record, return a (NestedMap, bucketing key) pair.
      feature_map = ...
      features = tf.io.parse_single_example(record, feature_map)
      # Each example is represented by a NestedMap of tensors (without a
      # batch dimension).
      example = py_utils.NestedMap(field1=..., field2=...)
      # bucketing_key is a scalar convertible to tf.int32.
      # Use 1 if all examples are of the same size.
      bucketing_key = 1
      return example, bucketing_key

    resource, out_types, output_tmpl = GenericInputV2Create(ParseRecord, ...)
    input_batch, ... = GenericInputV2GetNext(resource, out_types, output_tmpl)
    # input_batch is a NestedMap of tensors, where dim 0 of each tensor
    # represents the batch dimension.
    input_batch.field1 = ...

  ParseRecord can also take both 'source_id' and 'record' as inputs (the arg
  names must be exactly 'source_id' and 'record'):

    def ParseRecord(source_id, record):
      # Given a tf.int32 source_id and a tf.string record, return a (NestedMap,
      # bucketing key) pair.
      example = py_utils.NestedMap(source_id=source_id, ...)
      ...
      return example, bucketing_key

    resource, out_types, output_tmpl = GenericInputV2Create(
        ParseRecord, file_pattern=..., ...)
    input_batch, bucket_keys = GenericInputV2GetNext(
        resource, out_types, output_tmpl)

  Args:
    processor: a function that takes either a tf.string record or a
      (source_id: tf.int32, record: tf.string) pair as input and returns a
      tuple (output, bucketing_key). `output` must be a NestedMap or a list of
      tensors representing an example. `bucketing_key` must be a scalar
      convertible to a tf.int32 tensor that represents the bucketing key
      (e.g., sequence length for sequence inputs). If `bucketing_key` is a
      negative number, the record is dropped.
    **kwargs: additional keyword args for x_ops.generic_input_v2_create.

  Returns:
    A tuple of (resource, out_types, output_tmpl):

    - resource: a handle that uniquely identifies the created GenericInputV2
        resource.
    - out_types: a list of tensor types representing the types in each batch.
    - output_tmpl: a NestedMap that will be used to pack each batch.
  """
  # pyformat: enable
  proc_fn, out_types, output_tmpl = _ParseProcessor(processor)
  # "Lifts" the resource creation outside of tf.function Graphs (i.e.
  # FuncGraphs). This is necessary when tf.function is retraced, but the
  # same GenericInputV2 resource needs to be used.
  with tf.init_scope():
    return ops.gen_x_ops.generic_input_v2_create(
        processor=proc_fn, out_types=out_types[:-1],
        **kwargs), out_types[:-1], output_tmpl


def GenericInputV2GetNext(resource, out_types, output_tmpl):
  """Gets a batch from the GenericInputV2 dataset represented by `resource`.

  `resource` is a handle that uniquely identifies a GenericInputV2 dataset. More
  details can be seen in the docstring of GenericInputV2Create.

  Example usage:

    # resource, out_types, output_tmpl are returned from `GenericInputV2Create`.
    ... = generic_input.GenericInputV2GetNext(resource, out_types, output_tmpl)

  Args:
    resource: a handle that uniquely identifies the created GenericInputV2
      resource.
    out_types: a list of tensor types representing the types in each batch.
    output_tmpl: a NestedMap that will be used to pack each batch.

  Returns:
    A tuple of (outputs, bucket_keys):

    - outputs: a NestedMap or a tuple of tensors, similar to `processor`'s
        return,  except every tensor will have an additional dimension 0 that
        represents the batch dimension.
    - bucket_keys: a tf.int32 vector.
  """
  flat_outputs, bucket_keys = ops.gen_x_ops.generic_input_v2_get_next(
      resource, out_types)
  tf.logging.debug('x_ops.generic_input_v2_get_next flat_outputs=%s',
                   flat_outputs)
  # Pack flat_outputs to outputs.
  outputs = output_tmpl.Pack(flat_outputs).out_values
  if isinstance(outputs, list):
    outputs = tuple(outputs)  # b/124336469
  tf.logging.debug('x_ops.generic_input_v2_get_next outputs=%s', outputs)
  return outputs, bucket_keys


def ReplicatedGenericInput(processor, num_replicas, replica_device_fn,
                           **kwargs):
  """Builds a replicated input pipeline.

  This is similar to GenericInput, except that the input processing can be
  distributed across devices and then concatenated at the current device.

  Args:
    processor: see comments for GenericInput.
    num_replicas: the number of input processing replicas. Usually set to number
      of infeed hosts.
    replica_device_fn: a int -> string function that takes the replica index in
      range [0, num_replicas) and returns a TF device string, e.g.,
      lambda i: '/task:{}/device:CPU:0'.format(i)
    **kwargs: additional keyword args for x_ops.generic_input.

  Returns:
    A tuple of (outputs, bucket_keys):

    - outputs: a NestedMap or a list of tensors, similar to `processor`'s
      return,  except every tensor will have an additional dimension 0 that
      represents the batch dimension. The batch size will be
      (num_replicas * bucket_batch_limit[...]), i.e.,
      kwargs['bucket_batch_limit'] specifies the per-replica batch size.
    - bucket_keys: a tf.int32 vector.

  Raises:
    RuntimeError: If called in pure Eager/tf.function mode without
      `generic_input_v2_key` defined.
  """
  if num_replicas > 1 and 'bucket_batch_limit' in kwargs:
    assert all(b == max(kwargs['bucket_batch_limit'])
               for b in kwargs['bucket_batch_limit'])
  replica_outputs = []
  if py_utils.IsEagerMode():
    current_key = kwargs.pop('generic_input_v2_key', None)
    if current_key is None:
      raise RuntimeError(_MISSING_KEY_ERR)

  for replica_i in range(num_replicas):
    # Blend `replica_i` into the key for _GENERIC_CACHE_V2 to distinguish
    # different GenericInputV2 ops in the same Datasource object.
    if py_utils.IsEagerMode():
      kwargs['generic_input_v2_key'] = (current_key, replica_i)
    replica_device = replica_device_fn(replica_i)
    with tf.device(replica_device):
      replica_outputs.append(GenericInput(processor, **kwargs))

  output_nmaps, output_bucket_keys = zip(*replica_outputs)
  concat_nmap = tf.nest.map_structure(lambda *t: tf.concat(t, axis=0),
                                      *output_nmaps)
  concat_bucket_keys = tf.concat(output_bucket_keys, axis=0)
  return concat_nmap, concat_bucket_keys
