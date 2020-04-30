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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lingvo.compat as tf
from lingvo.core import ops
from lingvo.core import py_utils
from tensorflow.python.util import tf_inspect  # pylint: disable=g-direct-tensorflow-import


def GenericInput(processor, **kwargs):
  """Builds a generic input pipeline.

  Example usage::

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

    - outputs: a NestedMap or a list of tensors, similar to `processor`'s
      return,  except every tensor will have an additional dimension 0 that
      represents the batch dimension.
    - bucket_keys: a tf.int32 vector.
  """
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

  proc_fn = _FlatOutputProcessor.get_concrete_function(
      tf.TensorSpec([], tf.int32), tf.TensorSpec([], tf.string))

  out_types = [
      tf.DType(a.type) for a in proc_fn.function_def.signature.output_arg
  ]
  assert out_types[-1] == tf.int32, ('%s is not expected.' % out_types[-1])
  flat_outputs, bucket_keys = ops.gen_x_ops.generic_input(
      processor=proc_fn, out_types=out_types[:-1], **kwargs)
  tf.logging.debug('x_ops.generic_input flat_outputs=%s', flat_outputs)
  # Pack flat_outputs to outputs.
  outputs = output_tmpl.Pack(flat_outputs).out_values
  tf.logging.debug('x_ops.generic_input outputs=%s', outputs)
  return outputs, bucket_keys
