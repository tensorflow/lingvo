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

import tensorflow as tf

from tensorflow.python.framework import function
from lingvo.core import py_utils
from lingvo.core.ops import py_x_ops


def GenericInput(processor, *args, **kwargs):
  """Builds a generic input pipeline.

  Example usage:

    def ParseRecord(record):
      # Given a tf.string record, return a (NestedMap, bucketing key) pair.
      feature_map = ...
      features = tf.parse_single_example(record, feature_map)
      # Each example is represented by a NestedMap of tensors (without a
      # batch dimension).
      example = py_utils.NestedMap(field1=..., field2=...)
      # bucketing_key is an int scalar tensor.
      # Use 1 if all examples are of the same size.
      bucketing_key = tf.to_int32(1)
      return example, bucketing_key

    input_batch = GenericInput(ParseRecord, file_pattern=..., ...)
    # input_batch is a NestedMap of tensors, where dim 0 of each tensor
    # represents the batch dimension.
    input_batch.field1 = ...

  Args:
    processor: a function that takes a string record as input and returns a list
      of tensors or NestedMaps representing one example. The last return value
      of processor must be an int32 scalar tensor that represents the bucketing
      key (e.g., sequence length for sequence inputs).
    *args: additional args for x_ops.generic_input.
    **kwargs: additional keyword args for x_ops.generic_input.

  Returns:
    A list of tensors or NestedMaps, similar to a list returned by 'processor',
    except:
      * The bucket key is not included in the output.
      * Every tensor will have an additional dimension 0 that represents the
        batch dimension.
  """
  output_tmpl = py_utils.NestedMap()

  def _FlatOutputProcessor(inputs):
    """Returns a flattened list of 'processor(inputs)'."""
    outputs = processor(inputs)
    tf.logging.debug('Processor outputs=%s', outputs)
    assert len(outputs) > 1, outputs
    # Add 'outputs' as a list so that each element will be flattened.
    output_tmpl.values = list(outputs)
    flat_outputs = output_tmpl.Flatten()
    tf.logging.debug('Processor flat outputs=%s', flat_outputs)
    return flat_outputs

  proc_fn = function.Defun(tf.string)(_FlatOutputProcessor)

  out_types = [
      tf.DType(a.type) for a in proc_fn.definition.signature.output_arg
  ]
  assert out_types[-1] == tf.int32, ('%s is not expected.' % out_types[-1])
  flat_outputs = py_x_ops.gen_x_ops.generic_input(
      processor=proc_fn, out_types=out_types[:-1], *args, **kwargs)
  tf.logging.debug('x_ops.generic_input flat_outputs=%s', flat_outputs)
  if not output_tmpl:
    return flat_outputs
  # Pack flat_outputs to outputs.
  output_tmpl.values.pop(-1)
  outputs = output_tmpl.Pack(flat_outputs).values
  tf.logging.debug('x_ops.generic_input outputs=%s', outputs)
  return outputs
