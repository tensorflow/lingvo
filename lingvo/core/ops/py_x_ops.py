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
"""Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.framework import function

gen_x_ops = tf.load_op_library(
    tf.resource_loader.get_path_to_datafile('x_ops.so'))

if 'assert_shape_match' not in dir(gen_x_ops):
  # Static linking:
  # pylint: disable=g-import-not-at-top
  from lingvo.core.ops import gen_x_ops
  # pylint: enable=g-import-not-at-top

assert_shape_match = gen_x_ops.assert_shape_match
assert_same_dim0 = gen_x_ops.assert_same_dim0
random_permutation_sequence = gen_x_ops.random_permutation_sequence

best_step = gen_x_ops.best_step

beam_search_step = gen_x_ops.beam_search_step
top_k_terminated_hyps = gen_x_ops.top_k_terminated_hyps
unpack_hyp = gen_x_ops.unpack_hyp
hyps_from_beam_search_outs = gen_x_ops.hyps_from_beam_search_outs

cached_call = gen_x_ops.cached_call


def generic_input(processor, *args, **kwargs):
  """Simple wrapper for gen_x_ops.generic_input."""
  # pylint: disable=protected-access
  if not isinstance(processor, function._DefinedFunction):
    # Helper if processor is a python callable.
    processor = function.Defun(tf.string)(processor)
  out_types = [
      tf.DType(a.type) for a in processor.definition.signature.output_arg
  ]
  assert out_types[-1] == tf.int32, ('%s is not expected.' % out_types[-1])
  return gen_x_ops.generic_input(
      processor=processor, out_types=out_types[:-1], *args, **kwargs)
