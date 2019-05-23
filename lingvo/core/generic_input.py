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
from lingvo.core.ops.py_x_ops import gen_x_ops


def GenericInput(processor, *args, **kwargs):
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


GenericInput.__doc__ = gen_x_ops.generic_input.__doc__
