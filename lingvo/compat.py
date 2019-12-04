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
"""The compatible tensorflow library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.compat.v2 import *  # pylint:disable=wildcard-import, g-bad-import-order

# Import of v1 symbols will be removed when all symbols are migrated to v2
# and tf.compat.v1. So after the migration only v2 symbols and some tf.compat.v1
# symbols are used in the codebase.
from tensorflow.compat.v1 import *  # pylint:disable=wildcard-import

# Import absl.flags and absl.logging to overwrite the Tensorflow ones.
# This is the intended behavior in TF 2.0.
# pylint:disable=g-bad-import-order, unused-import, g-import-not-at-top
from absl import flags
from absl import logging
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python import tf2
from tensorflow.python.compat import v2_compat

from tensorflow.python.framework import function as _function_lib
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import inplace_ops

# The following imports are needed to expose private _Send/_Recv ops
# on TensorFlow 1.X. The could be removed once support for 1.X is dropped.
from google.protobuf import text_format as _text_format
from tensorflow.core.framework import op_def_pb2 as _op_def_pb2
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.framework import op_def_registry as _op_def_registry
# pylint: enable=g-direct-tensorflow-import

_force_disable_v2 = True
if _force_disable_v2:
  v2_compat.disable_v2_behavior()
elif tf2.enabled():
  logging.warning("Lingvo does not support all TF2 behaviors yet. "
                  "Please disable V2 behavior with tf.disable_v2_behavior(), "
                  "or proceed at your own risk.")

# Aliases to a few routines lingvo libraries uses often.
Defun = _function_lib.Defun
While = functional_ops.While
If = functional_ops.If
InplaceUpdate = inplace_ops.alias_inplace_update
Empty = inplace_ops.empty
EmptyLike = inplace_ops.empty_like
GetExtraInputs = _function_lib.get_extra_inputs
GetExtraArgs = _function_lib.get_extra_args
# V1 symbols used in the codebase, and can be migrated to the v2 version later.
# pylint: disable=undefined-variable
add_to_collection = tf.compat.v1.add_to_collection
all_variables = tf.compat.v1.global_variables
assign = tf.compat.v1.assign
assign_add = tf.compat.v1.assign_add
assign_sub = tf.compat.v1.assign_sub
AUTO_REUSE = tf.compat.v1.AUTO_REUSE
colocate_with = tf.compat.v1.colocate_with
ConfigProto = tf.compat.v1.ConfigProto
constant_initializer = tf.compat.v1.constant_initializer
container = tf.compat.v1.container
count_nonzero = tf.compat.v1.count_nonzero
device = tf.compat.v1.device
Dimension = tf.compat.v1.Dimension
div = tf.compat.v1.div
expand_dims = tf.compat.v1.expand_dims
get_collection = tf.compat.v1.get_collection
get_default_graph = tf.compat.v1.get_default_graph
get_variable = tf.compat.v1.get_variable
get_variable_scope = tf.compat.v1.get_variable_scope
gfile = tf.compat.v1.gfile
global_variables = tf.compat.v1.global_variables
global_variables_initializer = tf.compat.v1.global_variables_initializer
gradients = tf.compat.v1.gradients
graph_util.extract_sub_graph = tf.compat.v1.graph_util.extract_sub_graph
GraphKeys = tf.compat.v1.GraphKeys
local_variables_initializer = tf.compat.v1.local_variables_initializer
mod = tf.compat.v1.mod
moving_average_variables = tf.compat.v1.moving_average_variables
name_scope = tf.compat.v1.name_scope
nn.conv2d = tf.compat.v1.nn.conv2d
nn.convolution = tf.compat.v1.nn.convolution
nn.depthwise_conv2d = tf.compat.v1.nn.depthwise_conv2d
nn.dropout = tf.compat.v1.nn.dropout
nn.embedding_lookup = tf.compat.v1.nn.embedding_lookup
nn.pool = tf.compat.v1.nn.pool
nn.sampled_softmax_loss = tf.compat.v1.nn.sampled_softmax_loss
nn.sufficient_statistics = tf.compat.v1.nn.sufficient_statistics
placeholder = tf.compat.v1.placeholder
placeholder_with_default = tf.compat.v1.placeholder_with_default
py_func = tf.compat.v1.py_func
random_normal_initializer = tf.compat.v1.random_normal_initializer
random_poisson = tf.compat.v1.random_poisson
random.stateless_multinomial = tf.compat.v1.random.stateless_multinomial
random_uniform_initializer = tf.compat.v1.random_uniform_initializer
reduce_min = tf.compat.v1.reduce_min
report_uninitialized_variables = tf.compat.v1.report_uninitialized_variables
reset_default_graph = tf.compat.v1.reset_default_graph
resource_loader = tf.compat.v1.resource_loader
Session = tf.compat.v1.Session
set_random_seed = tf.compat.v1.set_random_seed
strings.split = tf.compat.v1.strings.split
Summary = tf.compat.v1.Summary
if tf.compat.v1.summary is not None:
  # tf.summary are not supported on TPU so we sometimes set tf.summary to None
  # to prohibit the direct use of it.
  # It is safe to skip copying tf.summary members in such cases.
  summary.FileWriter = tf.compat.v1.summary.FileWriter
  summary.merge_all = tf.compat.v1.summary.merge_all
  summary.Summary = tf.compat.v1.summary.Summary
tables_initializer = tf.compat.v1.tables_initializer
test.get_temp_dir = tf.compat.v1.test.get_temp_dir
test.mock = tf.compat.v1.test.mock
to_int32 = tf.compat.v1.to_int32
train.AdamOptimizer = tf.compat.v1.train.AdamOptimizer
train.get_or_create_global_step = tf.compat.v1.train.get_or_create_global_step
train.get_global_step = tf.compat.v1.train.get_global_step
train.GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer
train.Optimizer = tf.compat.v1.train.Optimizer
train.Saver = tf.compat.v1.train.Saver
trainable_variables = tf.compat.v1.trainable_variables
truncated_normal_initializer = tf.compat.v1.truncated_normal_initializer
uniform_unit_scaling_initializer = tf.compat.v1.uniform_unit_scaling_initializer
Variable = tf.compat.v1.Variable
variables_initializer = tf.compat.v1.variables_initializer
VariableScope = tf.compat.v1.VariableScope
variable_scope = tf.compat.v1.variable_scope
where = tf.compat.v1.where
while_loop = tf.compat.v1.while_loop

# tf.compat.v2 symbols. Will remove the 'tf.compat.v2' prefix when the migration
# is done.
assert_greater_equal = tf.compat.v2.debugging.assert_greater_equal
check_numerics = tf.compat.v2.debugging.check_numerics
cumprod = tf.compat.v2.math.cumprod
erf = tf.compat.v2.math.erf
FixedLenFeature = tf.compat.v2.io.FixedLenFeature
floordiv = tf.compat.v2.math.floordiv
is_finite = tf.compat.v2.math.is_finite
is_inf = tf.compat.v2.math.is_inf
is_nan = tf.compat.v2.math.is_nan
log = tf.compat.v2.math.log
matrix_band_part = tf.compat.v2.linalg.band_part
parse_single_example = tf.compat.v2.io.parse_single_example
random_uniform = tf.compat.v2.random.uniform
random_normal = tf.compat.v2.random.normal
reciprocal = tf.compat.v2.math.reciprocal
rint = tf.compat.v2.math.rint
rsqrt = tf.compat.v2.math.rsqrt
train.Server = tf.compat.v2.distribute.Server
unsorted_segment_sum = tf.compat.v2.math.unsorted_segment_sum
VarLenFeature = tf.compat.v2.io.VarLenFeature

# pylint: enable=undefined-variable


# TODO(slebedev): Remove after there is no need to support 1.X.
def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """\
op {
  name: "_Recv"
  output_arg {
    name: "tensor"
    type_attr: "tensor_type"
  }
  attr {
    name: "tensor_type"
    type: "type"
  }
  attr {
    name: "tensor_name"
    type: "string"
  }
  attr {
    name: "send_device"
    type: "string"
  }
  attr {
    name: "send_device_incarnation"
    type: "int"
  }
  attr {
    name: "recv_device"
    type: "string"
  }
  attr {
    name: "client_terminated"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
op {
  name: "_Send"
  input_arg {
    name: "tensor"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
  attr {
    name: "tensor_name"
    type: "string"
  }
  attr {
    name: "send_device"
    type: "string"
  }
  attr {
    name: "send_device_incarnation"
    type: "int"
  }
  attr {
    name: "recv_device"
    type: "string"
  }
  attr {
    name: "client_terminated"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
"""


def _Recv(tensor_type,
          tensor_name,
          send_device,
          send_device_incarnation,
          recv_device,
          name=None):
  return _op_def_lib.apply_op(
      "_Recv",
      tensor_type=tensor_type,
      tensor_name=tensor_name,
      send_device=send_device,
      send_device_incarnation=send_device_incarnation,
      recv_device=recv_device,
      client_terminated=False,
      name=name if name else "Recv")


def _Send(tensor,
          tensor_name,
          send_device,
          send_device_incarnation,
          recv_device,
          name=None):
  return _op_def_lib.apply_op(
      "_Send",
      tensor=tensor,
      tensor_name=tensor_name,
      send_device=send_device,
      send_device_incarnation=send_device_incarnation,
      recv_device=recv_device,
      client_terminated=False,
      name=name if name else "Send")


# pylint: disable=undefined-variable
if not hasattr(raw_ops, "Send") and not hasattr(raw_ops, "Recv"):
  _op_def_lib = _InitOpDefLibrary()
  raw_ops.Send = _Send
  raw_ops.Recv = _Recv
# pylint: enable=undefined-variable

del _Send, _Recv, _InitOpDefLibrary
