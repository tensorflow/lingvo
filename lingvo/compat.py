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
from tensorflow.python.platform import app

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
arg_max = tf.compat.v1.arg_max
assert_integer = tf.compat.v1.assert_integer
assert_positive = tf.compat.v1.assert_positive
assert_type = tf.compat.v1.assert_type
assert_scalar = tf.compat.v1.assert_scalar
assign = tf.compat.v1.assign
assign_add = tf.compat.v1.assign_add
assign_sub = tf.compat.v1.assign_sub
AUTO_REUSE = tf.compat.v1.AUTO_REUSE
batch_gather = tf.compat.v1.batch_gather
colocate_with = tf.compat.v1.colocate_with
cond = tf.compat.v1.cond
ConfigProto = tf.compat.v1.ConfigProto
constant = tf.compat.v1.constant
constant_initializer = tf.compat.v1.constant_initializer
container = tf.compat.v1.container
convert_to_tensor = tf.compat.v1.convert_to_tensor
count_nonzero = tf.compat.v1.count_nonzero
data.make_initializable_iterator = tf.compat.v1.data.make_initializable_iterator
data.make_one_shot_iterator = tf.compat.v1.data.make_one_shot_iterator
data.TFRecordDataset = tf.compat.v1.data.TFRecordDataset
decode_raw = tf.compat.v1.decode_raw
device = tf.compat.v1.device
Dimension = tf.compat.v1.Dimension
div = tf.compat.v1.div
expand_dims = tf.compat.v1.expand_dims
floor_div = tf.compat.v1.floor_div
get_collection = tf.compat.v1.get_collection
get_collection_ref = tf.compat.v1.get_collection_ref
get_default_graph = tf.compat.v1.get_default_graph
get_local_variable = tf.compat.v1.get_local_variable
get_seed = tf.compat.v1.get_seed
get_variable = tf.compat.v1.get_variable
get_variable_scope = tf.compat.v1.get_variable_scope
gfile = tf.compat.v1.gfile
global_variables = tf.compat.v1.global_variables
global_variables_initializer = tf.compat.v1.global_variables_initializer
gradients = tf.compat.v1.gradients
graph_util.convert_variables_to_constants = (
    tf.compat.v1.graph_util.convert_variables_to_constants)
graph_util.extract_sub_graph = tf.compat.v1.graph_util.extract_sub_graph
GraphKeys = tf.compat.v1.GraphKeys
GraphOptions = tf.compat.v1.GraphOptions
image.resize_bilinear = tf.compat.v1.image.resize_bilinear
image.resize_images = tf.compat.v1.image.resize_images
image.resize_nearest_neighbor = tf.compat.v1.image.resize_nearest_neighbor
initialize_all_tables = tf.compat.v1.initialize_all_tables
initialize_all_variables = global_variables_initializer
initializers.global_variables = tf.compat.v1.initializers.global_variables
initializers.variables = tf.compat.v1.initializers.variables
io.tf_record_iterator = tf.compat.v1.io.tf_record_iterator
layers = tf.compat.v1.layers
local_variables_initializer = tf.compat.v1.local_variables_initializer
losses.absolute_difference = tf.compat.v1.losses.absolute_difference
losses.add_loss = tf.compat.v1.losses.add_loss
losses.compute_weighted_loss = tf.compat.v1.losses.compute_weighted_loss
losses.get_regularization_loss = tf.compat.v1.losses.get_regularization_loss
losses.huber_loss = tf.compat.v1.losses.huber_loss
losses.mean_squared_error = tf.compat.v1.losses.mean_squared_error
losses.Reduction.MEAN = tf.compat.v1.losses.Reduction.MEAN
losses.Reduction.SUM = tf.compat.v1.losses.Reduction.SUM
losses.sigmoid_cross_entropy = tf.compat.v1.losses.sigmoid_cross_entropy
losses.softmax_cross_entropy = tf.compat.v1.losses.softmax_cross_entropy
losses.sparse_softmax_cross_entropy = (
    tf.compat.v1.losses.sparse_softmax_cross_entropy)
make_template = tf.compat.v1.make_template
metrics.accuracy = tf.compat.v1.metrics.accuracy
metrics.auc = tf.compat.v1.metrics.auc
metrics.precision = tf.compat.v1.metrics.precision
metrics.recall = tf.compat.v1.metrics.recall
mod = tf.compat.v1.mod
moving_average_variables = tf.compat.v1.moving_average_variables
multinomial = tf.compat.v1.multinomial
name_scope = tf.compat.v1.name_scope
nn.conv2d = tf.compat.v1.nn.conv2d
nn.convolution = tf.compat.v1.nn.convolution
nn.ctc_beam_search_decoder = tf.compat.v1.nn.ctc_beam_search_decoder
nn.depthwise_conv2d = tf.compat.v1.nn.depthwise_conv2d
nn.dropout = tf.compat.v1.nn.dropout
nn.embedding_lookup = tf.compat.v1.nn.embedding_lookup
nn.fused_batch_norm = tf.compat.v1.nn.fused_batch_norm
nn.in_top_k = tf.compat.v1.nn.in_top_k
nn.pool = tf.compat.v1.nn.pool
nn.rnn_cell = tf.compat.v1.nn.rnn_cell
nn.sampled_softmax_loss = tf.compat.v1.nn.sampled_softmax_loss
nn.separable_conv2d = tf.compat.v1.nn.separable_conv2d
nn.softmax = tf.compat.v1.nn.softmax
nn.softmax_cross_entropy_with_logits_v2 = (
    tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2)
nn.sufficient_statistics = tf.compat.v1.nn.sufficient_statistics
nn.xw_plus_b = tf.compat.v1.nn.xw_plus_b
OptimizerOptions = tf.compat.v1.OptimizerOptions
pad = tf.compat.v1.pad
placeholder = tf.compat.v1.placeholder
placeholder_with_default = tf.compat.v1.placeholder_with_default
Print = tf.compat.v1.Print
py_func = tf.compat.v1.py_func
python_io = tf.compat.v1.python_io
random_normal_initializer = tf.compat.v1.random_normal_initializer
random_poisson = tf.compat.v1.random_poisson
random.stateless_multinomial = tf.compat.v1.random.stateless_multinomial
random_uniform_initializer = tf.compat.v1.random_uniform_initializer
reduce_join = tf.compat.v1.reduce_join
reduce_max = tf.compat.v1.reduce_max
reduce_mean = tf.compat.v1.reduce_mean
reduce_min = tf.compat.v1.reduce_min
reduce_sum = tf.compat.v1.reduce_sum
report_uninitialized_variables = tf.compat.v1.report_uninitialized_variables
reset_default_graph = tf.compat.v1.reset_default_graph
resource_loader = tf.compat.v1.resource_loader
reverse_sequence = tf.compat.v1.reverse_sequence
RunMetadata = tf.compat.v1.RunMetadata
RunOptions = tf.compat.v1.RunOptions
saved_model.build_signature_def = tf.compat.v1.saved_model.build_signature_def
saved_model.loader = tf.compat.v1.saved_model.loader
saved_model.signature_constants = tf.compat.v1.saved_model.signature_constants
saved_model.tag_constants = tf.compat.v1.saved_model.tag_constants
saved_model.utils = tf.compat.v1.saved_model.utils
Session = tf.compat.v1.Session
set_random_seed = tf.compat.v1.set_random_seed
sparse_tensor_dense_matmul = tf.compat.v1.sparse_tensor_dense_matmul
sparse_to_dense = tf.compat.v1.sparse_to_dense
sparse_tensor_to_dense = tf.compat.v1.sparse_tensor_to_dense
string_split = tf.compat.v1.string_split
strings.reduce_join = reduce_join
strings.split = tf.compat.v1.strings.split
Summary = tf.compat.v1.Summary
if tf.compat.v1.summary is not None:
  # tf.summary are not supported on TPU so we sometimes set tf.summary to None
  # to prohibit the direct use of it.
  # It is safe to skip copying tf.summary members in such cases.
  summary.FileWriter = tf.compat.v1.summary.FileWriter
  summary.image = tf.compat.v1.summary.image
  summary.merge = tf.compat.v1.summary.merge
  summary.merge_all = tf.compat.v1.summary.merge_all
  summary.scalar = tf.compat.v1.summary.scalar
  summary.Summary = tf.compat.v1.summary.Summary
  summary.Summary.FromString = tf.compat.v1.summary.Summary.FromString
tables_initializer = tf.compat.v1.tables_initializer
test.compute_gradient_error = tf.compat.v1.test.compute_gradient_error
test.get_temp_dir = tf.compat.v1.test.get_temp_dir
test.mock = tf.compat.v1.test.mock
to_complex64 = tf.compat.v1.to_complex64
to_double = tf.compat.v1.to_double
to_float = tf.compat.v1.to_float
to_int32 = tf.compat.v1.to_int32
to_int64 = tf.compat.v1.to_int64
tpu = tf.compat.v1.tpu
train.AdagradOptimizer = tf.compat.v1.train.AdagradOptimizer
train.AdamOptimizer = tf.compat.v1.train.AdamOptimizer
train.export_meta_graph = tf.compat.v1.train.export_meta_graph
train.get_or_create_global_step = tf.compat.v1.train.get_or_create_global_step
train.get_global_step = tf.compat.v1.train.get_global_step
train.GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer
train.MomentumOptimizer = tf.compat.v1.train.MomentumOptimizer
train.MonitoredTrainingSession = tf.compat.v1.train.MonitoredTrainingSession
train.NewCheckpointReader = tf.compat.v1.train.NewCheckpointReader
train.Optimizer = tf.compat.v1.train.Optimizer
train.RMSPropOptimizer = tf.compat.v1.train.RMSPropOptimizer
train.Saver = tf.compat.v1.train.Saver
train.SaverDef = tf.compat.v1.train.SaverDef
train.summary_iterator = tf.compat.v1.train.summary_iterator
trainable_variables = tf.compat.v1.trainable_variables
truncated_normal_initializer = tf.compat.v1.truncated_normal_initializer
uniform_unit_scaling_initializer = tf.compat.v1.uniform_unit_scaling_initializer
unsorted_segment_max = tf.compat.v1.unsorted_segment_max
unsorted_segment_mean = tf.compat.v1.unsorted_segment_mean
Variable = tf.compat.v1.Variable
variables_initializer = tf.compat.v1.variables_initializer
VariableScope = tf.compat.v1.VariableScope
variable_scope = tf.compat.v1.variable_scope
where = tf.compat.v1.where
while_loop = tf.compat.v1.while_loop

# tf.compat.v2 symbols. Will remove the 'tf.compat.v2' prefix when the migration
# is done.
assert_greater_equal = tf.compat.v2.debugging.assert_greater_equal
assert_less_equal = tf.compat.v2.debugging.assert_less_equal
ceil = tf.compat.v2.math.ceil
extract_image_patches = tf.compat.v2.image.extract_patches
check_numerics = tf.compat.v2.debugging.check_numerics
cross = tf.compat.v2.linalg.cross
cumprod = tf.compat.v2.math.cumprod
dequantize = tf.compat.v2.quantization.dequantize
diag = tf.compat.v2.linalg.tensor_diag
erf = tf.compat.v2.math.erf
fake_quant_with_min_max_args = (
    tf.compat.v2.quantization.fake_quant_with_min_max_args)
fake_quant_with_min_max_vars = (
    tf.compat.v2.quantization.fake_quant_with_min_max_vars)
FixedLenFeature = tf.compat.v2.io.FixedLenFeature
FixedLenSequenceFeature = tf.compat.v2.io.FixedLenSequenceFeature
floordiv = tf.compat.v2.math.floordiv
floormod = tf.compat.v2.math.floormod
imag = tf.compat.v2.math.imag
image.resize_image_with_crop_or_pad = tf.compat.v2.image.resize_with_crop_or_pad
is_finite = tf.compat.v2.math.is_finite
is_inf = tf.compat.v2.math.is_inf
is_nan = tf.compat.v2.math.is_nan
is_non_decreasing = tf.compat.v2.math.is_non_decreasing
lin_space = tf.compat.v2.linspace
log = tf.compat.v2.math.log
log1p = tf.compat.v2.math.log1p
log_sigmoid = tf.compat.v2.math.log_sigmoid
logical_xor = tf.compat.v2.math.logical_xor
ceil = tf.compat.v2.math.ceil
matrix_band_part = tf.compat.v2.linalg.band_part
matrix_inverse = tf.compat.v2.linalg.inv
OpError = tf.compat.v2.errors.OpError
parse_example = tf.compat.v2.io.parse_example
parse_single_example = tf.compat.v2.io.parse_single_example
parse_single_sequence_example = tf.compat.v2.io.parse_single_sequence_example
parse_tensor = tf.compat.v2.io.parse_tensor
random_gamma = tf.compat.v2.random.gamma
random_normal = tf.compat.v2.random.normal
random_shuffle = tf.compat.v2.random.shuffle
random_uniform = tf.compat.v2.random.uniform
real = tf.compat.v2.math.real
reciprocal = tf.compat.v2.math.reciprocal
regex_replace = tf.compat.v2.strings.regex_replace
rint = tf.compat.v2.math.rint
rsqrt = tf.compat.v2.math.rsqrt
serialize_tensor = tf.compat.v2.io.serialize_tensor
sparse_reorder = tf.compat.v2.sparse.reorder
squared_difference = tf.compat.v2.math.squared_difference
string_join = tf.compat.v2.strings.join
string_to_number = tf.compat.v2.strings.to_number
train.Server = tf.compat.v2.distribute.Server
train.write_graph = tf.compat.v2.io.write_graph
truncated_normal = tf.compat.v2.random.truncated_normal
unsorted_segment_min = tf.compat.v2.math.unsorted_segment_min
unsorted_segment_sum = tf.compat.v2.math.unsorted_segment_sum
VarLenFeature = tf.compat.v2.io.VarLenFeature
where_v2 = tf.compat.v2.where
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
