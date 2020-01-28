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

import importlib

import tensorflow.compat.v1 as tf1
from tensorflow.compat.v2 import *  # pylint:disable=wildcard-import, g-bad-import-order

# Import absl.flags and absl.logging to overwrite the Tensorflow ones.
# This is the intended behavior in TF 2.0.
# pylint:disable=g-bad-import-order, unused-import, g-import-not-at-top
from absl import flags
from absl import logging
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python import tf2

from tensorflow.python.framework import function as _function_lib
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import inplace_ops

from tensorflow.python.platform import app
# pylint: enable=g-direct-tensorflow-import
# pylint: enable=unused-import, g-bad-import-order, g-import-not-at-top

_force_disable_v2 = True
if _force_disable_v2:
  tf1.disable_v2_behavior()
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

# Move this V2 symbol here to avoid being overwritten by its following V1
# version.
where_v2 = where  # pylint: disable=undefined-variable, used-before-assignment

# Import the local V2 module to maker sure the following V1 overwritting never
# applies to the global module and symbol.
data = importlib.import_module("tensorflow.compat.v2.data")
graph_util = importlib.import_module("tensorflow.compat.v2.graph_util")
image = importlib.import_module("tensorflow.compat.v2.image")
initializers = importlib.import_module(
    "tensorflow.compat.v2.keras.initializers")
io = importlib.import_module("tensorflow.compat.v2.io")
losses = importlib.import_module("tensorflow.compat.v2.keras.losses")
metrics = importlib.import_module("tensorflow.compat.v2.keras.metrics")
nn = importlib.import_module("tensorflow.compat.v2.nn")
random = importlib.import_module("tensorflow.compat.v2.random")
saved_model = importlib.import_module("tensorflow.compat.v2.saved_model")
strings = importlib.import_module("tensorflow.compat.v2.strings")
summary = importlib.import_module("tensorflow.summary")
test = importlib.import_module("tensorflow.compat.v2.test")
train = importlib.import_module("tensorflow.compat.v2.train")

# V1 symbols used in the codebase, and can be migrated to the v2 version later.
# pylint: disable=undefined-variable
add_to_collection = tf1.add_to_collection
all_variables = tf1.global_variables
arg_max = tf1.arg_max
arg_min = tf1.arg_min
assert_integer = tf1.assert_integer
assert_positive = tf1.assert_positive
assert_type = tf1.assert_type
assert_scalar = tf1.assert_scalar
assign = tf1.assign
assign_add = tf1.assign_add
assign_sub = tf1.assign_sub
AUTO_REUSE = tf1.AUTO_REUSE
batch_gather = tf1.batch_gather
colocate_with = tf1.colocate_with
cond = tf1.cond
ConfigProto = tf1.ConfigProto
constant = tf1.constant
constant_initializer = tf1.constant_initializer
container = tf1.container
convert_to_tensor = tf1.convert_to_tensor
count_nonzero = tf1.count_nonzero
data.make_initializable_iterator = tf1.data.make_initializable_iterator
data.make_one_shot_iterator = tf1.data.make_one_shot_iterator
data.Dataset = tf1.data.Dataset
data.TFRecordDataset = tf1.data.TFRecordDataset
decode_raw = tf1.decode_raw
decode_csv = tf1.decode_csv
device = tf1.device
Dimension = tf1.Dimension
div = tf1.div
enable_eager_execution = tf1.enable_eager_execution
expand_dims = tf1.expand_dims
floor_div = tf1.floor_div
get_collection = tf1.get_collection
get_collection_ref = tf1.get_collection_ref
get_default_graph = tf1.get_default_graph
get_local_variable = tf1.get_local_variable
get_seed = tf1.get_seed
get_variable = tf1.get_variable
get_variable_scope = tf1.get_variable_scope
gfile = tf1.gfile
global_variables = tf1.global_variables
global_variables_initializer = tf1.global_variables_initializer
gradients = tf1.gradients
graph_util.convert_variables_to_constants = (
    tf1.graph_util.convert_variables_to_constants)
graph_util.extract_sub_graph = tf1.graph_util.extract_sub_graph
GraphDef = tf1.GraphDef
GraphKeys = tf1.GraphKeys
GraphOptions = tf1.GraphOptions
image.resize_bilinear = tf1.image.resize_bilinear
image.resize_images = tf1.image.resize_images
image.resize_nearest_neighbor = tf1.image.resize_nearest_neighbor
initialize_all_tables = tf1.initialize_all_tables
initialize_all_variables = global_variables_initializer
initializers.constant = tf1.initializers.constant
initializers.global_variables = tf1.initializers.global_variables
initializers.variables = tf1.initializers.variables
io.tf_record_iterator = tf1.io.tf_record_iterator
layers = tf1.layers
local_variables_initializer = tf1.local_variables_initializer
losses.absolute_difference = tf1.losses.absolute_difference
losses.add_loss = tf1.losses.add_loss
losses.compute_weighted_loss = tf1.losses.compute_weighted_loss
losses.get_regularization_loss = tf1.losses.get_regularization_loss
losses.huber_loss = tf1.losses.huber_loss
losses.mean_squared_error = tf1.losses.mean_squared_error
losses.Reduction.MEAN = tf1.losses.Reduction.MEAN
losses.Reduction.SUM = tf1.losses.Reduction.SUM
losses.sigmoid_cross_entropy = tf1.losses.sigmoid_cross_entropy
losses.softmax_cross_entropy = tf1.losses.softmax_cross_entropy
losses.sparse_softmax_cross_entropy = (tf1.losses.sparse_softmax_cross_entropy)
make_template = tf1.make_template
metrics.accuracy = tf1.metrics.accuracy
metrics.auc = tf1.metrics.auc
metrics.precision = tf1.metrics.precision
metrics.recall = tf1.metrics.recall
mod = tf1.mod
moving_average_variables = tf1.moving_average_variables
multinomial = tf1.multinomial
name_scope = tf1.name_scope
nn.conv2d = tf1.nn.conv2d
nn.convolution = tf1.nn.convolution
nn.ctc_beam_search_decoder = tf1.nn.ctc_beam_search_decoder
nn.depthwise_conv2d = tf1.nn.depthwise_conv2d
nn.dropout = tf1.nn.dropout
nn.embedding_lookup = tf1.nn.embedding_lookup
nn.fused_batch_norm = tf1.nn.fused_batch_norm
nn.in_top_k = tf1.nn.in_top_k
nn.pool = tf1.nn.pool
nn.rnn_cell = tf1.nn.rnn_cell
nn.sampled_softmax_loss = tf1.nn.sampled_softmax_loss
nn.separable_conv2d = tf1.nn.separable_conv2d
nn.softmax = tf1.nn.softmax
nn.softmax_cross_entropy_with_logits_v2 = (
    tf1.nn.softmax_cross_entropy_with_logits_v2)
nn.sufficient_statistics = tf1.nn.sufficient_statistics
nn.xw_plus_b = tf1.nn.xw_plus_b
OptimizerOptions = tf1.OptimizerOptions
pad = tf1.pad
placeholder = tf1.placeholder
placeholder_with_default = tf1.placeholder_with_default
Print = tf1.Print
py_func = tf1.py_func
python_io = tf1.python_io
random_normal_initializer = tf1.random_normal_initializer
random_poisson = tf1.random_poisson
random.stateless_multinomial = tf1.random.stateless_multinomial
random_uniform_initializer = tf1.random_uniform_initializer
reduce_join = tf1.reduce_join
reduce_max = tf1.reduce_max
reduce_mean = tf1.reduce_mean
reduce_min = tf1.reduce_min
reduce_sum = tf1.reduce_sum
report_uninitialized_variables = tf1.report_uninitialized_variables
reset_default_graph = tf1.reset_default_graph
resource_loader = tf1.resource_loader
reverse_sequence = tf1.reverse_sequence
RunMetadata = tf1.RunMetadata
RunOptions = tf1.RunOptions
saved_model.build_signature_def = tf1.saved_model.build_signature_def
saved_model.load = tf1.saved_model.load
saved_model.loader = tf1.saved_model.loader
saved_model.signature_constants = tf1.saved_model.signature_constants
saved_model.simple_save = tf1.saved_model.simple_save
saved_model.tag_constants = tf1.saved_model.tag_constants
saved_model.utils = tf1.saved_model.utils
Session = tf1.Session
set_random_seed = tf1.set_random_seed
sparse_tensor_dense_matmul = tf1.sparse_tensor_dense_matmul
sparse_to_dense = tf1.sparse_to_dense
sparse_tensor_to_dense = tf1.sparse_tensor_to_dense
string_split = tf1.string_split
strings.reduce_join = reduce_join
strings.split = tf1.strings.split
string_to_hash_bucket_fast = tf1.string_to_hash_bucket_fast
Summary = tf1.Summary
if tf1.summary is not None:
  # tf.summary are not supported on TPU so we sometimes set tf.summary to None
  # to prohibit the direct use of it.
  # It is safe to skip copying tf.summary members in such cases.
  summary.FileWriter = tf1.summary.FileWriter
  summary.histogram = tf1.summary.histogram
  summary.image = tf1.summary.image
  summary.merge = tf1.summary.merge
  summary.merge_all = tf1.summary.merge_all
  summary.scalar = tf1.summary.scalar
  summary.Summary = tf1.summary.Summary
  summary.Summary.FromString = tf1.summary.Summary.FromString
tables_initializer = tf1.tables_initializer
test.compute_gradient_error = tf1.test.compute_gradient_error
test.get_temp_dir = tf1.test.get_temp_dir
test.mock = tf1.test.mock
to_complex64 = tf1.to_complex64
to_double = tf1.to_double
to_float = tf1.to_float
to_int32 = tf1.to_int32
to_int64 = tf1.to_int64
tpu = tf1.tpu
train.AdadeltaOptimizer = tf1.train.AdadeltaOptimizer
train.AdagradOptimizer = tf1.train.AdagradOptimizer
train.AdamOptimizer = tf1.train.AdamOptimizer
train.export_meta_graph = tf1.train.export_meta_graph
train.get_or_create_global_step = tf1.train.get_or_create_global_step
train.get_global_step = tf1.train.get_global_step
train.GradientDescentOptimizer = tf1.train.GradientDescentOptimizer
train.MomentumOptimizer = tf1.train.MomentumOptimizer
train.MonitoredTrainingSession = tf1.train.MonitoredTrainingSession
train.NewCheckpointReader = tf1.train.NewCheckpointReader
train.Optimizer = tf1.train.Optimizer
train.RMSPropOptimizer = tf1.train.RMSPropOptimizer
train.Saver = tf1.train.Saver
train.SaverDef = tf1.train.SaverDef
train.summary_iterator = tf1.train.summary_iterator
trainable_variables = tf1.trainable_variables
truncated_normal_initializer = tf1.truncated_normal_initializer
uniform_unit_scaling_initializer = tf1.uniform_unit_scaling_initializer
unsorted_segment_max = tf1.unsorted_segment_max
unsorted_segment_mean = tf1.unsorted_segment_mean
Variable = tf1.Variable
variables_initializer = tf1.variables_initializer
VariableScope = tf1.VariableScope
variable_scope = tf1.variable_scope
where = tf1.where
while_loop = tf1.while_loop
wrap_function = tf1.wrap_function

# tf.compat.v2 symbols. Will remove the 'tf.compat.v2' prefix when the migration
# is done.
assert_greater_equal = debugging.assert_greater_equal
assert_less_equal = debugging.assert_less_equal
ceil = math.ceil
extract_image_patches = image.extract_patches
check_numerics = debugging.check_numerics
cross = linalg.cross
cumprod = math.cumprod
dequantize = quantization.dequantize
diag = linalg.tensor_diag
erf = math.erf
fake_quant_with_min_max_args = (quantization.fake_quant_with_min_max_args)
fake_quant_with_min_max_vars = (quantization.fake_quant_with_min_max_vars)
FixedLenFeature = io.FixedLenFeature
FixedLenSequenceFeature = io.FixedLenSequenceFeature
floordiv = math.floordiv
floormod = math.floormod
imag = math.imag
image.resize_image_with_crop_or_pad = image.resize_with_crop_or_pad
is_finite = math.is_finite
is_inf = math.is_inf
is_nan = math.is_nan
is_non_decreasing = math.is_non_decreasing
lin_space = linspace
log = math.log
log1p = math.log1p
log_sigmoid = math.log_sigmoid
logical_xor = math.logical_xor
ceil = math.ceil
matrix_band_part = linalg.band_part
matrix_inverse = linalg.inv
OpError = errors.OpError
parse_example = io.parse_example
parse_single_example = io.parse_single_example
parse_single_sequence_example = io.parse_single_sequence_example
parse_tensor = io.parse_tensor
random_gamma = random.gamma
random_normal = random.normal
random_shuffle = random.shuffle
random_uniform = random.uniform
real = math.real
reciprocal = math.reciprocal
regex_replace = strings.regex_replace
rint = math.rint
rsqrt = math.rsqrt
serialize_tensor = io.serialize_tensor
sparse_reorder = sparse.reorder
squared_difference = math.squared_difference
string_join = strings.join
string_to_number = strings.to_number
train.Server = distribute.Server
train.write_graph = io.write_graph
truncated_normal = random.truncated_normal
unsorted_segment_min = math.unsorted_segment_min
unsorted_segment_sum = math.unsorted_segment_sum
VarLenFeature = io.VarLenFeature
# pylint: enable=undefined-variable
