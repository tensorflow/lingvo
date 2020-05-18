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

import tensorflow.compat.v1 as tf1
from tensorflow.compat.v2 import *  # pylint:disable=wildcard-import, g-bad-import-order

# Import absl.flags and absl.logging to overwrite the Tensorflow ones.
# This is the intended behavior in TF 2.0.
# pylint:disable=g-bad-import-order, unused-import, g-import-not-at-top
from absl import flags
from absl import logging
# pylint: disable=g-direct-tensorflow-import

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import function as _function_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import inplace_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import module_wrapper as _module_wrapper

from tensorflow.python.platform import app
# pylint: enable=g-direct-tensorflow-import
# pylint: enable=unused-import, g-bad-import-order, g-import-not-at-top

if tf1.executing_eagerly():
  logging.warning("Lingvo does not support eager execution yet. Please disable "
                  "eager execution with tf.compat.v1.disable_eager_execution() "
                  "or proceed at your own risk.")


def _clone_module(m):
  """Shallow clone of module `m`."""
  if isinstance(m, _module_wrapper.TFModuleWrapper):
    # pylint: disable=protected-access
    return _module_wrapper.TFModuleWrapper(
        wrapped=_clone_module(m._tfmw_wrapped_module),
        module_name=m._tfmw_module_name,
        public_apis=m._tfmw_public_apis,
        deprecation=m._tfmw_print_deprecation_warnings,
        has_lite=m._tfmw_has_lite)
    # pylint: enable=protected-access
  out = type(m)(m.__name__, m.__doc__)
  out.__dict__.update(m.__dict__)
  return out

# Aliases to a few routines lingvo libraries uses often.
Defun = _function_lib.Defun
While = functional_ops.While
If = functional_ops.If
InplaceUpdate = inplace_ops.alias_inplace_update
Empty = inplace_ops.empty
EmptyLike = inplace_ops.empty_like

# pylint: disable=undefined-variable, used-before-assignment
# Move this V2 symbol here to avoid being overwritten by its following V1
# version.
where_v2 = where
while_loop_v2 = while_loop

# Import the local V2 module to maker sure the following V1 overwritting never
# applies to the global module and symbol.
data = _clone_module(data)
graph_util = _clone_module(graph_util)
image = _clone_module(image)
io = _clone_module(io)
losses = _clone_module(keras.losses)
metrics = _clone_module(keras.metrics)
nn = _clone_module(nn)
saved_model = _clone_module(saved_model)
strings = _clone_module(strings)
summary = _clone_module(summary)
test = _clone_module(test)
train = _clone_module(train)
# pylint: enable=undefined-variable, used-before-assignment

# TF 1.x symbols used in the codebase.
# To keep this list short, please use TF 2.x API whenever applicable.
# Only use TF 1.x API if it has no 2.x equivalent.
# pylint: disable=undefined-variable
add_to_collection = tf1.add_to_collection
all_variables = tf1.global_variables
# The following asserts can be directly replaced with TF2 `tf.debugging.*`
# after TF2/eager is enabled.
assert_integer = tf1.assert_integer
assert_positive = tf1.assert_positive
assert_type = tf1.assert_type
assert_scalar = tf1.assert_scalar
assign = tf1.assign
assign_add = tf1.assign_add
assign_sub = tf1.assign_sub
AUTO_REUSE = tf1.AUTO_REUSE
container = tf1.container
data.Dataset = tf1.data.Dataset
data.TFRecordDataset = tf1.data.TFRecordDataset
device = tf1.device
Dimension = tf1.Dimension
disable_eager_execution = tf1.disable_eager_execution
div = tf1.div
enable_eager_execution = tf1.enable_eager_execution
floor_div = tf1.floor_div
get_collection = tf1.get_collection
get_collection_ref = tf1.get_collection_ref
get_default_graph = tf1.get_default_graph
get_local_variable = tf1.get_local_variable
get_seed = tf1.get_seed
get_variable = tf1.get_variable
get_variable_scope = tf1.get_variable_scope
global_variables = tf1.global_variables
global_variables_initializer = tf1.global_variables_initializer
gradients = tf1.gradients
graph_util.convert_variables_to_constants = (
    tf1.graph_util.convert_variables_to_constants)
graph_util.extract_sub_graph = tf1.graph_util.extract_sub_graph
GraphDef = tf1.GraphDef
GraphKeys = tf1.GraphKeys
GraphOptions = tf1.GraphOptions
group = tf1.group
image.resize_bilinear = tf1.image.resize_bilinear
image.resize_images = tf1.image.resize_images
image.resize_nearest_neighbor = tf1.image.resize_nearest_neighbor
initialize_all_tables = tf1.initialize_all_tables
InteractiveSession = tf1.InteractiveSession
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
moving_average_variables = tf1.moving_average_variables
multinomial = tf1.multinomial
name_scope = tf1.name_scope
OptimizerOptions = tf1.OptimizerOptions
placeholder = tf1.placeholder
placeholder_with_default = tf1.placeholder_with_default
Print = tf1.Print
py_func = tf1.py_func
python_io = tf1.python_io
report_uninitialized_variables = tf1.report_uninitialized_variables
reset_default_graph = tf1.reset_default_graph
resource_loader = tf1.resource_loader
RunMetadata = tf1.RunMetadata
RunOptions = tf1.RunOptions
saved_model.build_signature_def = tf1.saved_model.build_signature_def
saved_model.Builder = tf1.saved_model.Builder
saved_model.load = tf1.saved_model.load
saved_model.loader = tf1.saved_model.loader
saved_model.signature_constants = tf1.saved_model.signature_constants
saved_model.simple_save = tf1.saved_model.simple_save
saved_model.tag_constants = tf1.saved_model.tag_constants
saved_model.utils = tf1.saved_model.utils
Session = tf1.Session
sparse_to_dense = tf1.sparse_to_dense
string_split = tf1.string_split
strings.reduce_join = tf1.reduce_join
strings.split = tf1.strings.split
Summary = tf1.Summary
if tf1.summary is not None:
  # tf.summary are not supported on TPU so we sometimes set tf.summary to None
  # to prohibit the direct use of it.
  # It is safe to skip copying tf.summary members in such cases.
  summary.audio = tf1.summary.audio
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
Variable = tf1.Variable
variables_initializer = tf1.variables_initializer
VariableScope = tf1.VariableScope
variable_scope = tf1.variable_scope
where = tf1.where
while_loop = tf1.while_loop
wrap_function = tf1.wrap_function

# Explicit 1.x symbol import.
data.make_initializable_iterator = dataset_ops.make_initializable_iterator
data.make_one_shot_iterator = dataset_ops.make_one_shot_iterator
# For `nn.embedding_lookup`, v2 doesn't have the arg 'partition_strategy' in
# the API, and uses 'partition_strategy="div"' by default;
# while v1 uses 'partition_strategy="mod"' by default. Keep this for now.
nn.embedding_lookup = embedding_ops.embedding_lookup
# pylint: enable=undefined-variable
