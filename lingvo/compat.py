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
"""The compatible tensorflow library."""

import os

# pylint: disable=g-bad-import-order, unused-import, g-import-not-at-top
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf2
from tensorflow.compat.v2 import *  # pylint: disable=wildcard-import

# Import absl.flags and absl.logging to overwrite the Tensorflow ones.
# This is the intended behavior in TF 2.0.
from absl import flags
from absl import logging
# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import random_seed
from tensorflow.python.framework import function as _function_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import inplace_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.tf2 import enabled as tf2_enabled
from tensorflow.python.util import module_wrapper as _module_wrapper
# For determining if we are running with --define=tf_api_version=1 or 2.
from tensorflow import _major_api_version
# pylint: enable=g-direct-tensorflow-import
# pylint: enable=unused-import, g-bad-import-order, g-import-not-at-top

if tf2.executing_eagerly():
  logging.info(
      "Lingvo with eager execution is in early development. "
      "Please reach out to go/lingvo-eager-migration with bugs. "
      "Eager mode can be disable with tf.compat.v1.disable_eager_execution().")


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


def summarize_tf2_status():
  """Summarize the TF version environment."""
  tf2_behavior_env = os.environ.get("TF2_BEHAVIOR")
  return "; ".join([
      f"tf._major_api_version: {_major_api_version}",
      f"tf2_enabled() == {tf2_enabled()}",
      f"TF2_BEHAVIOR == {tf2_behavior_env}",
  ])


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

# By default, with TF2 enabled and (eager execution or tf.function),
# `tf.data` API will choose the stateful implementation for methods
# `tf.data.Dataset.shuffle()`, `tf.data.Dataset.cache()` and
# `tf.data.Dataset.list_files()`. which is not compatible with
# `tf.data.make_one_shot_iterator` in TF2 (see b/162270607).
# Here is a stateless implementation of `shuffle`, `cache` and
# `list_files` to resolve the TF2 imcompatibility issue.

# Note that, these methods are meant for internal use only. Please don't use
# it unless you know exactly what you do.


class _CacheDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A `Dataset` that caches elements of its input."""

  def __init__(self, input_dataset, filename):
    """Caches the elements in the dataset."""
    self._input_dataset = input_dataset
    self._filename = ops.convert_to_tensor(
        filename, dtype=string, name="filename")
    variant_tensor = gen_dataset_ops.cache_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        filename=self._filename,
        **self._flat_structure)
    super().__init__(input_dataset, variant_tensor)


class _ShuffleDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A `Dataset` that randomly shuffles the elements of its input."""

  def __init__(self,
               input_dataset,
               buffer_size,
               seed=None,
               reshuffle_each_iteration=None):
    """Randomly shuffles the elements of this dataset."""
    self._input_dataset = input_dataset
    self._buffer_size = ops.convert_to_tensor(
        buffer_size, dtype=int64, name="buffer_size")
    self._seed, self._seed2 = random_seed.get_seed(seed)
    if reshuffle_each_iteration is None:
      reshuffle_each_iteration = True
    self._reshuffle_each_iteration = reshuffle_each_iteration

    variant_tensor = gen_dataset_ops.shuffle_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        buffer_size=self._buffer_size,
        seed=self._seed,
        seed2=self._seed2,
        reshuffle_each_iteration=self._reshuffle_each_iteration,
        **self._flat_structure)
    super().__init__(input_dataset, variant_tensor)


def stateless_shuffle_dataset(buffer_size,
                              seed=None,
                              reshuffle_each_iteration=None):
  """Randomly shuffles the elements of the dataset based on a stateless shuffle implementation.

  This method returns a stateless ShuffleDataset unconditionally. It can be
  used with `dataset.apply()` to obtain a stateless shuffled dataset, which
  supports the TF1 compatibility API `tf.data.make_one_shot_iterator()` in TF2.
  Example:
    >>> dataset = tf.data.Dataset.range(3)
    >>> dataset = dataset.apply(
    ...     stateless_shuffle_dataset((3, reshuffle_each_iteration=True))

  Args:
    buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
      elements from this dataset from which the new dataset will sample.
    seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random
      seed that will be used to create the distribution. See
      `tf.random.set_seed` for behavior.
    reshuffle_each_iteration: (Optional.) A boolean, which if true indicates
      that the dataset should be pseudorandomly reshuffled each time it is
      iterated over. (Defaults to `True`.)

  Returns:
    Dataset: A `Dataset`.
  """

  def _apply_fn(dataset):
    out_dataset = dataset_ops.DatasetV1Adapter(
        _ShuffleDataset(dataset, buffer_size, seed, reshuffle_each_iteration))
    return out_dataset

  return _apply_fn


def stateless_cache_dataset(filename=""):
  """Caches the elements in the dataset based on a stateless cache implementation.

  This method returns a stateless CacheDataset unconditionally. It can be
  used with `dataset.apply()` to obtain a stateless cached dataset, which
  supports the TF1 compatibility API `tf.data.make_one_shot_iterator()` in TF2.

  Example:
    >>> dataset = tf.data.Dataset.range(3)
    >>> dataset = dataset.apply(stateless_cache_dataset())


  Args:
    filename: A `tf.string` scalar `tf.Tensor`, representing the name of a
      directory on the filesystem to use for caching elements in this Dataset.
      If a filename is not provided, the dataset will be cached in memory.

  Returns:
    Dataset: A `Dataset`.
  """

  def _apply_fn(dataset):
    out_dataset = dataset_ops.DatasetV1Adapter(_CacheDataset(dataset, filename))
    return out_dataset

  return _apply_fn


def stateless_list_files(file_pattern, shuffle=None, seed=None):
  """A dataset of all files matching one or more glob patterns.

  Note that, if `shuffle` is not None, it will use a stateless shuffle
  implementation. Then the returned dataset supports the TF1 compatibility API
  `tf.data.make_one_shot_iterator()` in TF2.

  Example:
    >>> dataset = tf.stateless_list_files("some_file_pattern")

  Args:
    file_pattern: A string, a list of strings, or a `tf.Tensor` of string type
      (scalar or vector), representing the filename glob (i.e. shell wildcard)
      pattern(s) that will be matched.
    shuffle: (Optional.) If `True`, the file names will be shuffled randomly
      based on a stateless implementation. Defaults to `True`.
    seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random
      seed that will be used to create the distribution. See
      `tf.random.set_seed` for behavior.

  Returns:
   Dataset: A `Dataset` of strings corresponding to file names.
  """
  with ops.name_scope("list_files"):
    if shuffle is None:
      shuffle = True
    file_pattern = ops.convert_to_tensor(
        file_pattern, dtype=string, name="file_pattern")
    matching_files = gen_io_ops.matching_files(file_pattern)

    # Raise an exception if `file_pattern` does not match any files.
    condition = math_ops.greater(
        array_ops.shape(matching_files)[0], 0, name="match_not_empty")
    message = math_ops.add(
        "No files matched pattern: ",
        strings.reduce_join(file_pattern, separator=", "),
        name="message")

    assert_not_empty = debugging.Assert(
        condition, [message], summarize=1, name="assert_not_empty")
    with control_dependencies([assert_not_empty]):
      matching_files = identity(matching_files)

    dataset = data.Dataset.from_tensor_slices(matching_files)
    if shuffle:
      buffer_size = math_ops.maximum(
          shape(matching_files, out_type=dtypes.int64)[0], 1)
      # Use stateless shuffled dataset
      dataset = dataset.apply(stateless_shuffle_dataset(buffer_size, seed=seed))
    return dataset
# pylint: enable=undefined-variable, used-before-assignment


class variable_scope(tf1.variable_scope):  # pylint: disable=invalid-name
  """Override tf.compat.v1.variable_scope with an additional error message."""

  def __init__(self, *args, **kwargs):
    if tf2.executing_eagerly():
      # In eager mode, the reuse arg to variable_scope is silently overwritten
      # to AUTO_REUSE. We opt to raise an error instead.
      # https://github.com/tensorflow/tensorflow/blob/9345aee6988f50b7c571295a9e70e40e47221a64/tensorflow/python/ops/variable_scope.py#L1166
      reuse = kwargs.get("reuse", None)
      if reuse in (True, False):
        raise ValueError(
            "Setting reuse to True or False is not supported in eager mode.")
    super().__init__(*args, **kwargs)


# TF 1.x symbols used in the codebase.
# To keep this list short, please use TF 2.x API whenever applicable.
# Only use TF 1.x API if it has no 2.x equivalent.
# pylint: disable=undefined-variable
add_to_collection = tf1.add_to_collection
all_variables = tf1.global_variables
# The following asserts can be directly replaced with TF2 `tf.debugging.*`
# after TF2/eager is enabled.
app = tf1.app
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
disable_v2_behavior = tf1.disable_v2_behavior
div = tf1.div
enable_eager_execution = tf1.enable_eager_execution
executing_eagerly_outside_functions = tf1.executing_eagerly_outside_functions
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
is_variable_initialized = tf1.is_variable_initialized
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
saved_model.load_v2 = saved_model.load
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
  summary.text = tf1.summary.text
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
variance_scaling_initializer = tf1.variance_scaling_initializer
where = tf1.where
while_loop = tf1.while_loop
wrap_function = tf1.wrap_function
convert_to_tensor_or_indexed_slices = tf1.convert_to_tensor_or_indexed_slices

# Explicit 1.x symbol import.
data.make_initializable_iterator = dataset_ops.make_initializable_iterator
data.make_one_shot_iterator = dataset_ops.make_one_shot_iterator
# For `nn.embedding_lookup` and `nn.embedding_lookup_sparse`, v2 doesn't have
# the arg 'partition_strategy' in the API, and uses 'partition_strategy="div"'
# by default; while v1 uses 'partition_strategy="mod"' by default.
# Keep this for now.
nn.embedding_lookup = embedding_ops.embedding_lookup
nn.embedding_lookup_sparse = embedding_ops.embedding_lookup_sparse
# pylint: enable=undefined-variable
