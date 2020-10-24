# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Implementation for distributed Shampoo optimizer."""

import functools
import lingvo.compat as tf
from lingvo.core import matrix_functions
from lingvo.core import ops as x_ops
import numpy as np

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
# pylint: enable=g-direct-tensorflow-import


class PartitionConfig:
  """Config for tensor partitioning."""

  def __init__(self, max_dim_size, partition_size):
    """Initialize the `PartitionConfig`.

    Args:
      max_dim_size: Partitions dimensions with size greater than this value.
      partition_size: Size of each partition
    """
    self.max_dim_size = max_dim_size
    self.partition_size = partition_size


class PartitionMetadata:
  """Metadata for partitioning."""

  def __init__(self, split_sizes_per_dim, num_splits_per_dim):
    """Initialize the `PartitionMetadata`.

    Args:
      split_sizes_per_dim: Split sizes per dimemsion.
      num_splits_per_dim: Number of splits per dimension ( inferred from
        split_sizes_per_dim).
    """
    self.split_sizes_per_dim = split_sizes_per_dim
    self.num_splits_per_dim = num_splits_per_dim


class TensorPartitioner:
  """Shards Tensor's across its axis.

  In cases of TPUs, these partitions are zero cost, and does not involve data
  movement.
  """

  @classmethod
  def partition_metadata(cls, tensor, partition_info):
    """Returns metadata required for partitioning and reforming tensors.

    Args:
      tensor: Tensor to partition.
      partition_info: Partitioning info.

    Returns:
      split_sizes_per_dim and num_splits_per_dim.
    """
    shape = tensor.get_shape()
    # Split if dim is greater than max_dim.
    axis_to_shard = [s > partition_info.max_dim_size for s in shape]
    split_sizes_per_dim = []
    # Compute the number of splits, and the sizes of the splits for
    # each dimension
    for sharded, dim in zip(axis_to_shard, shape):
      dim = int(dim)
      split_sizes_per_dim.append([dim])
      if sharded:
        split_sizes = []
        num_shards = dim // partition_info.partition_size
        if num_shards > 0:
          split_sizes = [partition_info.partition_size] * num_shards
          last_shard_size = dim % partition_info.partition_size
          if last_shard_size > 0:
            split_sizes.append(last_shard_size)
        else:
          split_sizes.append(dim)
        split_sizes_per_dim[-1] = split_sizes
    num_splits_per_dim = [len(v) for v in split_sizes_per_dim]
    return PartitionMetadata(split_sizes_per_dim, num_splits_per_dim)

  @classmethod
  def partition_tensor(cls, tensor, partition_info):
    """Returns partitioned tensors."""
    metadata = (TensorPartitioner.partition_metadata(tensor, partition_info))
    # Split from last to first axis.
    partitioned_tensors = [tensor]
    rank = len(metadata.num_splits_per_dim)
    for raxis, (num_splits, sizes) in enumerate(
        zip(
            reversed(metadata.num_splits_per_dim),
            reversed(metadata.split_sizes_per_dim))):
      if num_splits > 1:
        tmp_partitioned_tensors = []
        for item in partitioned_tensors:
          tmp_partitioned_tensors += tf.split(
              item, sizes, axis=rank - raxis - 1)
        partitioned_tensors = tmp_partitioned_tensors
    return partitioned_tensors

  @classmethod
  def reform_tensor(cls, partitioned_tensors, num_splits_per_dim):
    """Returns a tensor concatenated from the given partitions."""
    # Concatenates tensors across all dimension. Assumes the `partitions` tensor
    # was created by partition_tensor.
    for axis, num_splits in enumerate(num_splits_per_dim):
      if num_splits > 1:
        tmp_partitioned_tensors = []
        num_concat = len(partitioned_tensors) // num_splits
        for i in range(num_concat):
          tensors_to_concat = (
              partitioned_tensors[i * num_splits:(i + 1) * num_splits])
          tmp_partitioned_tensors.append(
              tf.concat(tensors_to_concat, axis=axis))
        partitioned_tensors = tmp_partitioned_tensors
    return partitioned_tensors[0]


class DistributedShampoo(optimizer.Optimizer):
  """Approximates full-matrix AdaGrad per layer.

  Approximates full-matrix AdaGrad with kronecker-products of two statistics
  matrices based on only the first-order gradients of the layer.

  "Second-order optimization made practical.", 2019
  Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer.
  """

  def __init__(self,
               learning_rate,
               momentum=0.0,
               initial_accumulator_value=0.0,
               start_preconditioning_steps=1000,
               statistics_computation_frequency=1,
               matrix_epsilon=1e-6,
               synchronous_preconditioning=False,
               second_moment_averaging=1.0,
               fallback_to_diagonal_dim=4096,
               max_any_dim=6656,
               block_size=4096,
               block_partition_threshold_size=1000000,
               global_step=None,
               exponent_multiplier=1.0,
               name="DistributedShampoo"):
    """Construct a DistributedShampoo optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      momentum: A `Tensor` or a floating point value. Momentum is not applied to
        sparse updates.
      initial_accumulator_value: A floating point value.
      start_preconditioning_steps: A int32 value which indicates when to start
        preconditioning.
      statistics_computation_frequency: A int32 step value which indicates how
        often to compute statistics for preconditioning.
      matrix_epsilon: An epsilon regularizer to make the matrices positive
        definite.
      synchronous_preconditioning: Whether to run preconditioning synchronously.
      second_moment_averaging: 1.0 means sum of gradients squares, while less
        than 1.0 switches to RMSProp style exponential moving averages of the
        second moments.
      fallback_to_diagonal_dim: Fallback to diagonal version of AFMA if the any
        of the dimension is larger than fallback_to_diagonal_dim.
      max_any_dim: If maximum value for any dimension is greater than this value
        we skip preconditioning and fall back to the diagonal.
      block_size: Dimension of the partitioned tensors.
      block_partition_threshold_size: Partitions diemnsions beyond this size.
      global_step: Global step for training.
      exponent_multiplier: A multiplier 'e` for the exponent for the inverse
        calculation. e * -1/(2*rank). Only applies when calculating inverses
        through svd.
      name: Optional name prefix for the operations created when applying
        gradients.
    """
    super().__init__(False, name)
    self._learning_rate = learning_rate
    self._momentum = momentum
    self._initial_accumulator_value = initial_accumulator_value
    self._start_preconditioning_steps = start_preconditioning_steps
    self._matrix_epsilon = matrix_epsilon
    self._synchronous_preconditioning = synchronous_preconditioning
    self._second_moment_averaging = second_moment_averaging
    self._fallback_to_diagonal_dim = fallback_to_diagonal_dim
    self._max_any_dim = max_any_dim
    self._block_size = block_size
    # NOTE: On XLA - int64 is not handled properly.
    if global_step is not None:
      self._global_step = tf.cast(tf.identity(global_step), tf.int32)
    else:
      self._global_step = tf.cast(
          tf.identity(tf.train.get_or_create_global_step()), tf.int32)
    self._run_nondiagonal_update = tf.greater_equal(
        self._global_step, self._start_preconditioning_steps)
    start_steps_f = tf.cast(self._start_preconditioning_steps, tf.float32)
    global_step_f = tf.cast(self._global_step, tf.float32)
    if start_preconditioning_steps > 0:
      self._run_nondiagonal_update_warmup = tf.minimum(
          1.0, tf.maximum((global_step_f - start_steps_f) / start_steps_f, 0.0))
    else:
      self._run_nondiagonal_update_warmup = tf.cast(1.0, tf.float32)
    # Computes statistics every K steps.
    self._statistics_computation_frequency = statistics_computation_frequency
    self._run_statistics_computation = tf.equal(
        tf.math.floormod(self._global_step,
                         self._statistics_computation_frequency), 0)
    # All vars that are preconditioned.
    self._all_vars_for_preconditioning = []
    self._exponent_multiplier = exponent_multiplier
    self._partition_info = PartitionConfig(block_partition_threshold_size,
                                           block_size)
    self._partitioner_metadata = {}

  def _fallback_to_diagonal_for_shape(self, shape):
    """Returns whether we should fallback to the diagonal update given shape."""
    # We fallback to diagonal for the following usecases:
    #
    # (a) Rank <= 1 tensors
    # (b) if any dim of Tensor is > max_any_dim.
    # (c) if all dims are 1 or are greater than fallback_to_diagonal_dim
    #
    if len(shape) <= 1:
      return True
    if any([d > self._max_any_dim for d in shape]):
      return True
    if all([d == 1 for d in shape]):
      return True
    return False

  def _preconditioner_available_for_dims(self, shape):
    """Returns indicator vector if preconditioner exists for each axis."""
    # If any of the dims < fallback_to_diagonal_dim and not 1, we run a
    # a preconditioner for that particular dimension.
    return [d <= self._fallback_to_diagonal_dim and d != 1 for d in shape]

  def _preconditioner_indices(self, shape):
    """Returns indices of the available preconditioner."""
    preconditioners_available_for_dims = (
        self._preconditioner_available_for_dims(shape))
    indices = []
    index = 0
    for is_avail_for_dim_i in preconditioners_available_for_dims:
      indices.append(index)
      if is_avail_for_dim_i:
        index += 1
    return indices

  def _make_named_slot(self, var, val, slot_name):
    _ = self._get_or_make_slot(var, val, slot_name,
                               self._name + "_" + slot_name)

  def make_named_zeros_slot(self, var, slot_name):
    self._zeros_slot(var, slot_name, self._name + "_" + slot_name)

  def _generalized_inverse_pth_root(self, input_t, exponent, epsilon=1e-12):
    input_t_f64 = tf.cast(input_t, tf.float64)
    s, u, v = tf.linalg.svd(
        input_t_f64 +
        tf.eye(tf.shape(input_t_f64)[0], dtype=tf.float64) * epsilon,
        full_matrices=True)
    inv_s = tf.reshape(
        tf.pow(tf.maximum(s, epsilon), tf.cast(exponent, tf.float64)), [1, -1])
    val = tf.matmul(u * inv_s, v, adjoint_b=True)
    return tf.cast(val, tf.float32), tf.reduce_max(tf.abs(u - v))

  def _specialized_inverse_pth_root(self, input_t, exponent, epsilon=1e-12):
    input_t_f64 = tf.cast(input_t, tf.float64)
    val, error = matrix_functions.inlined_matrix_inverse_pth_root(
        input_t_f64,
        tf.shape(input_t_f64)[0],
        exponent,
        iter_count=40,
        ridge_epsilon=epsilon)
    return tf.cast(val, tf.float32), error

  def _inverse_pth_root_graph(self, epsilon):
    graph = tf.Graph()
    with graph.as_default():
      exponent_t = tf.reshape(
          tf.placeholder(dtype=tf.float32, name="exponent", shape=None), [])
      # Apply exponent multiplier.
      exponent_t = exponent_t * self._exponent_multiplier
      input_t = tf.placeholder(dtype=tf.float32, name="input", shape=None)
      # For p = 2, 4 or 8, we use the iterative Newton-Schur method for
      # computing the inverse-pth root.
      either_p_2_4_8 = tf.math.logical_or(
          tf.math.logical_or(
              tf.equal(-1.0 / exponent_t, 2), tf.equal(-1.0 / exponent_t, 4)),
          tf.equal(-1.0 / exponent_t, 8))
      # 4096 is the larger dimension SVD is tractable for.
      greater_than_4096 = tf.greater(tf.shape(input_t)[0], 4096)
      run_specialized_iterative_method = tf.math.logical_and(
          greater_than_4096, either_p_2_4_8)
      specialized_fn = functools.partial(self._specialized_inverse_pth_root,
                                         input_t, exponent_t, epsilon)
      generalized_fn = functools.partial(self._generalized_inverse_pth_root,
                                         input_t, exponent_t, epsilon)
      output, diff = tf.cond(run_specialized_iterative_method, specialized_fn,
                             generalized_fn)

      tf.identity(output, "output")
      tf.identity(tf.cast(diff, tf.float32), "diff")
    return graph.as_graph_def().SerializeToString()

  def _create_slots(self, var_list):
    self._preconditioner_compute_graphdef = self._inverse_pth_root_graph(
        epsilon=self._matrix_epsilon)
    for v in var_list:
      self._make_named_slot(v,
                            tf.ones_like(v) * self._initial_accumulator_value,
                            "accumulator")

      if self._momentum > 0.0:
        self.make_named_zeros_slot(v, "momentum")
      shape = np.array(v.get_shape())
      self._partitioner_metadata[v] = TensorPartitioner.partition_metadata(
          v, self._partition_info)
      partitioned_v = TensorPartitioner.partition_tensor(
          v, self._partition_info)
      if not self._fallback_to_diagonal_for_shape(shape):
        self._all_vars_for_preconditioning.append(v)
        if self._momentum > 0.0:
          self.make_named_zeros_slot(v, "precond_grad_momentum")
        num_partitions = len(partitioned_v)
        for pt_idx, pt_v in enumerate(partitioned_v):
          pt_v_shape = pt_v.get_shape()
          preconditioner_exists_for_dim = (
              self._preconditioner_available_for_dims(pt_v_shape))
          for i, d in enumerate(pt_v_shape):
            if preconditioner_exists_for_dim[i]:
              mat_stat_init = array_ops.zeros([d, d], dtype=pt_v.dtype)
              self._make_named_slot(
                  v, mat_stat_init,
                  self._statistics_key_for_partition_and_dim(
                      i, pt_idx, num_partitions))
              self._make_named_slot(
                  v, mat_stat_init,
                  self._preconditioner_key_for_partition_and_dim(
                      i, pt_idx, num_partitions))

  def _prepare(self):
    learning_rate = self._call_if_callable(self._learning_rate)
    self._learning_rate_tensor = ops.convert_to_tensor(
        learning_rate, name="learning_rate")
    momentum = self._call_if_callable(self._momentum)
    self._momentum_tensor = ops.convert_to_tensor(momentum, name="momentum")

  def invoke_async_preconditioner_computation(self, global_step_int32):
    """Invokes SVD preconditioner and graph runs on the CPU."""
    keys_stats_and_rank = []
    for var in self._all_vars_for_preconditioning:
      shape = var.get_shape()
      if not self._fallback_to_diagonal_for_shape(shape):
        partitioned_v = TensorPartitioner.partition_tensor(
            var, self._partition_info)
        num_partitions = len(partitioned_v)
        for pt_idx, pt_v in enumerate(partitioned_v):
          pt_v_shape = pt_v.get_shape()
          preconditioner_exists_for_dim = (
              self._preconditioner_available_for_dims(pt_v_shape))
          for i in range(len(pt_v_shape)):
            if preconditioner_exists_for_dim[i]:
              rank = sum(preconditioner_exists_for_dim)
              key = self._key_for_var(var, i, pt_idx)
              stat = self.get_slot(
                  var,
                  self._statistics_key_for_partition_and_dim(
                      i, pt_idx, num_partitions))
              keys_stats_and_rank.append((key, stat, rank))

    if not keys_stats_and_rank:
      return tf.no_op()
    keys, stats, ranks = zip(*keys_stats_and_rank)

    return x_ops.compute_preconditioners(
        stats, [-1.0 / (2.0 * r) for r in ranks],
        global_step_int32,
        keys=keys,
        sync=self._synchronous_preconditioning,
        preconditioner_compute_graphdef=self._preconditioner_compute_graphdef)

  def assign_preconditioner_to_host_vars(self):
    """Assign/Grab latest copy of preconditioners."""
    keys_shapes_and_preconditioner_vars = []
    assign_ops = []
    for var in self._all_vars_for_preconditioning:
      shape = var.get_shape()
      if not self._fallback_to_diagonal_for_shape(shape):
        partitioned_v = TensorPartitioner.partition_tensor(
            var, self._partition_info)
        num_partitions = len(partitioned_v)
        for pt_idx, pt in enumerate(partitioned_v):
          pt_shape = pt.get_shape()
          preconditioner_exists_for_dim = (
              self._preconditioner_available_for_dims(pt_shape))
          var_rank = len(pt_shape)
          for i in range(var_rank):
            if preconditioner_exists_for_dim[i]:
              key = self._key_for_var(var, i, pt_idx)
              preconditioner = self.get_slot(
                  var,
                  self._preconditioner_key_for_partition_and_dim(
                      i, pt_idx, num_partitions))
              keys_shapes_and_preconditioner_vars.append(
                  (key, tf.shape(preconditioner), preconditioner))

      if not keys_shapes_and_preconditioner_vars:
        return tf.no_op()

      keys, shapes, preconditioner_vars = zip(
          *keys_shapes_and_preconditioner_vars)

      preconditioner_vals, successes = x_ops.get_preconditioners(
          shapes,
          keys=keys,
          preconditioner_compute_graphdef=(
              self._preconditioner_compute_graphdef))

      for preconditioner_var, preconditioner_val, success in zip(
          preconditioner_vars, preconditioner_vals, successes):
        success_mult = tf.cast(success, preconditioner.dtype)
        assign_ops.append(
            state_ops.assign(preconditioner_var,
                             (1.0 - success_mult) * preconditioner_var +
                             success_mult * preconditioner_val))
    return tf.group(*assign_ops)

  def _statistics_key_for_partition_and_dim(self, dim_index, partition_index,
                                            num_partitions):
    if num_partitions == 1:
      return "mat_statistics_" + str(dim_index)
    else:
      return str(partition_index) + "_mat_statistics_" + str(dim_index)

  def _preconditioner_key_for_partition_and_dim(self, dim_index,
                                                partition_index,
                                                num_partitions):
    if num_partitions == 1:
      return "mat_preconditioner_" + str(dim_index)
    else:
      return str(partition_index) + "_mat_preconditioner_" + str(dim_index)

  def _key_for_var(self, var, dim_index, partition_index):
    return "P_" + str(partition_index) + "_D_" + str(dim_index) + "_" + var.name

  def _updated_statistics(self, var, partitioned_grads):
    """Returns updated Shampoo statistics L_t, R_t, etc.

    Args:
      var: tf.Variable associated with the gradient.
      partitioned_grads: Partitioned gradient tensor.

    Returns:
      A list of updated statistics matrices.
    """
    precond_statistics_update = []
    num_partitions = len(partitioned_grads)
    mat_stats = []
    mat_grads = []
    mat_dims = []
    for pt_idx, pt_grad in enumerate(partitioned_grads):
      pt_shape = pt_grad.get_shape()
      preconditioner_exists_for_dim = (
          self._preconditioner_available_for_dims(pt_shape))
      rank = len(pt_shape)
      # Calculates the preconditioner statistics for each tensor.
      for i in range(rank):
        if preconditioner_exists_for_dim[i]:
          mat_stats.append(
              self.get_slot(
                  var,
                  self._statistics_key_for_partition_and_dim(
                      i, pt_idx, num_partitions)))
          mat_grads.append(pt_grad)
          mat_dims.append(i)

    # axes is the list of indices to reduce - everything but
    # the current i.
    def _update_statistics(dim, stat_var, grad):
      """Update preconditioner statistics."""
      with tf.name_scope("GradientStatistics"):
        var_rank = len(grad.get_shape())
        axes = list(range(dim)) + list(range(dim + 1, var_rank))
        new_stat = math_ops.tensordot(grad, grad, axes=(axes, axes))
        if self._second_moment_averaging == 1.0:
          updated_stat = state_ops.assign_add(stat_var, new_stat)
        else:
          updated_stat = state_ops.assign_add(
              stat_var, (self._second_moment_averaging - 1.0) * stat_var +
              (1.0 - self._second_moment_averaging) * new_stat)
        return updated_stat

    if self._statistics_computation_frequency <= 1:
      for mat_stat, mat_grad, dim in zip(mat_stats, mat_grads, mat_dims):
        precond_statistics_update.append(
            _update_statistics(dim, mat_stat, mat_grad))
    else:

      # NOTE: We rewrite tf.cond() as a while loop to avoid certain overheads
      # in XLA from buffer allocation.
      def _loop_body(mat_stats, mat_grads, mat_dims, unused_perform_step):
        precond_statistics_update_ops = []
        for mat_stat, mat_grad, dim in zip(mat_stats, mat_grads, mat_dims):
          precond_statistics_update_ops.append(
              _update_statistics(dim, mat_stat, mat_grad))
        with tf.control_dependencies(precond_statistics_update_ops):
          return tf.constant(False)

      loop_body_fn = functools.partial(_loop_body, mat_stats, mat_grads,
                                       mat_dims)
      precond_statistics_update.append(
          tf.while_loop(lambda perform_step: perform_step, loop_body_fn,
                        [self._run_statistics_computation]))

    return precond_statistics_update

  def _compute_preconditioned_raw_grad(self, var, partitioned_grads):
    """Returns preconditioned gradient.

    Args:
      var: tf.Variable associated with the gradient.
      partitioned_grads: Partitioned gradient tensor.

    Returns:
      A preconditioned gradient tensor.
    """

    partitioned_preconditioned_grads = []
    num_partitions = len(partitioned_grads)
    for pt_idx, pt_grad in enumerate(partitioned_grads):
      pt_shape = pt_grad.get_shape()
      rank = len(pt_shape)
      preconditioner_exists_for_dim = (
          self._preconditioner_available_for_dims(pt_shape))
      preconditioner_indices = self._preconditioner_indices(pt_shape)
      mat_preconditioner_list = []
      for i in range(rank):
        if preconditioner_exists_for_dim[i]:
          mat_preconditioner_list.append(
              self.get_slot(
                  var,
                  self._preconditioner_key_for_partition_and_dim(
                      i, pt_idx, num_partitions)))
      precond_grad = pt_grad
      if rank == 2 and all(preconditioner_exists_for_dim):
        # Fast path for speedup.
        precond_grad = tf.matmul(
            tf.matmul(mat_preconditioner_list[0], precond_grad),
            mat_preconditioner_list[1])
      else:
        for i in range(rank):
          if preconditioner_exists_for_dim[i]:
            precond_grad = tf.tensordot(
                precond_grad,
                mat_preconditioner_list[preconditioner_indices[i]],
                axes=([0], [0]))
          else:
            # if preconditioner is not available we transpose it to
            # permute the axis for the next preconditioner.
            precond_grad = tf.transpose(
                precond_grad, perm=list(range(1, rank)) + [0])
      partitioned_preconditioned_grads.append(precond_grad)
    return TensorPartitioner.reform_tensor(
        partitioned_preconditioned_grads,
        self._partitioner_metadata[var].num_splits_per_dim)

  def _preconditioned_update(self, var, partitioned_grads,
                             diagonal_grad_update):
    """Computes the matrix preconditioned update.

    Args:
      var: Variable for which we are computing the preconditioned gradient.
      partitioned_grads: Partitioned gradients.
      diagonal_grad_update: Update as given by diagonal adagrad.

    Returns:
      scaled preconditioned gradient.
    """

    def _l2_norm(v):
      return tf.sqrt(tf.reduce_sum(tf.square(v)))

    precond_grad = self._compute_preconditioned_raw_grad(var, partitioned_grads)
    if self._momentum > 0.0:
      gbar = self.get_slot(var, "precond_grad_momentum")
      matrix_preconditioned_grad = state_ops.assign(
          gbar, gbar * self._momentum_tensor + precond_grad *
          (1.0 - self._momentum_tensor))
    else:
      matrix_preconditioned_grad = precond_grad

    # We use the direction from Shampoo while using the step size scale from
    # diagonal AdaGrad.
    precond_l2_norm = _l2_norm(matrix_preconditioned_grad)
    diagonal_l2_norm = _l2_norm(diagonal_grad_update)
    multiplier = tf.where(
        tf.greater(precond_l2_norm, 0.0),
        tf.maximum(diagonal_l2_norm, 1e-30) /
        (tf.maximum(precond_l2_norm, 1e-30)), 1.0)
    return matrix_preconditioned_grad * multiplier

  def _apply_dense(self, grad, var):
    # Calculates the preconditioner statistics for each tensor.
    partitioned_grads = TensorPartitioner.partition_tensor(
        grad, self._partition_info)
    shape = var.get_shape()
    fallback_to_diagonal = self._fallback_to_diagonal_for_shape(shape)

    precond_statistics_update = []
    if not fallback_to_diagonal:
      precond_statistics_update = self._updated_statistics(
          var, partitioned_grads)

    accumulator = self.get_slot(var, "accumulator")
    accumulator_updated = state_ops.assign_add(accumulator, grad * grad)
    accumulator_inv_sqrt = math_ops.rsqrt(accumulator_updated + 1e-30)
    if self._momentum > 0.0:
      scaled_g = (1.0 - self._momentum_tensor) * (grad * accumulator_inv_sqrt)
      gbar = self.get_slot(var, "momentum")
      gbar_updated = state_ops.assign_add(
          gbar,
          gbar * (self._momentum_tensor - 1.0) + scaled_g)
    else:
      gbar_updated = (grad * accumulator_inv_sqrt)

    if not fallback_to_diagonal:
      # Update the preconditioner statistics followed by computing the
      # preconditioned gradient.
      with ops.control_dependencies(precond_statistics_update):
        s = tf.cast(self._run_nondiagonal_update, tf.float32)
        preconditioned_grad = self._preconditioned_update(
            var, partitioned_grads, gbar_updated)
        # slowly adapt from diagonal to preconditioned gradient.
        w = self._run_nondiagonal_update_warmup
        warmup_update = s * self._learning_rate_tensor * (
            w * preconditioned_grad + (1.0 - w) * gbar_updated)
        fallback_update = (1 - s) * (self._learning_rate_tensor * gbar_updated)
        return state_ops.assign_sub(var, warmup_update + fallback_update)
    else:
      return state_ops.assign_sub(var,
                                  self._learning_rate_tensor * gbar_updated)

  def _resource_apply_dense(self, grad, var):
    return self._apply_dense(grad, var)

  # Sparse gradients are not handled currently and is part of future work.
  def _resource_apply_sparse(self, grad_values, var, grad_indices):
    return tf.no_op()

  def _apply_sparse(self, grad, var):
    return tf.no_op()
