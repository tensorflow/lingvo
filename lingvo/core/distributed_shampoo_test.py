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
"""Tests for lingvo.distributed_shampoo."""

import lingvo.compat as tf
from lingvo.core import distributed_shampoo
from lingvo.core import test_utils

import numpy as np


class DistributedShampooTest(test_utils.TestCase):
  """A test that demonstrates the use of distributed matrix preconditioner."""

  def testShampooWithMatrixShapedTensors(self):
    # Parameter matrix of size [4,2] would result in L_{t}, and R_{t} of
    # sizes [4, 4] and [2, 2]
    size = [4, 2]
    init_var_np = np.zeros(size)
    # Initialize gradient as random tensor.
    grad_np = np.random.rand(size[0], size[1])

    with tf.Session():
      global_step = tf.Variable(0, dtype=tf.int64)
      var = tf.Variable(init_var_np, dtype=tf.float32)
      grad = tf.constant(grad_np, dtype=tf.float32)

      opt = distributed_shampoo.DistributedShampoo(
          learning_rate=1.0,
          momentum=0.0,
          start_preconditioning_steps=0,
          synchronous_preconditioning=True,
          global_step=global_step)

      # Run a single step of gradient update.
      update = opt.apply_gradients(zip([grad], [var]), global_step=global_step)

      # Preconditioner computation and assignments to variables.
      compute_preconditioner_op = opt.invoke_async_preconditioner_computation(
          tf.cast(global_step, tf.int32))
      assign_preconditioners_to_vars_op = (
          opt.assign_preconditioner_to_host_vars())

      self.evaluate(tf.global_variables_initializer())
      tf.tables_initializer().run()

      init_val = self.evaluate(var)
      self.assertAllCloseAccordingToType(init_var_np, init_val)

      def np_power(mat_g, alpha, matrix_epsilon=1e-6):
        """Computes mat_g^alpha for a square symmetric matrix mat_g."""
        mat_for_svd = mat_g + np.eye(mat_g.shape[0]) * matrix_epsilon
        mat_u, diag_d, mat_v = np.linalg.svd(mat_for_svd, full_matrices=True)
        diag_d = np.power(np.maximum(diag_d, matrix_epsilon), alpha)
        return np.dot(mat_u, np.dot(np.diag(diag_d), mat_v))

      def norm(val):
        return np.sqrt(np.sum(np.square(val)))

      # Run a step of preconditioner update.
      update.run()

      mat_g1 = np.dot(grad_np, grad_np.transpose())
      expected_mat_g1 = self.evaluate(opt.get_slot(var, 'mat_statistics_0'))
      self.assertAllCloseAccordingToType(mat_g1, expected_mat_g1, atol=1e-1)

      mat_g2 = np.dot(grad_np.transpose(), grad_np)
      expected_mat_g2 = self.evaluate(opt.get_slot(var, 'mat_statistics_1'))
      self.assertAllCloseAccordingToType(mat_g2, expected_mat_g2, atol=1e-1)

      compute_preconditioner_op.run()
      assign_preconditioners_to_vars_op.run()

      mat_left = np_power(mat_g1, -0.25)
      expected_mat_left = self.evaluate(
          opt.get_slot(var, 'mat_preconditioner_0'))
      self.assertAllCloseAccordingToType(mat_left, expected_mat_left, atol=1e-1)

      mat_right = np_power(mat_g2, -0.25)
      expected_mat_right = self.evaluate(
          opt.get_slot(var, 'mat_preconditioner_1'))
      self.assertAllCloseAccordingToType(
          mat_right, expected_mat_right, atol=1e-1)

      # As the preconditioners are initialized to all zero. We don't make
      # any update.
      var_step_0_val = self.evaluate(var)
      self.assertAllCloseAccordingToType(init_var_np, var_step_0_val, atol=1e-1)

      # Run another step of training.
      update.run()
      var_step_1_val = self.evaluate(var)

      # New update has the scale of the second diagonal adagrad update.
      adagrad_update = grad_np / np.sqrt(2 * np.square(grad_np))
      preconditioned_grad_update = np.dot(np.dot(mat_left, grad_np), mat_right)

      # With normalization by diagonal enabled.
      var_step_1_np = init_var_np - preconditioned_grad_update * norm(
          adagrad_update) / norm(preconditioned_grad_update)
      self.assertAllCloseAccordingToType(
          var_step_1_np, var_step_1_val, atol=1e-1)

      # Compute new preconditioners.
      compute_preconditioner_op.run()
      assign_preconditioners_to_vars_op.run()

      # Gradients are summed over time.
      mat_g1 += np.dot(grad_np, grad_np.transpose())
      mat_left = np_power(mat_g1, -0.25)
      expected_mat_left = self.evaluate(
          opt.get_slot(var, 'mat_preconditioner_0'))
      self.assertAllCloseAccordingToType(mat_left, expected_mat_left, atol=1e-1)

      mat_g2 += np.dot(grad_np.transpose(), grad_np)
      mat_right = np_power(mat_g2, -0.25)
      expected_mat_right = self.evaluate(
          opt.get_slot(var, 'mat_preconditioner_1'))
      self.assertAllCloseAccordingToType(
          mat_right, expected_mat_right, atol=1e-1)

  def testShampooWithMatrixShapedTensorsRightOnlyPreconditioner(self):
    # Parameter matrix of size [4,2] would result in L_{t}, and R_{t} of
    # sizes [4, 4] and [2, 2]. Since max_any_dim is set to 3, it would skip
    # L_{t} and only use R_{t}. The exponent in the inverse used to compute
    # the preconditioner becomes -1/2.
    size = [4, 2]
    init_var_np = np.zeros(size)
    # Initialize gradient as random tensor.
    grad_np = np.random.rand(size[0], size[1])

    with tf.Session():
      global_step = tf.Variable(0, dtype=tf.int64)
      var = tf.Variable(init_var_np, dtype=tf.float32)
      grad = tf.constant(grad_np, dtype=tf.float32)

      opt = distributed_shampoo.DistributedShampoo(
          learning_rate=1.0,
          momentum=0.0,
          fallback_to_diagonal_dim=3,
          start_preconditioning_steps=0,
          synchronous_preconditioning=True,
          global_step=global_step)

      # Run a single step of gradient update.
      update = opt.apply_gradients(zip([grad], [var]), global_step=global_step)

      # Preconditioner computation and assignments to variables.
      compute_preconditioner_op = opt.invoke_async_preconditioner_computation(
          tf.cast(global_step, tf.int32))
      assign_preconditioners_to_vars_op = (
          opt.assign_preconditioner_to_host_vars())

      self.evaluate(tf.global_variables_initializer())
      tf.tables_initializer().run()

      init_val = self.evaluate(var)
      self.assertAllCloseAccordingToType(init_var_np, init_val)

      def np_power(mat_g, alpha, matrix_epsilon=1e-6):
        """Computes mat_g^alpha for a square symmetric matrix mat_g."""
        mat_for_svd = mat_g + np.eye(mat_g.shape[0]) * matrix_epsilon
        mat_u, diag_d, mat_v = np.linalg.svd(mat_for_svd, full_matrices=True)
        diag_d = np.power(np.maximum(diag_d, matrix_epsilon), alpha)
        return np.dot(mat_u, np.dot(np.diag(diag_d), mat_v))

      def norm(val):
        return np.sqrt(np.sum(np.square(val)))

      # Run a step of preconditioner update.
      update.run()

      mat_g2 = np.dot(grad_np.transpose(), grad_np)
      expected_mat_g2 = self.evaluate(opt.get_slot(var, 'mat_statistics_1'))
      self.assertAllCloseAccordingToType(mat_g2, expected_mat_g2, atol=1e-1)

      compute_preconditioner_op.run()
      assign_preconditioners_to_vars_op.run()

      mat_right = np_power(mat_g2, -0.5)
      expected_mat_right = self.evaluate(
          opt.get_slot(var, 'mat_preconditioner_1'))
      self.assertAllCloseAccordingToType(
          mat_right, expected_mat_right, atol=1e-1)

      # As the preconditioners are initialized to all zero. We don't make
      # any update.
      var_step_0_val = self.evaluate(var)
      self.assertAllCloseAccordingToType(init_var_np, var_step_0_val, atol=1e-1)

      # Run another step of training.
      update.run()
      var_step_1_val = self.evaluate(var)

      # New update has the scale of the second diagonal adagrad update.
      adagrad_update = grad_np / np.sqrt(2 * np.square(grad_np))
      preconditioned_grad_update = np.matmul(grad_np, mat_right)

      # With normalization by diagonal enabled.
      var_step_1_np = init_var_np - preconditioned_grad_update * norm(
          adagrad_update) / norm(preconditioned_grad_update)

      self.assertAllCloseAccordingToType(
          var_step_1_np, var_step_1_val, atol=1e-1)

      # Compute new preconditioners.
      compute_preconditioner_op.run()
      assign_preconditioners_to_vars_op.run()

      # Gradients are summed over time.

      mat_g2 += np.dot(grad_np.transpose(), grad_np)
      mat_right = np_power(mat_g2, -0.5)
      expected_mat_right = self.evaluate(
          opt.get_slot(var, 'mat_preconditioner_1'))
      self.assertAllCloseAccordingToType(
          mat_right, expected_mat_right, atol=1e-1)

  def testShampooWithMatrixShapedTensorsWithBlocks(self):
    # Parameter matrix of size [4,2] would result in 4 L_{t}, and R_{t} of
    # sizes [2, 2] and [2, 2].
    size = [4, 2]
    init_var_np = np.zeros(size)
    # Initialize gradient as random tensor.
    grad_np = np.random.rand(size[0], size[1])

    with tf.Session():
      global_step = tf.Variable(0, dtype=tf.int64)
      var = tf.Variable(init_var_np, dtype=tf.float32)
      grad = tf.constant(grad_np, dtype=tf.float32)

      opt = distributed_shampoo.DistributedShampoo(
          learning_rate=1.0,
          momentum=0.0,
          block_partition_threshold_size=3,
          block_size=2,
          start_preconditioning_steps=0,
          synchronous_preconditioning=True,
          global_step=global_step)

      # Run a single step of gradient update.
      update = opt.apply_gradients(zip([grad], [var]), global_step=global_step)

      # Preconditioner computation and assignments to variables.
      compute_preconditioner_op = opt.invoke_async_preconditioner_computation(
          tf.cast(global_step, tf.int32))
      assign_preconditioners_to_vars_op = (
          opt.assign_preconditioner_to_host_vars())

      self.evaluate(tf.global_variables_initializer())
      tf.tables_initializer().run()

      init_val = self.evaluate(var)
      self.assertAllCloseAccordingToType(init_var_np, init_val)

      def np_power(mat_g, alpha, matrix_epsilon=1e-6):
        """Computes mat_g^alpha for a square symmetric matrix mat_g."""
        mat_for_svd = mat_g + np.eye(mat_g.shape[0]) * matrix_epsilon
        mat_u, diag_d, mat_v = np.linalg.svd(mat_for_svd, full_matrices=True)
        diag_d = np.power(np.maximum(diag_d, matrix_epsilon), alpha)
        return np.dot(mat_u, np.dot(np.diag(diag_d), mat_v))

      def norm(val):
        return np.sqrt(np.sum(np.square(val)))

      # Run a step of preconditioner update.
      update.run()

      block_0_grad_np = grad_np[:2, :2]
      block_1_grad_np = grad_np[2:4, :2]

      block_0_mat_g1 = np.dot(block_0_grad_np, block_0_grad_np.transpose())
      expected_block_0_mat_g1 = self.evaluate(
          opt.get_slot(var, '0_mat_statistics_0'))

      self.assertAllCloseAccordingToType(
          block_0_mat_g1, expected_block_0_mat_g1, atol=1e-1)

      block_0_mat_g2 = np.dot(block_0_grad_np.transpose(), block_0_grad_np)
      expected_block_0_mat_g2 = self.evaluate(
          opt.get_slot(var, '0_mat_statistics_1'))
      self.assertAllCloseAccordingToType(
          block_0_mat_g2, expected_block_0_mat_g2, atol=1e-1)

      block_1_mat_g1 = np.dot(block_1_grad_np, block_1_grad_np.transpose())
      expected_block_1_mat_g1 = self.evaluate(
          opt.get_slot(var, '1_mat_statistics_0'))
      self.assertAllCloseAccordingToType(
          block_1_mat_g1, expected_block_1_mat_g1, atol=1e-1)

      block_1_mat_g2 = np.dot(block_1_grad_np.transpose(), block_1_grad_np)
      expected_block_1_mat_g2 = self.evaluate(
          opt.get_slot(var, '1_mat_statistics_1'))
      self.assertAllCloseAccordingToType(
          block_1_mat_g2, expected_block_1_mat_g2, atol=1e-1)

      compute_preconditioner_op.run()
      assign_preconditioners_to_vars_op.run()

      block_0_mat_left = np_power(block_0_mat_g1, -0.25)
      expected_block_0_mat_left = self.evaluate(
          opt.get_slot(var, '0_mat_preconditioner_0'))
      self.assertAllCloseAccordingToType(
          block_0_mat_left, expected_block_0_mat_left, atol=1e-1)

      block_0_mat_right = np_power(block_0_mat_g2, -0.25)
      expected_block_0_mat_right = self.evaluate(
          opt.get_slot(var, '0_mat_preconditioner_1'))
      self.assertAllCloseAccordingToType(
          block_0_mat_right, expected_block_0_mat_right, atol=1e-1)

      block_1_mat_left = np_power(block_1_mat_g1, -0.25)
      expected_block_1_mat_left = self.evaluate(
          opt.get_slot(var, '1_mat_preconditioner_0'))
      self.assertAllCloseAccordingToType(
          block_1_mat_left, expected_block_1_mat_left, atol=1e-1)

      block_1_mat_right = np_power(block_1_mat_g2, -0.25)
      expected_block_1_mat_right = self.evaluate(
          opt.get_slot(var, '1_mat_preconditioner_1'))
      self.assertAllCloseAccordingToType(
          block_1_mat_right, expected_block_1_mat_right, atol=1e-1)

      # As the preconditioners are initialized to all zero. We don't make
      # any update.
      var_step_0_val = self.evaluate(var)
      self.assertAllCloseAccordingToType(init_var_np, var_step_0_val, atol=1e-1)

      # Run another step of training.
      update.run()
      var_step_1_val = self.evaluate(var)

      # New update has the scale of the second diagonal adagrad update.
      adagrad_update = grad_np / np.sqrt(2 * np.square(grad_np))

      block_0_update = np.dot(
          np.dot(block_0_mat_left, block_0_grad_np), block_0_mat_right)
      block_1_update = np.dot(
          np.dot(block_1_mat_left, block_1_grad_np), block_1_mat_right)
      preconditioned_grad_update = np.concatenate(
          (block_0_update, block_1_update), axis=0)
      # With normalization by diagonal enabled.
      var_step_1_np = init_var_np - preconditioned_grad_update * norm(
          adagrad_update) / norm(preconditioned_grad_update)
      self.assertAllCloseAccordingToType(
          var_step_1_np, var_step_1_val, atol=1e-1)

      # Compute new preconditioners.
      compute_preconditioner_op.run()
      assign_preconditioners_to_vars_op.run()

      # Gradients are summed over time.
      block_0_mat_g1 += np.dot(block_0_grad_np, block_0_grad_np.transpose())
      block_0_mat_left = np_power(block_0_mat_g1, -0.25)
      expected_block_0_mat_left = self.evaluate(
          opt.get_slot(var, '0_mat_preconditioner_0'))
      self.assertAllCloseAccordingToType(
          block_0_mat_left, expected_block_0_mat_left, atol=1e-1)

      block_0_mat_g2 += np.dot(block_0_grad_np.transpose(), block_0_grad_np)
      block_0_mat_right = np_power(block_0_mat_g2, -0.25)
      expected_block_0_mat_right = self.evaluate(
          opt.get_slot(var, '0_mat_preconditioner_1'))
      self.assertAllCloseAccordingToType(
          block_0_mat_right, expected_block_0_mat_right, atol=1e-1)

      block_1_mat_g1 += np.dot(block_1_grad_np, block_1_grad_np.transpose())
      block_1_mat_left = np_power(block_1_mat_g1, -0.25)
      expected_block_1_mat_left = self.evaluate(
          opt.get_slot(var, '1_mat_preconditioner_0'))
      self.assertAllCloseAccordingToType(
          block_1_mat_left, expected_block_1_mat_left, atol=1e-1)

      block_1_mat_g2 += np.dot(block_1_grad_np.transpose(), block_1_grad_np)
      block_1_mat_right = np_power(block_1_mat_g2, -0.25)
      expected_block_1_mat_right = self.evaluate(
          opt.get_slot(var, '1_mat_preconditioner_1'))
      self.assertAllCloseAccordingToType(
          block_1_mat_right, expected_block_1_mat_right, atol=1e-1)


class TensorPartitionerTest(test_utils.TestCase):
  """Tensor partitioner tests."""

  def testTensorPartitioner(self):
    with tf.Session():
      w1 = tf.get_variable('w1', [255, 255], tf.float32)
      self.evaluate(tf.global_variables_initializer())
      partition_info = distributed_shampoo.PartitionConfig(200, 128)
      grad = tf.constant(w1.eval())
      metadata = distributed_shampoo.TensorPartitioner.partition_metadata(
          w1, partition_info)
      partitioned_grad = distributed_shampoo.TensorPartitioner.partition_tensor(
          w1, partition_info)
      reformed_grad = distributed_shampoo.TensorPartitioner.reform_tensor(
          partitioned_grad, metadata.num_splits_per_dim)
      self.assertAllCloseAccordingToType(reformed_grad, grad)


if __name__ == '__main__':
  tf.test.main()
