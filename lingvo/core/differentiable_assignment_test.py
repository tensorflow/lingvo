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
"""Tests for lingvo.differentiable_assignment."""

from absl import flags
from lingvo import compat as tf
from lingvo.core import differentiable_assignment
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string("tpu", "", "Name of TPU to connect to.")
flags.DEFINE_string("project", None, "Name of GCP project with TPU.")
flags.DEFINE_string("zone", None, "Name of GCP zone with TPU.")


class TfOpsTest(tf.test.TestCase):

  def test_max_assign_bound_all_one(self):
    score = tf.convert_to_tensor([[1.0, 1.5], [0.5, 1.1]])
    row_sums = tf.convert_to_tensor([1.0, 1.0])
    col_sums = tf.convert_to_tensor([1.0, 1.0])
    upper_bound = tf.convert_to_tensor([[1.0, 1.0], [1.0, 1.0]])

    score = score[tf.newaxis]
    row_sums = row_sums[tf.newaxis]
    col_sums = col_sums[tf.newaxis]
    upper_bound = upper_bound[tf.newaxis]

    results = differentiable_assignment.max_assignment(
        score,
        elementwise_upper_bound=upper_bound,
        row_sums=row_sums,
        col_sums=col_sums,
        epsilon=1e-3,
        num_iterations=100)
    assignment, used_iter, eps, delta = results
    correct_assignment = [[1.0, 0.0], [0.0, 1.0]]

    print("")
    print("Test case 1:")
    print("Used iter:", used_iter)
    print("Last eps:", eps)
    print("Last delta:", delta)
    print("Assignment:", assignment[0])

    self.assertShapeEqual(np.ones((1, 2, 2)), assignment)
    self.assertNDArrayNear(assignment[0], correct_assignment, err=1e-2)

  def test_max_assign_disable_one_entry(self):
    score = tf.convert_to_tensor([[1.0, 1.5], [0.5, 1.1]])
    row_sums = tf.convert_to_tensor([1.0, 1.0])
    col_sums = tf.convert_to_tensor([1.0, 1.0])
    upper_bound = tf.convert_to_tensor([[0.0, 1.0], [1.0, 1.0]])

    score = score[tf.newaxis]
    row_sums = row_sums[tf.newaxis]
    col_sums = col_sums[tf.newaxis]
    upper_bound = upper_bound[tf.newaxis]

    results = differentiable_assignment.max_assignment(
        score,
        elementwise_upper_bound=upper_bound,
        row_sums=row_sums,
        col_sums=col_sums,
        epsilon=1e-3,
        num_iterations=100)
    assignment, used_iter, eps, delta = results
    correct_assignment = [[0.0, 1.0], [1.0, 0.0]]

    print("")
    print("Test case 2:")
    print("Used iter:", used_iter)
    print("Last eps:", eps)
    print("Last delta:", delta)
    print("Assignment:", assignment[0])

    self.assertShapeEqual(np.ones((1, 2, 2)), assignment)
    self.assertNDArrayNear(assignment[0], correct_assignment, err=1e-2)

  def test_max_assign_no_epsilon_scaling(self):
    score = tf.convert_to_tensor([[1.0, 1.5], [0.5, 1.1]])
    row_sums = tf.convert_to_tensor([1.0, 1.0])
    col_sums = tf.convert_to_tensor([1.0, 1.0])
    upper_bound = tf.convert_to_tensor([[0.0, 1.0], [1.0, 1.0]])

    score = score[tf.newaxis]
    row_sums = row_sums[tf.newaxis]
    col_sums = col_sums[tf.newaxis]
    upper_bound = upper_bound[tf.newaxis]

    results = differentiable_assignment.max_assignment(
        score,
        elementwise_upper_bound=upper_bound,
        row_sums=row_sums,
        col_sums=col_sums,
        epsilon=1e-3,
        num_iterations=800,
        use_epsilon_scaling=False)
    assignment, used_iter, eps, delta = results
    correct_assignment = [[0.0, 1.0], [1.0, 0.0]]

    print("")
    print("Test case 3:")
    print("Used iter:", used_iter)
    print("Last eps:", eps)
    print("Last delta:", delta)
    print("Assignment:", assignment[0])

    self.assertShapeEqual(np.ones((1, 2, 2)), assignment)
    self.assertNDArrayNear(assignment[0], correct_assignment, err=1e-2)

  def test_max_assign_batch_version(self):
    # 2x2 example
    score1 = tf.convert_to_tensor([[0.5, 1.0], [0.2, 0.6]])
    row_sums1 = tf.convert_to_tensor([1.0, 1.0])
    col_sums1 = tf.convert_to_tensor([1.0, 1.0])
    upper_bound1 = tf.ones_like(score1)

    # 3x3 example
    score2 = tf.convert_to_tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    row_sums2 = tf.convert_to_tensor([1.0, 1.0, 1.0])
    col_sums2 = tf.convert_to_tensor([1.0, 1.0, 1.0])
    upper_bound2 = tf.ones_like(score2)

    score1 = score1[tf.newaxis]
    row_sums1 = row_sums1[tf.newaxis]
    col_sums1 = col_sums1[tf.newaxis]
    upper_bound1 = upper_bound1[tf.newaxis]

    score2 = score2[tf.newaxis]
    row_sums2 = row_sums2[tf.newaxis]
    col_sums2 = col_sums2[tf.newaxis]
    upper_bound2 = upper_bound2[tf.newaxis]

    # A batch with example 1 and example 2. We need to pad example 1.
    # Padded scores should have very large negative value.
    # Padded sums and upper bound should be zero.
    #
    score1_ = tf.pad(score1, [[0, 0], [0, 1], [0, 1]], constant_values=-1e+20)
    row_sums1_ = tf.pad(row_sums1, [[0, 0], [0, 1]])
    col_sums1_ = tf.pad(col_sums1, [[0, 0], [0, 1]])
    upper_bound1_ = tf.pad(upper_bound1, [[0, 0], [0, 1], [0, 1]])
    score3 = tf.concat([score1_, score2], axis=0)
    row_sums3 = tf.concat([row_sums1_, row_sums2], axis=0)
    col_sums3 = tf.concat([col_sums1_, col_sums2], axis=0)
    upper_bound3 = tf.concat([upper_bound1_, upper_bound2], axis=0)

    results1 = differentiable_assignment.max_assignment(
        score1,
        elementwise_upper_bound=upper_bound1,
        row_sums=row_sums1,
        col_sums=col_sums1,
        epsilon=0.01,
        num_iterations=200)
    results2 = differentiable_assignment.max_assignment(
        score2,
        elementwise_upper_bound=upper_bound2,
        row_sums=row_sums2,
        col_sums=col_sums2,
        epsilon=0.01,
        num_iterations=200)
    results3 = differentiable_assignment.max_assignment(
        score3,
        elementwise_upper_bound=upper_bound3,
        row_sums=row_sums3,
        col_sums=col_sums3,
        epsilon=0.01,
        num_iterations=200)
    assignment1 = results1[0]
    assignment2 = results2[0]
    assignment3 = results3[0]

    print("")
    print("Test case - batched:")
    print("Used iter:", results1[1], results2[1], results3[1])
    print("Delta:", results1[-1], results2[-1], results3[-1])
    print("Assignments:")
    print(assignment1[0])
    print(assignment2[0])
    print(assignment3)
    self.assertNDArrayNear(assignment1[0], assignment3[0, :2, :2], err=1e-4)
    self.assertNDArrayNear(assignment2[0], assignment3[1], err=1e-4)


if __name__ == "__main__":
  tf.test.main()
