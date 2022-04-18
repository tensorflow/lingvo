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
r"""Implementation of differentiable assignment operators in TF.

References:
[1] Csisz{\'a}r, 2008. On Iterative Algoirthms with an Information Geometry
Background.
[2] Cuturi, 2013. Lightspeed Computation of Optimal Transport.
[3] Schmitzer 2019. Stabilized Sparse Scaling Algorithms for Entropy
Regularized Transport Problems.
"""

from lingvo import compat as tf


def max_assignment(score: tf.Tensor,
                   *,
                   elementwise_upper_bound: tf.Tensor,
                   row_sums: tf.Tensor,
                   col_sums: tf.Tensor,
                   epsilon: float = 0.1,
                   num_iterations: int = 50,
                   use_epsilon_scaling: bool = True):
  """Differentiable max assignment with margin and upper bound constraints.

  Args:
    score: a 3D tensor of size [batch_size, n_rows, n_columns]. score[i, j, k]
      denotes the weight if the assignment on this entry is non-zero.
    elementwise_upper_bound: a 3D tensor of size [batch_size, n_rows,
      n_columns]. Each entry denotes the maximum value assignment[i, j, k] can
      take and must be a non-negative value. For example, upper_bound[i, j,
      k]=1.0 for binary assignment problem.
    row_sums: a 2D tensor of size [batch_size, n_rows]. The row sum constraint.
      The output assignment p[i, j, :] must sum to row_sums[i, j].
    col_sums: a 2D tensor of size [batch_size, n_columns]. The column sum
      constraint. The output assignment p[i, :, k] must sum to col_sums[i, k].
    epsilon: the epsilon coefficient of entropy regularization. The value should
      be within the range (0, 1]. `0.01` might work better than `0.1`. `0.1` may
      not make the assignment close enough to 0 or 1.
    num_iterations: the maximum number of iterations to perform.
    use_epsilon_scaling: whether to use epsilon scaling. In practice, the
      convergence of the iterative algorithm is much better if we start by
      solving the optimization with a larger epsilon value and re-use the
      solution (i.e. dual variables) for the instance with a smaller epsilon.
      This is called the epsilon scaling trick. See [Schmitzer 2019]
      (https://arxiv.org/pdf/1610.06519.pdf) as a reference. Here if
      use_epsilon_scaling=True, after each iteration we decrease the running
      epsilon by a constant factor until it reaches the target epsilon
      value. We found this to work well for gradient backward propagation,
      while the original scaling trick doesn't.

  Returns:
    A tuple with the following values.
      - assignment: a 3D tensor of size [batch_size, n_rows, n_columns].
        The output assignment.
      - used_iter: a scalar tensor indicating the number of iterations used.
      - eps: a scalar tensor indicating the stopping epsilon value.
      - delta: a scalar tensor indicating the stopping delta value (the relative
        change on the margins of assignment p in the last iteration).
  """

  # Check if all shapes are correct
  score_shape = score.shape
  bsz = score_shape[0]
  n = score_shape[1]
  m = score_shape[2]
  score = tf.ensure_shape(score, [bsz, n, m])
  elementwise_upper_bound = tf.ensure_shape(elementwise_upper_bound,
                                            [bsz, n, m])
  row_sums = tf.ensure_shape(tf.expand_dims(row_sums, axis=2), [bsz, n, 1])
  col_sums = tf.ensure_shape(tf.expand_dims(col_sums, axis=1), [bsz, 1, m])

  # the total sum of row sums must be equal to total sum of column sums
  sum_diff = tf.reduce_sum(row_sums, axis=1) - tf.reduce_sum(col_sums, axis=2)
  sum_diff = tf.abs(sum_diff)
  tf.Assert(tf.reduce_all(sum_diff < 1e-6), [sum_diff])

  # Convert upper_bound constraint into another margin constraint
  # by adding auxiliary variables & scores. Tensor `a`, `b` and `c`
  # represent the margins (i.e. reduced sum) of 3 axes respectively.
  #
  max_row_sums = tf.reduce_sum(elementwise_upper_bound, axis=-1, keepdims=True)
  max_col_sums = tf.reduce_sum(elementwise_upper_bound, axis=-2, keepdims=True)
  score_ = tf.stack([score, tf.zeros_like(score)], axis=1)  # (bsz, 2, n, m)
  a = tf.stack([row_sums, max_row_sums - row_sums], axis=1)  # (bsz, 2, n, 1)
  b = tf.stack([col_sums, max_col_sums - col_sums], axis=1)  # (bsz, 2, 1, m)
  c = tf.expand_dims(elementwise_upper_bound, axis=1)  # (bsz, 1, n, m)

  # Clip log(0) to a large negative values -1e+36 to avoid
  # getting inf or NaN values in computation. Cannot use larger
  # values because float32 would use `-inf` automatically.
  #
  tf.Assert(tf.reduce_all(a >= 0), [a])
  tf.Assert(tf.reduce_all(b >= 0), [b])
  tf.Assert(tf.reduce_all(c >= 0), [c])
  log_a = tf.maximum(tf.math.log(a), -1e+36)
  log_b = tf.maximum(tf.math.log(b), -1e+36)
  log_c = tf.maximum(tf.math.log(c), -1e+36)

  # Initialize the dual variables of margin constraints
  u = tf.zeros_like(a)
  v = tf.zeros_like(b)
  w = tf.zeros_like(c)

  eps = tf.constant(1.0 if use_epsilon_scaling else epsilon, dtype=score.dtype)
  epsilon = tf.constant(epsilon, dtype=score.dtype)

  def do_updates(cur_iter, eps, u, v, w):  # pylint: disable=unused-argument
    # Epsilon scaling, i.e. gradually decreasing `eps` until it
    # reaches the target `epsilon` value
    cur_iter = tf.cast(cur_iter, u.dtype)
    scaling = tf.minimum(0.6 * 1.04**cur_iter, 0.85)
    eps = tf.maximum(epsilon, eps * scaling)
    score_div_eps = score_ / eps

    # Update u
    log_q_1 = score_div_eps + (w + v) / eps
    log_q_1 = tf.reduce_logsumexp(log_q_1, axis=-1, keepdims=True)
    new_u = (log_a - tf.maximum(log_q_1, -1e+30)) * eps

    # Update v
    log_q_2 = score_div_eps + (w + new_u) / eps
    log_q_2 = tf.reduce_logsumexp(log_q_2, axis=-2, keepdims=True)
    new_v = (log_b - tf.maximum(log_q_2, -1e+30)) * eps

    # Update w
    log_q_3 = score_div_eps + (new_u + new_v) / eps
    log_q_3 = tf.reduce_logsumexp(log_q_3, axis=-3, keepdims=True)
    new_w = (log_c - tf.maximum(log_q_3, -1e+30)) * eps
    return eps, new_u, new_v, new_w

  def compute_relative_changes(eps, u, v, w, new_eps, new_u, new_v, new_w):
    prev_sum_uvw = tf.stop_gradient((u + v + w) / eps)
    sum_uvw = tf.stop_gradient((new_u + new_v + new_w) / new_eps)

    # Compute the relative changes on margins of P.
    # This will be used for stopping criteria.
    # Note the last update on w would guarantee the
    # margin constraint c is satisfied, so we don't
    # need to check it here.
    p = tf.exp(tf.stop_gradient(score_ / new_eps + sum_uvw))
    p_a = tf.reduce_sum(p, axis=-1, keepdims=True)
    p_b = tf.reduce_sum(p, axis=-2, keepdims=True)
    delta_a = tf.abs(a - p_a) / (a + 1e-6)
    delta_b = tf.abs(b - p_b) / (b + 1e-6)
    new_delta = tf.reduce_max(delta_a)
    new_delta = tf.maximum(new_delta, tf.reduce_max(delta_b))

    # Compute the relative changes on assignment solution P.
    # This will be used for stopping criteria.
    delta_p = tf.abs(tf.exp(prev_sum_uvw) - tf.exp(sum_uvw)) / (
        tf.exp(sum_uvw) + 1e-6)
    new_delta = tf.maximum(new_delta, tf.reduce_max(delta_p))
    return new_delta

  for cur_iter in tf.range(num_iterations):
    prev_eps, prev_u, prev_v, prev_w = eps, u, v, w
    eps, u, v, w = do_updates(cur_iter, eps, u, v, w)
  delta = compute_relative_changes(prev_eps, prev_u, prev_v, prev_w, eps, u, v,
                                   w)
  cur_iter = num_iterations
  assignment = tf.exp((score_ + u + v + w) / eps)
  assignment = assignment[:, 0]
  return assignment, cur_iter, eps, delta
