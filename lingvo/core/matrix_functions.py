# Lint as: python2, python3
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
"""Matrix functions contains iterative methods for M^p."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import lingvo.compat as tf


def matrix_square_root(mat_a, mat_a_size, iter_count=100, ridge_epsilon=1e-4):
  """Iterative method to get matrix square root.

  Stable iterations for the matrix square root, Nicholas J. Higham

  Page 231, Eq 2.6b
  http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.6.8799&rep=rep1&type=pdf

  Args:
    mat_a: the symmetric PSD matrix whose matrix square root be computed
    mat_a_size: size of mat_a.
    iter_count: Maximum number of iterations.
    ridge_epsilon: Ridge epsilon added to make the matrix positive definite.

  Returns:
    mat_a^0.5
  """

  def _iter_condition(i, unused_mat_y, unused_old_mat_y, unused_mat_z,
                      unused_old_mat_z, err, old_err):
    """This method require that we check for divergence every step."""
    return tf.math.logical_and(i < iter_count, err < old_err)

  def _iter_body(i, mat_y, unused_old_mat_y, mat_z, unused_old_mat_z, err,
                 unused_old_err):
    """Iterative method to compute the square root of matrix."""
    current_iterate = 0.5 * (3.0 * identity - tf.matmul(mat_z, mat_y))
    current_mat_y = tf.matmul(mat_y, current_iterate)
    current_mat_z = tf.matmul(current_iterate, mat_z)
    # Compute the error in approximation.
    mat_sqrt_a = current_mat_y * tf.sqrt(norm)
    mat_a_approx = tf.matmul(mat_sqrt_a, mat_sqrt_a)
    residual = mat_a - mat_a_approx
    current_err = tf.sqrt(tf.reduce_sum(residual * residual)) / norm
    return i + 1, current_mat_y, mat_y, current_mat_z, mat_z, current_err, err

  identity = tf.eye(tf.cast(mat_a_size, tf.int32))
  mat_a = mat_a + ridge_epsilon * identity
  norm = tf.sqrt(tf.reduce_sum(mat_a * mat_a))
  mat_init_y = mat_a / norm
  mat_init_z = identity
  init_err = norm

  _, _, prev_mat_y, _, _, _, _ = tf.while_loop(_iter_condition, _iter_body, [
      0, mat_init_y, mat_init_y, mat_init_z, mat_init_z, init_err,
      init_err + 1.0
  ])
  return prev_mat_y * tf.sqrt(norm)


def inlined_matrix_inverse_pth_root(mat_g,
                                    mat_g_size,
                                    alpha,
                                    iter_count=100,
                                    error_tolerance=1e-6,
                                    ridge_epsilon=1e-6):
  """Computes mat_g^alpha, where alpha = -1/p, p is one of 2, 4, or 8.

  We use an iterative Schur-Newton method from equation 3.2 on page 9 of:

  A Schur-Newton Method for the Matrix p-th Root and its Inverse
  by Chun-Hua Guo and Nicholas J. Higham
  SIAM Journal on Matrix Analysis and Applications,
  2006, Vol. 28, No. 3 : pp. 788-804
  https://pdfs.semanticscholar.org/0abe/7f77433cf5908bfe2b79aa91af881da83858.pdf

  Args:
    mat_g: the symmetric PSD matrix whose power it to be computed
    mat_g_size: size of mat_g.
    alpha: exponent, must be -1/p for p a positive integer.
    iter_count: Maximum number of iterations.
    error_tolerance: Error indicator, useful for early termination.
    ridge_epsilon: Ridge epsilon added to make the matrix positive definite.

  Returns:
    mat_g^alpha
  """
  alpha = tf.cast(alpha, tf.float64)
  neg_alpha = -1.0 * alpha
  exponent = 1.0 / neg_alpha
  identity = tf.eye(tf.cast(mat_g_size, tf.int32), dtype=tf.float64)

  def _unrolled_mat_pow_2(mat_m):
    """Computes mat_m^2."""
    return tf.matmul(mat_m, mat_m)

  def _unrolled_mat_pow_4(mat_m):
    """Computes mat_m^4."""
    mat_pow_2 = _unrolled_mat_pow_2(mat_m)
    return tf.matmul(mat_pow_2, mat_pow_2)

  def _unrolled_mat_pow_8(mat_m):
    """Computes mat_m^4."""
    mat_pow_4 = _unrolled_mat_pow_4(mat_m)
    return tf.matmul(mat_pow_4, mat_pow_4)

  def mat_power(mat_m, p):
    """Computes mat_m^p, for p == 2 or 4 or 8.

    Args:
      mat_m: a square matrix
      p: a positive integer

    Returns:
      mat_m^p
    """
    branch_index = tf.cast(p / 2 - 1, tf.int32)
    return tf.switch_case(
        branch_index, {
            0: functools.partial(_unrolled_mat_pow_2, mat_m),
            1: functools.partial(_unrolled_mat_pow_4, mat_m),
            2: functools.partial(_unrolled_mat_pow_8, mat_m),
        })

  def _iter_condition(i, unused_mat_m, unused_mat_h, unused_old_mat_h, error,
                      run_step):
    return tf.math.logical_and(
        tf.math.logical_and(i < iter_count, error > error_tolerance), run_step)

  def _iter_body(i, mat_m, mat_h, unused_old_mat_h, error, unused_run_step):
    mat_m_i = (1 - alpha) * identity + alpha * mat_m
    new_mat_m = tf.matmul(mat_power(mat_m_i, exponent), mat_m)
    new_mat_h = tf.matmul(mat_h, mat_m_i)
    new_error = tf.reduce_max(tf.abs(new_mat_m - identity))
    return (i + 1, new_mat_m, new_mat_h, mat_h, new_error, new_error < error)

  if mat_g_size == 1:
    mat_h = tf.pow(mat_g + ridge_epsilon, alpha)
  else:
    damped_mat_g = mat_g + ridge_epsilon * identity
    z = (1 - 1 / alpha) / (2 * tf.norm(damped_mat_g))
    # The best value for z is
    # (1 - 1/alpha) * (c_max^{-alpha} - c_min^{-alpha}) /
    #                 (c_max^{1-alpha} - c_min^{1-alpha})
    # where c_max and c_min are the largest and smallest singular values of
    # damped_mat_g.
    # The above estimate assumes that c_max > c_min * 2^p. (p = -1/alpha)
    # Can replace above line by the one below, but it is less accurate,
    # hence needs more iterations to converge.
    # z = (1 - 1/alpha) / tf.trace(damped_mat_g)
    # If we want the method to always converge, use z = 1 / norm(damped_mat_g)
    # or z = 1 / tf.trace(damped_mat_g), but these can result in many
    # extra iterations.
    new_mat_m_0 = damped_mat_g * z
    new_error = tf.reduce_max(tf.abs(new_mat_m_0 - identity))
    new_mat_h_0 = identity * tf.pow(z, neg_alpha)
    _, mat_m, mat_h, old_mat_h, error, convergence = tf.while_loop(
        _iter_condition, _iter_body,
        [0, new_mat_m_0, new_mat_h_0, new_mat_h_0, new_error, True])
    error = tf.reduce_max(tf.abs(mat_m - identity))
    is_converged = tf.cast(convergence, old_mat_h.dtype)
    resultant_mat_h = is_converged * mat_h + (1 - is_converged) * old_mat_h
  return resultant_mat_h, error
