# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for cudnn_rnn_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os

from six.moves import zip
import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.python.training import saver as saver_lib
from lingvo.core import cudnn_rnn_utils
from lingvo.core import test_utils

UNIDIR = cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION
BIDIR = cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION

CUDNN_LSTM = cudnn_rnn_ops.CUDNN_LSTM
CUDNN_LSTM_PARAMS_PER_LAYER = cudnn_rnn_ops.CUDNN_LSTM_PARAMS_PER_LAYER


class CuDNNRNNUtilsTest(test_utils.TestCase):

  def testCuDNNInitializerWrapper(self):
    if not tf.test.is_gpu_available(cuda_only=True):
      return
    dirs = [UNIDIR, BIDIR]
    input_nodes = [128, 256, 512, 1024]
    cell_nodes = [128, 256, 512, 1024]
    dtypes = [tf.float32, tf.float64]
    for direction, input_dim, cell_dim, dtype in itertools.product(
        dirs, input_nodes, cell_nodes, dtypes):
      with self.session(use_gpu=True, graph=tf.Graph()):
        base_init = tf.ones_initializer()
        cudnn_initializer = cudnn_rnn_utils.CuDNNLSTMInitializer(
            input_dim, cell_dim, direction)
        actual = cudnn_initializer.InitOpaqueParams(dtype, base_init).eval()
        num_dir = 1 if direction == UNIDIR else 2
        expected = tf.concat(
            [
                tf.ones(
                    [num_dir * 4 * cell_dim * (cell_dim + input_dim)],
                    dtype=dtype),
                tf.zeros([num_dir * 8 * cell_dim], dtype=dtype)
            ],
            axis=0).eval()

        self.assertAllClose(expected, actual)


class CuDNNLSTMSaveableTest(test_utils.TestCase):

  def testSaveRestoreUnidi(self):
    if not tf.test.is_gpu_available(cuda_only=True):
      return
    with tf.device('/gpu:0'):
      self._TestSaveRestoreHelper(UNIDIR)

  def testSaveRestoreBiDi(self):
    if not tf.test.is_gpu_available(cuda_only=True):
      return
    with tf.device('/gpu:0'):
      self._TestSaveRestoreHelper(BIDIR)

  def _TestSaveRestoreHelper(self, direction):
    """Test opaque params stay 'equivalent' after save-restore."""
    input_dim = 4
    cell_dim = 3

    with tf.variable_scope('s1'):
      params_size_t = self._ParamsSize(input_dim, cell_dim, direction)
      params = tf.get_variable(
          'cudnn_params',
          initializer=tf.random_uniform([params_size_t]),
          validate_shape=False)
      reset_params_op = tf.assign(params, tf.zeros_like(params))
      cur_scope_name = tf.get_variable_scope().name
      saveable = self._CreateSaveable(params, input_dim, cell_dim, direction,
                                      cur_scope_name)
      canonical_wts, canonical_bs = (
          saveable.format_converter._opaque_to_cu_canonical(
              saveable._variables))
      saver = saver_lib.Saver()
    with self.session(use_gpu=True) as sess:
      sess.run(tf.global_variables_initializer())
      save_path = os.path.join(self.get_temp_dir(), 'save-restore-unidi')
      saver.save(sess, save_path)
      canonical_wts_v, canonical_bs_v = sess.run([canonical_wts, canonical_bs])

    with self.session(use_gpu=False) as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(reset_params_op)
      saver.restore(sess, save_path)
      canonical_wts_v_restored, canonical_bs_v_restored = sess.run(
          [canonical_wts, canonical_bs])
      # Weight porition of the opaque params are exactly the same. For biases
      # porition, it's expected that the sum of biases each gate stays the same.
      self._CompareWeights(canonical_wts_v, canonical_wts_v_restored)
      self._CompareBiases(canonical_bs_v, canonical_bs_v_restored, direction)

  def _CreateSaveable(self, opaque_params, input_dim, cell_dim, direction,
                      scope):
    rnn_cell_name = 'rnn_cell'
    if direction == UNIDIR:
      saveable = cudnn_rnn_utils.CuDNNLSTMSaveable(
          opaque_params, cell_dim, input_dim, rnn_cell_name, scope,
          opaque_params.name + '_saveable')
    else:
      fwd_cell_name = 'fwd'
      bak_cell_name = 'bak'
      saveable = cudnn_rnn_utils.BidiCuDNNLSTMSaveable(
          opaque_params, cell_dim, input_dim, fwd_cell_name, bak_cell_name,
          scope, opaque_params.name + '_saveable')
    tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
    return saveable

  def _ParamsSize(self, input_dim, cell_dim, direction, dtype=tf.float32):
    return cudnn_rnn_ops.cudnn_rnn_opaque_params_size(
        rnn_mode=cudnn_rnn_ops.CUDNN_LSTM,
        num_layers=1,
        num_units=cell_dim,
        input_size=input_dim,
        input_mode=cudnn_rnn_ops.CUDNN_INPUT_LINEAR_MODE,
        direction=direction,
        dtype=dtype)

  def _CompareWeights(self, lhs, rhs):
    self.assertEqual(len(lhs), len(rhs))
    for lw, rw in zip(lhs, rhs):
      self.assertAllEqual(lw, rw)

  def _CompareBiases(self, lhs, rhs, direction):
    self.assertEqual(len(lhs), len(rhs))
    if direction == UNIDIR:
      self._CompareSingleLayerBiases(lhs, rhs)
    else:
      size = len(lhs)
      fw_lhs, bw_lhs = lhs[:size // 2], lhs[size // 2:]
      fw_rhs, bw_rhs = rhs[:size // 2], rhs[size // 2:]
      self._CompareSingleLayerBiases(fw_lhs, fw_rhs)
      self._CompareSingleLayerBiases(bw_lhs, bw_rhs)

  def _CompareSingleLayerBiases(self, lhs, rhs):
    self.assertEqual(len(lhs), len(rhs))
    self.assertEqual(len(lhs) % 2, 0)

    lf_lhs, rt_lhs = lhs[:len(lhs) // 2], lhs[len(lhs) // 2:]
    lf_rhs, rt_rhs = rhs[:len(rhs) // 2], rhs[len(rhs) // 2:]

    sum_lhs, sum_rhs = [], []
    for lf, rt in zip(lf_lhs, rt_lhs):
      sum_lhs.append(lf + rt)
    for lf, rt in zip(lf_rhs, rt_rhs):
      sum_rhs.append(lf + rt)

    self.assertEqual(len(sum_lhs), len(sum_rhs))
    for lf, rt in zip(sum_lhs, sum_rhs):
      self.assertAllEqual(lf, rt)


if __name__ == '__main__':
  tf.test.main()
