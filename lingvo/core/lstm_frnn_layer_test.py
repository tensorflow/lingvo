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
"""Tests for lingvo.core.lstm_frnn_layer."""

from absl.testing import parameterized

from lingvo import compat as tf
from lingvo.core import lstm_frnn_layer
from lingvo.core import py_utils
from lingvo.core import rnn_cell
from lingvo.core import rnn_layers
from lingvo.core import test_utils

import numpy as np

_INIT_RANDOM_SEED = 2020
_RANDOM_SEED = 2019


class LSTMCellExtTest(test_utils.TestCase, parameterized.TestCase):

  def _GetParams(self, num_hidden_nodes=None):
    params = lstm_frnn_layer.LayerNormalizedLSTMCellLeanExt.Params()
    params.name = 'lstm'
    params.output_nonlinearity = True
    params.params_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)
    params.vn.global_vn = False
    params.vn.per_step_vn = False
    params.num_input_nodes = 2
    params.num_output_nodes = 2
    if num_hidden_nodes:
      params.num_hidden_nodes = num_hidden_nodes
    params.zo_prob = 0.0
    params.random_seed = _RANDOM_SEED
    return params

  def testLayerNormalizedLSTMCellLeanExt(self):
    cell_p = self._GetParams()

    seqlen, batch, input_dim = 4, 2, 2
    inputs = tf.convert_to_tensor(
        np.random.rand(seqlen, batch, input_dim).astype(np.float32))
    input_lens = np.random.randint(1, seqlen + 1, size=batch)
    paddings = 1. - tf.sequence_mask(
        input_lens, maxlen=seqlen, dtype=tf.float32)
    paddings = tf.transpose(paddings)
    reset_mask = tf.zeros((seqlen, batch), tf.float32)
    m0 = tf.convert_to_tensor(
        np.random.rand(batch, input_dim).astype(np.float32))
    c0 = tf.convert_to_tensor(
        np.random.rand(batch, input_dim).astype(np.float32))
    state0 = py_utils.NestedMap(m=m0, c=c0)

    with self.session():
      cell = cell_p.Instantiate()
      self.evaluate(tf.global_variables_initializer())

      # The canonical path
      state = state0
      for i in range(seqlen):
        state, _ = cell.FPropDefaultTheta(
            state,
            py_utils.NestedMap(
                act=[inputs[i, :, :]],
                padding=paddings[i, :, tf.newaxis],
                reset_mask=reset_mask[i, :, tf.newaxis]))
      expected_state = self.evaluate(state)

      # Taking input projection outside of the loop.
      cell_theta = cell.theta.copy()
      cell_theta.wm_i = cell_theta.wm[:cell.params.num_input_nodes, :]
      cell_theta.wm_h = cell_theta.wm[cell.params.num_input_nodes:, :]
      proj_inputs = cell.ProjectInputSequence(cell_theta,
                                              py_utils.NestedMap(act=[inputs]))
      state = state0
      for i in range(seqlen):
        state, _ = cell.FPropWithProjectedInput(
            cell_theta, state,
            py_utils.NestedMap(
                proj_inputs=proj_inputs[i, :, :],
                padding=paddings[i, :, tf.newaxis],
                reset_mask=reset_mask[i, :, tf.newaxis]))
      actual_state = self.evaluate(state)

    tf.logging.info('expected_state:{}'.format(expected_state))
    tf.logging.info('actual_state:{}'.format(actual_state))
    self.assertAllClose(expected_state.m, actual_state.m)
    self.assertAllClose(expected_state.c, actual_state.c)


class LstmFRNNTest(test_utils.TestCase, parameterized.TestCase):

  def _SetCellParams(self, cell_p, num_hidden_nodes=None):
    params = cell_p
    params.name = cell_p.cls.__name__
    params.output_nonlinearity = True
    params.params_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)
    params.vn.global_vn = False
    params.vn.per_step_vn = False
    params.num_input_nodes = 7
    params.num_output_nodes = 9
    if num_hidden_nodes:
      params.num_hidden_nodes = num_hidden_nodes
    params.zo_prob = 0.0
    params.random_seed = _RANDOM_SEED

  def _GetTestInputs(self, packed_input):
    seqlen, batch, input_dim, output_dim = 4, 5, 7, 9
    inputs = tf.convert_to_tensor(
        np.random.rand(seqlen, batch, input_dim).astype(np.float32))
    input_lens = np.random.randint(1, seqlen + 1, size=batch)
    padding = 1. - tf.sequence_mask(input_lens, maxlen=seqlen, dtype=tf.float32)
    padding = tf.transpose(padding)[:, :, tf.newaxis]
    segment_id = None
    if packed_input:
      segment_id = tf.convert_to_tensor(
          np.random.randint(0, seqlen, (seqlen, batch, 1), np.int32))

    m = tf.convert_to_tensor(
        np.random.rand(batch, output_dim).astype(np.float32))
    c = tf.convert_to_tensor(
        np.random.rand(batch, output_dim).astype(np.float32))
    return inputs, padding, m, c, segment_id

  def _testHelper(self, base_frnn_p, frnn_p, packed_input=False):
    inputs, padding, m0, c0, segment_id = self._GetTestInputs(packed_input)
    base_frnn = base_frnn_p.Instantiate()
    frnn = frnn_p.Instantiate()

    with self.session():
      self.evaluate(tf.global_variables_initializer())

      state0 = py_utils.NestedMap(m=m0, c=c0)
      act, state = base_frnn.FPropDefaultTheta(
          inputs, padding, state0=state0, segment_id=segment_id)
      # Compute grads
      loss = -tf.math.log(
          tf.sigmoid((tf.reduce_sum(tf.math.square(act)) +
                      tf.reduce_sum(state.m * state.c * state.c))))
      grads = tf.gradients(loss, base_frnn.vars.Flatten())

      expected_act, expected_state, expected_grads = self.evaluate(
          [act, state, grads])

      act, state = frnn.FPropDefaultTheta(
          inputs, padding, state0=state0, segment_id=segment_id)
      # Compute grads
      loss = -tf.math.log(
          tf.sigmoid((tf.reduce_sum(tf.math.square(act)) +
                      tf.reduce_sum(state.m * state.c * state.c))))
      grads = tf.gradients(loss, frnn.vars.Flatten())

      actual_act, actual_state, actual_grads = self.evaluate(
          [act, state, grads])

    tf.logging.info('expected_act:{}'.format(expected_act))
    tf.logging.info('actual_act:{}'.format(actual_act))

    tf.logging.info('expected_state:{}'.format(expected_state))
    tf.logging.info('actual_state:{}'.format(actual_state))

    tf.logging.info('expected_grads:{}'.format(expected_grads))
    tf.logging.info('actual_grads:{}'.format(actual_grads))

    self.assertAllClose(expected_act, actual_act)
    self.assertAllClose(expected_state.m, actual_state.m)
    self.assertAllClose(expected_state.c, actual_state.c)
    for (vname, _), expected, actual in zip(frnn.vars.FlattenItems(),
                                            expected_grads, actual_grads):
      self.assertAllClose(expected, actual, msg=vname)

  @parameterized.named_parameters(
      ('HasBias', True),
      ('NoBias', False),
      ('HasBiasPackedInputs', True, True),
      ('NoBiasPackedInputs', False, True),
  )
  def testLSTMCellSimple(self, enable_lstm_bias, packed_inputs=False):
    base_cell_p = rnn_cell.LSTMCellSimple.Params()
    base_cell_p.enable_lstm_bias = enable_lstm_bias
    self._SetCellParams(base_cell_p)
    base_frnn_p = rnn_layers.FRNN.Params().Set(
        name='base_frnn', cell=base_cell_p)

    cell_p = lstm_frnn_layer.LSTMCellSimpleExt.Params()
    cell_p.enable_lstm_bias = enable_lstm_bias
    self._SetCellParams(cell_p)
    frnn_p = lstm_frnn_layer.LstmFRNN.Params().Set(name='frnn', cell=cell_p)
    self._testHelper(base_frnn_p, frnn_p, packed_inputs)

  def testLayerNormalizedLSTMCellSimple(self):
    base_cell_p = rnn_cell.LayerNormalizedLSTMCellSimple.Params()
    base_cell_p.enable_lstm_bias = True
    self._SetCellParams(base_cell_p)
    base_frnn_p = rnn_layers.FRNN.Params().Set(
        name='base_frnn', cell=base_cell_p)

    cell_p = lstm_frnn_layer.LayerNormalizedLSTMCellSimpleExt.Params()
    cell_p.enable_lstm_bias = True
    self._SetCellParams(cell_p)
    frnn_p = lstm_frnn_layer.LstmFRNN.Params().Set(name='frnn', cell=cell_p)
    self._testHelper(base_frnn_p, frnn_p)

  def testLayerNormalizedLSTMCellLean(self):
    base_cell_p = rnn_cell.LayerNormalizedLSTMCellLean.Params()
    base_cell_p.enable_lstm_bias = True
    self._SetCellParams(base_cell_p)
    base_frnn_p = rnn_layers.FRNN.Params().Set(
        name='base_frnn', cell=base_cell_p)

    cell_p = lstm_frnn_layer.LayerNormalizedLSTMCellLeanExt.Params()
    cell_p.enable_lstm_bias = True
    self._SetCellParams(cell_p)
    frnn_p = lstm_frnn_layer.LstmFRNN.Params().Set(name='frnn', cell=cell_p)
    self._testHelper(base_frnn_p, frnn_p)


if __name__ == '__main__':
  tf.test.main()
