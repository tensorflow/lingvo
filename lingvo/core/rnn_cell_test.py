# Lint as: python3
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
"""Tests for rnn_cell."""

from absl.testing import parameterized
import lingvo.compat as tf
from lingvo.core import py_utils
from lingvo.core import quant_utils
from lingvo.core import rnn_cell
from lingvo.core import test_utils
import numpy as np

_INIT_RANDOM_SEED = 429891685
_NUMPY_RANDOM_SEED = 12345
_RANDOM_SEED = 98274


class RNNCellTest(test_utils.TestCase, parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('_NoInlineNoGruBias', False, False,
       [[-1.085402, 2.161964], [-0.933972, 1.995606], [-0.892969, 2.059967]],
       [[0.500292, 0.732436], [0.34267, 0.732542], [0.341799, 0.815305]]),
      ('_NoInlineGruBias', False, True,
       [[-1.206088, 2.558667], [-1.024555, 2.359131], [-1.006608, 2.385566]],
       [[0.726844, 0.932083], [0.537847, 0.932127], [0.536803, 0.967041]]),
      ('_InlineNoGruBias', True, False,
       [[-1.085402, 2.161964], [-0.933972, 1.995606], [-0.892969, 2.059967]],
       [[0.500292, 0.732436], [0.34267, 0.732542], [0.341799, 0.815305]]),
      ('_InlineGruBias', True, True,
       [[-1.206088, 2.558667], [-1.024555, 2.359131], [-1.006608, 2.385566]],
       [[0.726844, 0.932083], [0.537847, 0.932127], [0.536803, 0.967041]]))
  # pyformat: enable
  def testGRUCell(self, inline, enable_gru_bias, m_expected, c_expected):
    params = rnn_cell.GRUCell.Params().Set(
        name='gru_rnn',
        params_init=py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        bias_init=py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        num_input_nodes=2,
        num_output_nodes=2,
        num_hidden_nodes=2,
        enable_gru_bias=enable_gru_bias)
    params.vn.global_vn = False  # do not set variational noise
    params.vn.per_step_vn = False  # do not set step wise noise

    gru = rnn_cell.GRUCell(params)

    tf.logging.info('gru vars = %s', gru.vars)
    self.assertIn('w_r', gru.vars.w_r.name)
    self.assertIn('w_u', gru.vars.w_u.name)
    self.assertIn('w_n', gru.vars.w_n.name)

    if enable_gru_bias:
      self.assertIn('b_n', gru.vars.b_n.name)
      self.assertIn('b_r', gru.vars.b_r.name)
      self.assertIn('b_u', gru.vars.b_u.name)

    self.assertEqual(
        gru.theta.w_n.get_shape(),
        tf.TensorShape([
            params.num_input_nodes + params.num_output_nodes,
            params.num_hidden_nodes,
        ]))
    self.assertEqual(
        gru.theta.w_u.get_shape(),
        tf.TensorShape([
            params.num_input_nodes + params.num_output_nodes,
            params.num_hidden_nodes,
        ]))
    self.assertEqual(
        gru.theta.w_r.get_shape(),
        tf.TensorShape([
            params.num_input_nodes + params.num_output_nodes,
            params.num_output_nodes,
        ]))

    if enable_gru_bias:
      self.assertEqual(gru.theta.b_n.get_shape(),
                       tf.TensorShape([params.num_hidden_nodes]))
      self.assertEqual(gru.theta.b_u.get_shape(),
                       tf.TensorShape([params.num_hidden_nodes]))
      self.assertEqual(gru.theta.b_r.get_shape(),
                       tf.TensorShape([params.num_output_nodes]))

    # Start feeding in inputs.
    np.random.seed(_NUMPY_RANDOM_SEED)
    inputs = py_utils.NestedMap(
        act=[tf.constant(np.random.uniform(size=(3, 2)), tf.float32)],
        padding=tf.zeros([3, 1]))
    c_values = tf.constant(np.random.uniform(size=(3, 2)), tf.float32)
    m_values = tf.constant(np.random.uniform(size=(3, 2)), tf.float32)
    state0 = py_utils.NestedMap(c=c_values, m=m_values)
    state1, _ = gru.FPropDefaultTheta(state0, inputs)

    with self.session(
        use_gpu=False, config=py_utils.SessionConfig(inline=inline)):
      self.evaluate(tf.global_variables_initializer())

      variable_count = 11 if enable_gru_bias else 8
      wts = tf.get_collection('GRUCell_vars')
      self.assertLen(wts, variable_count)

      self.assertAllClose(m_expected, state1.m.eval())
      self.assertAllClose(c_expected, state1.c.eval())

  # pyformat: disable
  @parameterized.named_parameters(
      ('_NoInlineNoCIFGNoLSTMBias', False, False, False,
       [[0.095727, 0.476658], [0.04662, 0.180589], [0.001656, 0.374141]],
       [[0.241993, 0.820267], [0.086863, 0.349722], [0.003176, 0.655448]]),
      ('_NoInlineNoCIFGLSTMBias', False, False, True,
       [[0.007753, 0.66843], [-0.029904, 0.485617], [-0.026663, 0.654127]],
       [[0.033096, 1.013467], [-0.086807, 0.748031], [-0.08087, 1.04254]]),
      ('_NoInlineCIFGNoLSTMBias', False, True, False,
       [[0.22088, 0.244225], [0.123647, 0.25378], [0.163328, 0.214796]],
       [[0.355682, 0.711696], [0.313728, 0.633475], [0.485248, 0.961122]]),
      ('_NoInlineCIFGLSTMBias', False, True, True,
       [[0.342635, 0.182102], [0.140832, 0.210234], [0.224034, 0.155077]],
       [[0.499417, 0.701774], [0.278458, 0.697437], [0.51618, 0.964456]]),
      ('_InlineNoCIFGNoLSTMBias', True, False, False,
       [[0.095727, 0.476658], [0.04662, 0.180589], [0.001656, 0.374141]],
       [[0.241993, 0.820267], [0.086863, 0.349722], [0.003176, 0.655448]]),
      ('_InlineNoCIFGLSTMBias', True, False, True,
       [[0.007753, 0.66843], [-0.029904, 0.485617], [-0.026663, 0.654127]],
       [[0.033096, 1.013467], [-0.086807, 0.748031], [-0.08087, 1.04254]]),
      ('_InlineCIFGNoLSTMBias', True, True, False,
       [[0.22088, 0.244225], [0.123647, 0.25378], [0.163328, 0.214796]],
       [[0.355682, 0.711696], [0.313728, 0.633475], [0.485248, 0.961122]]),
      ('_InlineCIFGLSTMBias', True, True, True,
       [[0.342635, 0.182102], [0.140832, 0.210234], [0.224034, 0.155077]],
       [[0.499417, 0.701774], [0.278458, 0.697437], [0.51618, 0.964456]]))
  # pyformat: enable
  def testLSTMSimple_P1(self, inline, couple_input_forget_gates,
                        enable_lstm_bias, m_expected, c_expected):
    params = rnn_cell.LSTMCellSimple.Params().Set(
        name='lstm',
        params_init=py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        bias_init=py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        num_input_nodes=2,
        num_output_nodes=2,
        couple_input_forget_gates=couple_input_forget_gates,
        enable_lstm_bias=enable_lstm_bias)
    params.vn.global_vn = False
    params.vn.per_step_vn = False

    lstm = rnn_cell.LSTMCellSimple(params)
    tf.logging.info('lstm vars = %s', lstm.vars)
    self.assertIn('wm', lstm.vars.wm.name)

    if enable_lstm_bias:
      self.assertIn('b', lstm.vars.b.name)

    num_param_vectors = 6 if couple_input_forget_gates else 8
    self.assertEqual(lstm.theta.wm.get_shape(),
                     tf.TensorShape([4, num_param_vectors]))

    if enable_lstm_bias:
      self.assertEqual(lstm.theta.b.get_shape(),
                       tf.TensorShape([num_param_vectors]))

    np.random.seed(_NUMPY_RANDOM_SEED)
    inputs = py_utils.NestedMap(
        act=[tf.constant(np.random.uniform(size=(3, 2)), tf.float32)],
        padding=tf.zeros([3, 1]))
    state0 = py_utils.NestedMap(
        c=tf.constant(np.random.uniform(size=(3, 2)), tf.float32),
        m=tf.constant(np.random.uniform(size=(3, 2)), tf.float32))

    state1, _ = lstm.FPropDefaultTheta(state0, inputs)

    with self.session(
        use_gpu=False, config=py_utils.SessionConfig(inline=inline)):
      self.evaluate(tf.global_variables_initializer())

      variable_count = 2 if enable_lstm_bias else 1
      wts = tf.get_collection('LSTMCellSimple_vars')
      self.assertLen(wts, variable_count)

      # pyformat: disable
      # xmw_expected = [
      #     [-0.74310219, 1.10182762, 0.67478961, 0.62169313, 0.77394271,
      #      -0.1691505, -0.39185536, 0.87572402],
      #     [-0.78952235, 0.04464795, 0.00245538, -0.34931657, 0.22463873,
      #      0.02745318, 0.15253648, 0.14931624],
      #     [-1.58246589, 0.03950393, 0.18513964, -0.25745165, 0.73317981,
      #      0.68082684, 0.08576801, 0.62040436]]
      # pyformat: enable
      self.assertAllClose(m_expected, state1.m.eval())
      self.assertAllClose(c_expected, state1.c.eval())

  # pyformat: disable
  @parameterized.named_parameters(
      ('_Masked', 0, 2, True, False,
       [[0.095727, 0.476658], [0.04662, 0.180589], [0.001656, 0.374141]],
       [[0.241993, 0.820267], [0.086863, 0.349722], [0.003176, 0.655448]]),
      ('_MaskedProjections', 2, 1, True, True,
       [[0.414049], [0.076521], [0.356313]],
       [[0.270425, 0.840373], [0.349856, 0.440421], [0.261243, 0.889804]]),
      ('_Projections', 2, 1, False, False, [[0.414049], [0.076521], [0.356313]],
       [[0.270425, 0.840373], [0.349856, 0.440421], [0.261243, 0.889804]]))
  # pyformat: enable
  def testLSTMSimple_P2(self, num_hidden_nodes, num_output_nodes, apply_pruning,
                        apply_pruning_to_projection, m_expected, c_expected):
    params = rnn_cell.LSTMCellSimple.Params().Set(
        name='lstm',
        params_init=py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        num_input_nodes=2,
        num_hidden_nodes=num_hidden_nodes,
        num_output_nodes=num_output_nodes,
        apply_pruning=apply_pruning,
        apply_pruning_to_projection=apply_pruning_to_projection)
    params.vn.global_vn = False
    params.vn.per_step_vn = False
    lstm = rnn_cell.LSTMCellSimple(params)

    tf.logging.info('lstm vars = %s', lstm.vars)
    self.assertIn('wm', lstm.vars.wm.name)
    self.assertIn('b', lstm.vars.b.name)
    if apply_pruning:
      self.assertIn('mask', lstm.vars.mask.name)
      self.assertIn('threshold', lstm.vars.threshold.name)
    if apply_pruning_to_projection:
      self.assertIn('w_proj', lstm.vars.w_proj.name)
      self.assertIn('proj_mask', lstm.vars.proj_mask.name)
      self.assertIn('proj_threshold', lstm.vars.proj_threshold.name)

    num_io = params.num_input_nodes + num_output_nodes
    num_param_vectors = 8
    self.assertEqual(lstm.theta.wm.get_shape(),
                     tf.TensorShape([num_io, num_param_vectors]))
    self.assertEqual(lstm.theta.b.get_shape(),
                     tf.TensorShape([num_param_vectors]))
    if apply_pruning_to_projection:
      self.assertEqual(lstm.theta.w_proj.get_shape(), tf.TensorShape([2, 1]))
      self.assertEqual(lstm.theta.proj_mask.get_shape(), tf.TensorShape([2, 1]))

    np.random.seed(_NUMPY_RANDOM_SEED)
    inputs = py_utils.NestedMap(
        act=[tf.constant(np.random.uniform(size=(3, 2)), tf.float32)],
        padding=tf.zeros([3, 1]))
    state0 = py_utils.NestedMap(
        c=tf.constant(np.random.uniform(size=(3, 2)), tf.float32),
        m=tf.constant(
            np.random.uniform(size=(3, num_output_nodes)), tf.float32))

    state1, _ = lstm.FPropDefaultTheta(state0, inputs)

    with self.session(
        use_gpu=False, config=py_utils.SessionConfig(inline=False)):
      self.evaluate(tf.global_variables_initializer())

      if num_hidden_nodes > 0:
        variable_count = 3  # weights, biases, projection.
      else:
        variable_count = 2  # weights, biases.
      wts = tf.get_collection('LSTMCellSimple_vars')
      self.assertLen(wts, variable_count)

      if apply_pruning:
        num_vars = 2 if apply_pruning_to_projection else 1
        masks = tf.get_collection('masks')
        self.assertLen(masks, num_vars)

        threshold = tf.get_collection('thresholds')
        self.assertLen(threshold, num_vars)

      self.assertAllClose(m_expected, state1.m.eval())
      self.assertAllClose(c_expected, state1.c.eval())

  @parameterized.named_parameters(
      dict(
          testcase_name='_GlobalNoise',
          m_expected=[[-0.061369, 0.198229], [0.015448, 0.077887],
                      [-0.043196, 0.063092]],
          c_expected=[[-0.292653, 0.573982], [0.06449, 0.194121],
                      [-0.259611, 0.211775]]),
      dict(
          testcase_name='_Double',
          dtype=tf.float64,
          m_expected=[[0.209838, 0.010304], [0.071149, 0.024825],
                      [-0.051708, -0.02569]],
          c_expected=[[0.496024, 0.120541], [0.239338, 0.061809],
                      [-0.157122, -0.142631]]),
      dict(
          testcase_name='_NoOutputNonlinearity',
          global_vn=False,
          dtype=tf.float64,
          output_nonlinearity=False,
          m_expected=[[0.532625, 0.083511], [0.118662, 0.110532],
                      [0.121542, 0.084161]],
          c_expected=[[0.789908, 0.312811], [0.192642, 0.207369],
                      [0.167591, 0.172713]]),
      dict(
          testcase_name='_ByPass',
          global_vn=False,
          dtype=tf.float64,
          output_nonlinearity=False,
          bypass=True,
          m_expected=None,
          c_expected=None),
      dict(
          testcase_name='_WithForgetGateBias',
          dtype=tf.float64,
          forget_gate_bias=-1.,
          m_expected=[[1.501816e-01, -1.257807e-04],
                      [1.708532e-02, -1.134138e-02],
                      [-1.087235e-01, -5.264970e-02]],
          c_expected=[[0.341142, -0.001464], [0.05646, -0.028209],
                      [-0.340229, -0.298947]]),
  )
  def testLSTMSimple_P3(self,
                        global_vn=True,
                        couple_input_forget_gates=False,
                        dtype=tf.float32,
                        output_nonlinearity=True,
                        bypass=False,
                        num_output_nodes=2,
                        forget_gate_bias=0,
                        m_expected=None,
                        c_expected=None):
    params = rnn_cell.LSTMCellSimple.Params().Set(
        name='lstm',
        couple_input_forget_gates=couple_input_forget_gates,
        output_nonlinearity=output_nonlinearity,
        params_init=py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        num_input_nodes=2,
        num_output_nodes=num_output_nodes,
        dtype=dtype,
        forget_gate_bias=forget_gate_bias)
    params.vn.seed = 8820495
    params.vn.global_vn = global_vn
    params.vn.per_step_vn = False
    params.vn.scale = 0.5

    lstm = rnn_cell.LSTMCellSimple(params)

    np.random.seed(_NUMPY_RANDOM_SEED)
    padding = tf.ones([3, 1]) if bypass else tf.zeros([3, 1])
    inputs = py_utils.NestedMap(
        act=[tf.constant(np.random.uniform(size=(3, 2)), dtype)],
        padding=padding)
    state0 = py_utils.NestedMap(
        c=tf.constant(np.random.uniform(size=(3, 2)), dtype),
        m=tf.constant(np.random.uniform(size=(3, 2)), dtype))

    state1, _ = lstm.FPropDefaultTheta(state0, inputs)

    with self.session(use_gpu=False):
      self.evaluate(tf.global_variables_initializer())

      wts = tf.get_collection('LSTMCellSimple_vars')
      self.assertLen(wts, 2)

      if bypass:
        m_expected = state0.m.eval()
        c_expected = state0.c.eval()
      self.assertAllClose(m_expected, state1.m.eval())
      self.assertAllClose(c_expected, state1.c.eval())

  # pyformat: disable
  @parameterized.named_parameters(
      ('', False,
       [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.1, 2.1, 2.1, 0.1, 0.1, 0.1]),
      ('_CoupledInputForget', True,
       [0.1, 0.1, 0.1, 2.1, 2.1, 2.1, 0.1, 0.1, 0.1]))
  # pyformat: enable
  def testLSTMSimpleWithForgetGateInitBias(self, couple_input_forget_gates,
                                           b_expected):
    params = rnn_cell.LSTMCellSimple.Params().Set(
        name='lstm',
        params_init=py_utils.WeightInit.Constant(0.1),
        couple_input_forget_gates=couple_input_forget_gates,
        num_input_nodes=2,
        num_output_nodes=3,
        forget_gate_bias=2.0,
        bias_init=py_utils.WeightInit.Constant(0.1),
        dtype=tf.float64)

    lstm = rnn_cell.LSTMCellSimple(params)

    np.random.seed(_NUMPY_RANDOM_SEED)
    with self.session(use_gpu=False):
      self.evaluate(tf.global_variables_initializer())
      b_value = lstm._GetBias(lstm.theta).eval()
      tf.logging.info('testLSTMSimpleWithForgetGateInitBias b = %s',
                      np.array_repr(b_value))
      self.assertAllClose(b_value, b_expected)

  # pyformat: disable
  @parameterized.named_parameters(
      ('_NoLSTMBias', False,
       [[0.09375, 0.460938], [0.046875, 0.179688], [0.039062, 0.375]],
       [[0.234375, 0.789062], [0.09375, 0.351562], [0.078125, 0.664062]]),
      ('_LSTMBias', True,
       [[0.039062, 0.476562], [0.007812, 0.445312], [0.015625, 0.546875]],
       [[0.148438, 0.78125], [0.023438, 0.710938], [0.039062, 0.984375]]))
  # pyformat: enable
  def testQuantizedLSTMSimple(self, enable_lstm_bias, m_expected, c_expected):
    params = rnn_cell.LSTMCellSimple.Params().Set(
        name='lstm',
        params_init=py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        bias_init=py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        num_input_nodes=2,
        num_output_nodes=2,
        couple_input_forget_gates=False,
        enable_lstm_bias=enable_lstm_bias)
    params.vn.global_vn = False
    params.vn.per_step_vn = False

    cc_schedule = quant_utils.FakeQuantizationSchedule.Params().Set(
        clip_start_step=0,  # Step 0 is unclipped.
        clip_end_step=0,
        quant_start_step=0,
        start_cap=1.0,
        end_cap=1.0)
    params.qdomain.default = quant_utils.SymmetricScheduledClipQDomain.Params(
    ).Set(cc_schedule=cc_schedule.Copy())
    params.qdomain.c_state = quant_utils.SymmetricScheduledClipQDomain.Params(
    ).Set(cc_schedule=cc_schedule.Copy())
    params.qdomain.m_state = quant_utils.SymmetricScheduledClipQDomain.Params(
    ).Set(cc_schedule=cc_schedule.Copy())
    params.qdomain.fullyconnected = (
        quant_utils.SymmetricScheduledClipQDomain.Params().Set(
            cc_schedule=cc_schedule.Copy()))

    params.cell_value_cap = None

    lstm = rnn_cell.LSTMCellSimple(params)

    tf.logging.info('lstm vars = %s', lstm.vars)
    self.assertIn('wm', lstm.vars.wm.name)

    if enable_lstm_bias:
      self.assertIn('b', lstm.vars.b.name)

    num_param_vectors = 8
    self.assertEqual(lstm.theta.wm.get_shape(),
                     tf.TensorShape([4, num_param_vectors]))

    if enable_lstm_bias:
      self.assertEqual(lstm.theta.b.get_shape(),
                       tf.TensorShape([num_param_vectors]))

    np.random.seed(_NUMPY_RANDOM_SEED)
    inputs = py_utils.NestedMap(
        act=[tf.constant(np.random.uniform(size=(3, 2)), tf.float32)],
        padding=tf.zeros([3, 1]))
    state0 = py_utils.NestedMap(
        c=tf.constant(np.random.uniform(size=(3, 2)), tf.float32),
        m=tf.constant(np.random.uniform(size=(3, 2)), tf.float32))

    state1, _ = lstm.FPropDefaultTheta(state0, inputs)

    with self.session(
        use_gpu=False, config=py_utils.SessionConfig(inline=False)):
      self.evaluate(tf.global_variables_initializer())

      variable_count = 2 if enable_lstm_bias else 1
      wts = tf.get_collection('LSTMCellSimple_vars')
      self.assertLen(wts, variable_count)

      self.assertAllClose(m_expected, state1.m.eval())
      self.assertAllClose(c_expected, state1.c.eval())

  @parameterized.named_parameters(('_WithoutOutputShuffling', 1),
                                  ('_WithOutputShuffling', 2))
  def testLSTMCellGrouped(self, num_shuffle_shards):
    params = rnn_cell.LSTMCellGrouped.Params().Set(
        name='lstm',
        num_input_nodes=8,
        num_output_nodes=8,
        num_groups=4,
        num_shuffle_shards=num_shuffle_shards)
    child_p = params.child_lstm_tpl
    child_p.output_nonlinearity = True
    child_p.params_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)
    child_p.vn.global_vn = False
    child_p.vn.per_step_vn = False

    lstm = params.Instantiate()

    tf.logging.info('lstm vars = %s', lstm.vars)
    for child_lstm in lstm.groups:
      self.assertIn('wm', child_lstm.vars.wm.name)
      self.assertIn('b', child_lstm.vars.b.name)

      self.assertEqual(child_lstm.theta.wm.get_shape(), tf.TensorShape([4, 8]))
      self.assertEqual(child_lstm.theta.b.get_shape(), tf.TensorShape([8]))

    np.random.seed(_NUMPY_RANDOM_SEED)
    inputs = py_utils.NestedMap(
        act=[tf.constant(np.random.uniform(size=(3, 8)), tf.float32)],
        padding=tf.zeros([3, 1]))
    state0 = py_utils.NestedMap(
        groups=py_utils.SplitRecursively(
            py_utils.NestedMap(
                c=tf.constant(np.random.uniform(size=(3, 8)), tf.float32),
                m=tf.constant(np.random.uniform(
                    size=(3, 8)), tf.float32)), params.num_groups))

    state1, _ = lstm.FPropDefaultTheta(state0, inputs)
    self.assertLen(state1.groups, params.num_groups)
    out1 = lstm.GetOutput(state1)

    with self.session(
        use_gpu=False, config=py_utils.SessionConfig(inline=False)):
      self.evaluate(tf.global_variables_initializer())

      variable_count = 2 * params.num_groups  # one for weights, one for biases.
      wts = tf.get_collection('LSTMCellSimple_vars')
      self.assertLen(wts, variable_count)

      state1 = py_utils.ConcatRecursively(state1.groups)
      m_actual = state1.m.eval()
      c_actual = state1.c.eval()
      out_actual = out1.eval()

    tf.logging.info('m_actual = %s', np.array_repr(m_actual))
    tf.logging.info('c_actual = %s', np.array_repr(c_actual))
    tf.logging.info('out_actual = %s', np.array_repr(out_actual))

    # pyformat: disable
    # pylint: disable=bad-whitespace,bad-continuation
    m_expected = [
        [
            -0.07857136,  0.43932292,  0.11373602,  0.16337454,
             0.01618987,  0.09685542, -0.20168062,  0.52612996,
        ],
        [
             0.07929622,  0.18910739, -0.11084013,  0.32307294,
             0.03500029, -0.05823045,  0.16963124,  0.27039385,
        ],
        [
             0.11623365,  0.38104215,  0.00935007,  0.22124135,
            -0.17368057,  0.10859803, -0.06948104,  0.10925373,
        ],
    ]
    c_expected = [
        [
            -0.23670214,  0.66260374,  0.24650344,  0.28946888,
             0.03051668,  0.15143034, -0.52736223,  0.88325077,
        ],
        [
             0.16262427,  0.28568456, -0.19542629,  0.52116692,
             0.06872599, -0.1123996,   0.31477568,  0.49881396,
        ],
        [
             0.19667494,  0.68746102,  0.02078706,  0.30816019,
            -0.36376655,  0.16003416, -0.16141629,  0.16648693,
        ],
    ]
    # pylint: enable=bad-whitespace,bad-continuation
    # pyformat: enable
    out_expected = m_expected
    if num_shuffle_shards > 1:

      def _ShuffleShards(x):
        return [[row[i] for i in (0, 3, 2, 5, 4, 7, 6, 1)] for row in x]

      assert num_shuffle_shards == 2
      out_expected = _ShuffleShards(out_expected)
    self.assertAllClose(m_expected, m_actual)
    self.assertAllClose(c_expected, c_actual)
    self.assertAllClose(out_expected, out_actual)

  def testLSTMCellGroupedNoInputSplit(self):
    params = rnn_cell.LSTMCellGrouped.Params().Set(
        name='lstm',
        num_input_nodes=8,
        num_output_nodes=8,
        num_hidden_nodes=16,
        num_groups=4,
        num_shuffle_shards=1,
        split_inputs=False)
    child_p = params.child_lstm_tpl
    child_p.output_nonlinearity = True
    child_p.params_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)
    child_p.vn.global_vn = False
    child_p.vn.per_step_vn = False

    lstm = params.Instantiate()

    tf.logging.info('lstm vars = %s', lstm.vars)
    for child_lstm in lstm.groups:
      self.assertIn('wm', child_lstm.vars.wm.name)
      self.assertIn('b', child_lstm.vars.b.name)
      self.assertIn('w_proj', child_lstm.vars.w_proj.name)

      # 10 = 8 layer inputs + 2 recurrent
      # 16 = 4 gates * 4 hidden units/group
      self.assertEqual(child_lstm.theta.wm.get_shape(), tf.TensorShape([10,
                                                                        16]))
      self.assertEqual(child_lstm.theta.b.get_shape(), tf.TensorShape([16]))
      # Projection from 4 hidden units (16 total / 4 groups) to 2 outputs
      # (8 total / 4 groups)
      self.assertEqual(child_lstm.theta.w_proj.get_shape(),
                       tf.TensorShape([4, 2]))

    np.random.seed(_NUMPY_RANDOM_SEED)
    inputs = py_utils.NestedMap(
        act=[tf.constant(np.random.uniform(size=(3, 8)), tf.float32)],
        padding=tf.zeros([3, 1]))
    state0 = py_utils.NestedMap(
        groups=py_utils.SplitRecursively(
            py_utils.NestedMap(
                c=tf.constant(np.random.uniform(size=(3, 16)), tf.float32),
                m=tf.constant(np.random.uniform(
                    size=(3, 8)), tf.float32)), params.num_groups))

    state1, _ = lstm.FPropDefaultTheta(state0, inputs)
    self.assertLen(state1.groups, params.num_groups)
    out1 = lstm.GetOutput(state1)

    with self.session(
        use_gpu=False, config=py_utils.SessionConfig(inline=False)):
      self.evaluate(tf.global_variables_initializer())

      variable_count = 3 * params.num_groups  # [wm, b, w_proj] for each group.
      wts = tf.get_collection('LSTMCellSimple_vars')
      self.assertLen(wts, variable_count)

      state1 = py_utils.ConcatRecursively(state1.groups)
      m_actual = state1.m.eval()
      c_actual = state1.c.eval()
      out_actual = out1.eval()

    tf.logging.info('m_actual = %s', np.array_repr(m_actual))
    tf.logging.info('c_actual = %s', np.array_repr(c_actual))
    tf.logging.info('out_actual = %s', np.array_repr(out_actual))

    # pyformat: disable
    # pylint: disable=bad-whitespace,bad-continuation
    m_expected = [
        [
             0.61734521,  0.02338588,  0.19424279,  0.31576008,
             0.18000039,  0.1672723,   0.44075012, -0.06824636,
        ],
        [
             0.44694018, -0.01717547,  0.49302083, -0.27330822,
             0.35382932, -0.1967615,   0.44225505, -0.04489155,
        ],
        [
             0.66018867,  0.09434807,  0.643556,    0.0383133,
             0.74754262, -0.01860991,  0.48671043,  0.29460859,
        ],
    ]
    c_expected = [
        [
            -0.52246463,  0.67389512,  0.58692968,  0.75484836,
            -0.21763092,  0.45671225, -0.33593893,  1.03087521,
            -0.15525842,  0.31072262,  0.14663902,  0.64976436,
            -0.40176213,  0.36785093,  0.52653724,  0.73124039,
        ],
        [
            -0.27722716,  0.90508962,  0.39852297,  0.01676523,
            -0.7724061,   0.40351537,  0.20194794,  0.08798298,
            -0.39136624,  0.26601788,  0.21635406, -0.05538163,
            -0.36326468,  0.64099556,  0.25886536, -0.09711652,
        ],
        [
            -0.63169837,  0.99831283,  0.53726614,  0.77321815,
            -0.67881596,  1.01512539,  0.38799196,  0.26393941,
            -0.87696433,  1.29881907,  0.60203284,  0.42675141,
            -0.24902672,  1.15422893,  0.70180357,  0.12213309,
        ],
    ]
    out_expected = [
        [
             0.61734521,  0.02338588,  0.19424279,  0.31576008,
             0.18000039,  0.1672723,   0.44075012, -0.06824636,
        ],
        [
             0.44694018, -0.01717547,  0.49302083, -0.27330822,
             0.35382932, -0.1967615,   0.44225505, -0.04489155,
        ],
        [
             0.66018867,  0.09434807,  0.643556,    0.0383133,
             0.74754262, -0.01860991,  0.48671043,  0.29460859,
        ],
    ]
    # pylint: enable=bad-whitespace,bad-continuation
    # pyformat: enable
    self.assertAllClose(m_expected, m_actual)
    self.assertAllClose(c_expected, c_actual)
    self.assertAllClose(out_expected, out_actual)

  @parameterized.named_parameters(
      ('_NoInline', False, False, [
          0.4144063, 0.88831079, 0.56665027, 0.30154669, 0.2818037
      ], [4.72228432, 3.9454143, 3.77556086, 2.76972866, 1.87397099]),
      ('_Inline', True, False, [
          0.4144063, 0.88831079, 0.56665027, 0.30154669, 0.2818037
      ], [4.72228432, 3.9454143, 3.77556086, 2.76972866, 1.87397099]),
      ('_GlobalNoise', True, True, [
          1.891705, 1.759723, 0.981507, 2.172119, 1.073552
      ], [8.521435, 7.25943, 5.508372, 7.510872, 5.006944]))
  def testConvLSTM(self, inline, global_vn, m_expected, c_expected):
    params = rnn_cell.ConvLSTMCell.Params().Set(
        name='conv_lstm',
        params_init=py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        inputs_shape=[None, 4, 2, 3],
        cell_shape=[None, 4, 2, 2],
        filter_shape=[3, 2])
    params.vn.seed = 8820495
    params.vn.scale = 0.5
    params.vn.global_vn = global_vn
    params.vn.per_step_vn = False

    lstm = rnn_cell.ConvLSTMCell(params)
    lstm_vars = lstm.vars
    tf.logging.info('lstm vars = %s', lstm_vars)
    self.assertIn('wm', lstm_vars.wm.name)
    self.assertIn('b', lstm_vars.b.name)

    w = lstm.theta.wm
    b = lstm.theta.b

    self.assertEqual(w.get_shape(), tf.TensorShape([3, 2, 5, 8]))
    self.assertEqual(b.get_shape(), tf.TensorShape([8]))

    np.random.seed(_NUMPY_RANDOM_SEED)
    inputs = py_utils.NestedMap(
        act=[tf.constant(np.random.uniform(size=(5, 4, 2, 3)), tf.float32)],
        padding=tf.zeros([5, 1]))
    state0 = py_utils.NestedMap(
        c=tf.constant(np.random.uniform(size=(5, 4, 2, 2)), tf.float32),
        m=tf.constant(np.random.uniform(size=(5, 4, 2, 2)), tf.float32))
    state1, _ = lstm.FPropDefaultTheta(state0, inputs)
    m1 = tf.reduce_sum(state1.m, [1, 2, 3])
    c1 = tf.reduce_sum(state1.c, [1, 2, 3])

    with self.session(
        use_gpu=False, config=py_utils.SessionConfig(inline=inline)):
      self.evaluate(tf.global_variables_initializer())

      wts = tf.get_collection('ConvLSTMCell_vars')
      self.assertLen(wts, 2)

      self.assertAllClose(m_expected, m1.eval())
      self.assertAllClose(c_expected, c1.eval())

  @parameterized.named_parameters(
      ('_Disabled', 0., False, [[0.2, 0.], [0., 2.4], [0.2, 3.4]], False),
      ('_Enabled', 0.5, True, [[0.2, 0.], [0., 0.4], [0.2, 0.5]], False),
      ('_Eval', 0.5, False, [[0.2, 0.], [0.05, 1.4], [0.15, 1.95]], True))
  def testZoneOut(self, zo_prob, enable_random_uniform, v_expected, is_eval):
    params = rnn_cell.LSTMCellSimple.Params().Set(
        name='lstm',
        params_init=py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        num_input_nodes=2,
        num_output_nodes=2)
    lstm = rnn_cell.LSTMCellSimple(params)
    if enable_random_uniform:
      random_uniform = tf.random.uniform([3, 2], seed=98798202)
    else:
      random_uniform = None
    prev_v = [[0.2, 0.0], [0.1, 0.4], [0.1, 0.5]]
    cur_v = [[0.3, 1.0], [0.0, 2.4], [0.2, 3.4]]
    padding_v = [[1.0], [0.0], [0.0]]
    new_v = lstm._ZoneOut(
        prev_v,
        cur_v,
        padding_v,
        zo_prob=zo_prob,
        is_eval=is_eval,
        random_uniform=random_uniform)

    with self.session(use_gpu=False):
      # In eval mode, if padding[i] == 1, new_v equals prev_v.
      # Otherwise, new_v = zo_prob * prev_v + (1.0 - zo_prob) * cur_v
      new_v_evaled = new_v.eval()
      tf.logging.info('new_v_evaled = %s', np.array_repr(new_v_evaled))
      self.assertAllClose(v_expected, new_v_evaled)

  # pyformat: disable
  @parameterized.named_parameters(
      ('_LSTMSimple', rnn_cell.LSTMCellSimple, True, False,
       [[0.0083883, 0.10644437], [0.04662009, 0.18058866],
        [0.0016561, 0.37414068]],
       [[0.96451449, 0.65317708], [0.08686253, 0.34972212],
        [0.00317609, 0.6554482]]),
      ('_LSTMSimpleDeterministic', rnn_cell.LSTMCellSimple, False, True,
       [[-0.145889, 0.], [-0.008282, 0.073219], [-0.041057, 0.]],
       [[0., 0.532332], [-0.016117, 0.13752], [0., 0.]]))
  # pyformat: enable
  def testCellWithZoneOut(self, cell_cls, manual_state, deterministic,
                          m_expected, c_expected):
    params = cell_cls.Params().Set(
        name='lstm',
        params_init=py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        num_input_nodes=2,
        num_output_nodes=2,
        zo_prob=0.5,
        random_seed=_RANDOM_SEED)
    params.vn.global_vn = False
    params.vn.per_step_vn = False
    params.deterministic = deterministic
    lstm = cell_cls(params)

    np.random.seed(_NUMPY_RANDOM_SEED)
    inputs = py_utils.NestedMap(
        act=[tf.constant(np.random.uniform(size=(3, 2)), tf.float32)],
        padding=tf.zeros([3, 1]))
    if manual_state:
      state0 = py_utils.NestedMap(
          c=tf.constant(np.random.uniform(size=(3, 2)), tf.float32),
          m=tf.constant(np.random.uniform(size=(3, 2)), tf.float32))
    else:
      state0 = lstm.zero_state(lstm.theta, 3)
    state1, _ = lstm.FPropDefaultTheta(state0, inputs)

    with self.session(use_gpu=False):
      self.evaluate(tf.global_variables_initializer())

      m_v = state1.m.eval()
      c_v = state1.c.eval()

    tf.logging.info('m_v = %s', np.array_repr(m_v))
    tf.logging.info('c_v = %s', np.array_repr(c_v))
    self.assertAllClose(m_expected, m_v)
    self.assertAllClose(c_expected, c_v)

  # pyformat: disable
  @parameterized.named_parameters(
      ('LSTMCell', rnn_cell.LayerNormalizedLSTMCell, 2, 2, None, None,
       [[0.03960676, 0.26547235], [-0.00677715, 0.09782403],
        [-0.00272907, 0.31641623]],
       [[0.14834785, 0.3804915], [-0.00927538, 0.38059634],
        [-0.01014781, 0.46336061]]),
      ('LSTMCellSimple', rnn_cell.LayerNormalizedLSTMCellSimple, 2, 2, None,
       True, [[0.03960676, 0.26547235], [-0.00677715, 0.09782403],
              [-0.00272907, 0.31641623]],
       [[0.14834785, 0.3804915], [-0.00927538, 0.38059634],
        [-0.01014781, 0.46336061]]),
      ('WeightNormWithoutProj', rnn_cell.WeightNormalizedLSTMCellSimple,
       2, 5, None, True,
       [[0.06124492, 0.22344325, 0.2573673, 0.22021532, 0.2515218],
        [0.0851617, 0.10842345, -0.01816726, 0.16266146, 0.09313971],
        [0.03512726, 0.28154817, 0.32190764, 0.198605, 0.03860135]],
       [[0.27504668, 0.5392196, 0.86178875, 0.788363, 0.42013264],
        [0.46231857, 0.19085361, -0.03375426, 0.5479378, 0.14932579],
        [0.16894935, 0.63103956, 0.7552417, 0.6724432, 0.0733701]]),
      ('NormLSTMCellSimple', rnn_cell.NormalizedLSTMCellSimple, 2, 2, None,
       False, [[0.03960676, 0.26547235], [-0.00677715, 0.09782403],
               [-0.00272907, 0.31641623]],
       [[0.14834785, 0.3804915], [-0.00927538, 0.38059634],
        [-0.01014781, 0.46336061]]),
      ('LSTMCellLean', rnn_cell.LayerNormalizedLSTMCellLean, 2, 2, None, False,
       [[-0.20482419, 0.55676991], [-0.55648255, 0.20511301],
        [-0.20482422, 0.55676997]],
       [[0.14834785, 0.3804915], [-0.00927544, 0.38059637],
        [-0.01014781, 0.46336061]]),
      ('LSTMCellProj', rnn_cell.LayerNormalizedLSTMCellSimple, 2, 2, 4, True,
       [[0.39790073, 0.28511256], [0.41482946, 0.28972796],
        [0.47132283, 0.03284446]],
       [[-0.3667627, 1.03294277, 0.24229962, 0.43976486],
        [-0.15832338, 1.22740746, 0.19910297, -0.14970526],
        [-0.57552528, 0.9139322, 0.41805002, 0.58792269]]),
      ('WeightNormWithProj', rnn_cell.WeightNormalizedLSTMCellSimple,
       2, 2, 5, True,
       [[0.09886207, 0.2904314], [-0.1020302, 0.28588364],
        [0.18647942, 0.22519538]],
       [[0.31294015, 0.42476785, 0.831409, 0.9407059, 0.2278131],
        [0.37135834, 0.09326379, 0.0806338, 0.28109533, 0.16940905],
        [0.07096109, 0.49200284, 0.5689974, 0.8055533, 0.15250428]]),
      ('NormLSTMCellProj', rnn_cell.NormalizedLSTMCellSimple, 2, 2, 4, False,
       [[0.39790073, 0.28511256], [0.41482946, 0.28972796],
        [0.47132283, 0.03284446]],
       [[-0.3667627, 1.03294277, 0.24229962, 0.43976486],
        [-0.15832338, 1.22740746, 0.19910297, -0.14970526],
        [-0.57552528, 0.9139322, 0.41805002, 0.58792269]]),
      ('LSTMCellLeanProj', rnn_cell.LayerNormalizedLSTMCellLean, 2, 2, 4,
       False, [[0.51581347, 0.22646663], [0.56025136, 0.16842051],
               [0.58704823, -0.07126484]],
       [[-0.36676273, 1.03294277, 0.24229959, 0.43976486],
        [-0.15832338, 1.22740746, 0.19910295, -0.14970522],
        [-0.57552516, 0.9139322, 0.41805002, 0.58792269]]))
  # pyformat: enable
  def testNormalization(self, cell_cls, num_input_nodes, num_output_nodes,
                        num_hidden_nodes, enable_lstm_bias, m_expected,
                        c_expected):
    tf.logging.info('cell_cls is %s', cell_cls)
    cell_params = cell_cls.Params()
    if enable_lstm_bias is not None:
      cell_params.Set(enable_lstm_bias=enable_lstm_bias)
    m_v, c_v = self._testLNLSTMCell(cell_params, num_input_nodes,
                                    num_output_nodes, num_hidden_nodes)
    self.assertAllClose(m_expected, m_v)
    self.assertAllClose(c_expected, c_v)

  def testLNLSTMCellLeanNoLnOnC(self):
    """LayerNormalizedLSTMCellLean without normalization on 'c'."""
    m_v, c_v = self._testLNLSTMCell(
        rnn_cell.LayerNormalizedLSTMCellLean.Params().Set(enable_ln_on_c=False))
    m_expected = [[0.039607, 0.265472], [-0.006777, 0.097824],
                  [-0.002729, 0.316416]]
    c_expected = [[0.14834785, 0.3804915], [-0.00927544, 0.38059637],
                  [-0.01014781, 0.46336061]]
    self.assertAllClose(m_expected, m_v)
    self.assertAllClose(c_expected, c_v)

  @parameterized.named_parameters(('Enable', True), ('Disable', False))
  def testLNLSTMCellLeanLSTMBias(self, enable):
    m_expected, c_expected = self._testLNLSTMCell(
        rnn_cell.LayerNormalizedLSTMCellSimple.Params().Set(
            cell_value_cap=None,
            enable_lstm_bias=enable,
            bias_init=py_utils.WeightInit.Constant(1.0)))
    m, c = self._testLNLSTMCell(
        rnn_cell.LayerNormalizedLSTMCellLean.Params().Set(
            enable_ln_on_c=False,
            enable_lstm_bias=enable,
            bias_init=py_utils.WeightInit.Constant(1.0)))
    self.assertAllClose(m_expected, m)
    self.assertAllClose(c_expected, c)

  @parameterized.named_parameters(('HighThreshold', 0.5),
                                  ('LowThreshold', 5e-4))
  def testLNLSTMCellLeanCellValueCap(self, cell_value_cap):
    m_expected, c_expected = self._testLNLSTMCell(
        rnn_cell.LayerNormalizedLSTMCellSimple.Params().Set(
            enable_lstm_bias=False, cell_value_cap=cell_value_cap))
    m, c = self._testLNLSTMCell(
        rnn_cell.LayerNormalizedLSTMCellLean.Params().Set(
            enable_ln_on_c=False,
            enable_lstm_bias=False,
            cell_value_cap=cell_value_cap))
    self.assertAllClose(m_expected, m)
    self.assertAllClose(c_expected, c)

  def testLNLSTMCellLeanFeatureParity(self):
    """Tests feature parity with LayerNormalizedLSTMCellSimple ...

    under the same configuration.
    """
    m_expected, c_expected, grads_expected = self._testLNLSTMCellFPropBProp(
        rnn_cell.LayerNormalizedLSTMCellSimple.Params().Set(
            enable_lstm_bias=True, cell_value_cap=5e-4))
    m, c, grads = self._testLNLSTMCellFPropBProp(
        rnn_cell.LayerNormalizedLSTMCellLean.Params().Set(
            enable_lstm_bias=True,
            cell_value_cap=5e-4,
            enable_ln_on_c=False,
            use_ln_bias=False))
    self.assertAllClose(m_expected, m)
    self.assertAllClose(c_expected, c)
    tf.logging.info('grads_expected: %r', grads_expected)
    tf.logging.info('grads_actual: %r', grads)
    self.assertAllClose(grads_expected.wm, grads.wm)
    self.assertAllClose(grads_expected.b, grads.b)
    self.assertAllClose(
        grads_expected.ln_scale,
        np.concatenate([
            grads.ln_scale_i_i, grads.ln_scale_i_g, grads.ln_scale_f_g,
            grads.ln_scale_o_g
        ]))

  def _testLNLSTMCellHelper(self, params, num_input_nodes, num_output_nodes,
                            num_hidden_nodes):
    params = params.Copy().Set(
        name='lstm',
        params_init=py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        num_input_nodes=num_input_nodes,
        num_output_nodes=num_output_nodes,
        random_seed=_RANDOM_SEED)
    if num_hidden_nodes is not None:
      params.num_hidden_nodes = num_hidden_nodes
    params.vn.global_vn = False
    params.vn.per_step_vn = False
    lstm = params.Instantiate()
    np.random.seed(_NUMPY_RANDOM_SEED)
    inputs = py_utils.NestedMap(
        act=[
            tf.constant(
                np.random.uniform(size=(3, num_input_nodes)), tf.float32)
        ],
        padding=tf.zeros([3, 1]))
    state0 = py_utils.NestedMap(
        c=tf.constant(
            np.random.uniform(size=(3, lstm.hidden_size)), tf.float32),
        m=tf.constant(
            np.random.uniform(size=(3, lstm.output_size)), tf.float32))
    state1, _ = lstm.FPropDefaultTheta(state0, inputs)
    return lstm, state0, state1

  def _testLNLSTMCell(self,
                      params,
                      num_input_nodes=2,
                      num_output_nodes=2,
                      num_hidden_nodes=None):
    tf.reset_default_graph()
    _, _, state1 = self._testLNLSTMCellHelper(params, num_input_nodes,
                                              num_output_nodes,
                                              num_hidden_nodes)
    with self.session(use_gpu=False):
      self.evaluate(tf.global_variables_initializer())
      m_v = state1.m.eval()
      c_v = state1.c.eval()
    tf.logging.info('m_v = %s', np.array_repr(m_v))
    tf.logging.info('c_v = %s', np.array_repr(c_v))
    return m_v, c_v

  # pyformat: disable
  @parameterized.named_parameters(
      ('WithoutProj', rnn_cell.LSTMCellSimple, 2, 5, None,
       [[0.01737128, 0.19599544, 0.17018753, 0.1767421, 0.3115133],
        [0.02149553, 0.13240226, -0.04374956, 0.1305859, 0.081632],
        [0.00524007, 0.28841242, 0.3509299, 0.1629353, -0.04303718]],
       [[0.24370661, 0.522195, 1.0963748, 1.1247414, 0.46262175],
        [0.45510566, 0.20793101, -0.07393111, 0.74867463, 0.11499637],
        [0.08915544, 0.64467084, 0.8427882, 0.9257332, -0.07887113]]),
      ('WithProj', rnn_cell.LSTMCellSimple, 2, 2, 5,
       [[0.12637874, 0.18615241], [-0.09705047, 0.26344097],
        [0.22582228, 0.16268292]],
       [[0.25855693, 0.41018295, 0.85405004, 1.0882103, 0.16526523],
        [0.34494585, 0.13268812, 0.07282493, 0.34063944, 0.08520323],
        [0.00365126, 0.4749492, 0.55997956, 0.95868456, 0.12583911]]))
  # pyformat: enable
  def testLSTMCell(self, cell_cls, num_input_nodes, num_output_nodes,
                   num_hidden_nodes, m_expected, c_expected):
    params = cell_cls.Params()
    m_v, c_v = self._testLNLSTMCell(params, num_input_nodes, num_output_nodes,
                                    num_hidden_nodes)
    self.assertAllClose(m_expected, m_v)
    self.assertAllClose(c_expected, c_v)

  def _testLNLSTMCellFPropBProp(self, params, num_hidden_nodes=None):
    tf.reset_default_graph()
    lstm, _, state1 = self._testLNLSTMCellHelper(params, 2, 2, num_hidden_nodes)
    loss = -tf.math.log(
        tf.sigmoid(
            tf.reduce_sum(tf.square(state1.m)) +
            tf.reduce_sum(state1.m * state1.c * state1.c)))
    grads = tf.gradients(loss, lstm.vars.Flatten())

    with self.session(use_gpu=False):
      self.evaluate(tf.global_variables_initializer())
      m_v, c_v, grads_v = self.evaluate([state1.m, state1.c, grads])

    tf.logging.info('m_v = %s', np.array_repr(m_v))
    tf.logging.info('c_v = %s', np.array_repr(c_v))
    grads_val = py_utils.NestedMap()
    for (n, _), val in zip(lstm.vars.FlattenItems(), grads_v):
      tf.logging.info('%s : %s', n, np.array_repr(val))
      grads_val[n] = val
    return m_v, c_v, grads_val

  # pyformat: disable
  @parameterized.named_parameters(
      ('_NoLnOnC', False, [[-0.606178], [0.599713], [0.657852]],
       [[1.261887, -0.029158], [-0.00341, 1.034558], [-0.003731, 1.259534]]),
      ('_LnOnC', True, [[-0.751002], [0.784634], [0.784634]],
       [[1.261887, -0.029158], [-0.00341, 1.034558], [-0.003731, 1.259534]]))
  # pyformat: enable
  def testDoubleProjectionLSTMCell(self, enable_ln_on_c, m_expected,
                                   c_expected):
    params = rnn_cell.DoubleProjectionLSTMCell.Params().Set(
        name='lstm',
        params_init=py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        enable_ln_on_c=enable_ln_on_c,
        num_input_nodes=2,
        num_output_nodes=1,
        num_hidden_nodes=2,
        num_input_hidden_nodes=2)
    params.vn.global_vn = False
    params.vn.per_step_vn = False

    lstm = params.Instantiate()
    tf.logging.info('lstm vars = %s', lstm.vars)

    # Input projection.
    self.assertEqual(
        lstm.theta.w_input_proj.get_shape(),
        tf.TensorShape([
            params.num_input_nodes + params.num_output_nodes,
            params.num_input_hidden_nodes,
        ]))
    # W, LN, bias for the gates.
    for gate in ['i_i', 'i_g', 'f_g', 'o_g']:
      self.assertEqual(
          lstm.theta.get('wm_%s' % gate).get_shape(),
          tf.TensorShape(
              [params.num_input_hidden_nodes, params.num_hidden_nodes]))
      self.assertEqual(
          lstm.theta.get('ln_scale_%s' % gate).get_shape(),
          tf.TensorShape([params.num_hidden_nodes]))
      self.assertEqual(
          lstm.theta.get('bias_%s' % gate).get_shape(),
          tf.TensorShape([params.num_hidden_nodes]))
    # LN and bias for 'c'.
    if enable_ln_on_c:
      self.assertEqual(lstm.theta.ln_scale_c.get_shape(),
                       tf.TensorShape([params.num_hidden_nodes]))
      self.assertEqual(lstm.theta.bias_c.get_shape(),
                       tf.TensorShape([params.num_hidden_nodes]))
    else:
      self.assertNotIn('ln_scale_c', lstm.theta)
      self.assertNotIn('bias_c', lstm.theta)
    # Output projection.
    self.assertEqual(
        lstm.theta.w_output_proj.get_shape(),
        tf.TensorShape([params.num_hidden_nodes, params.num_output_nodes]))

    np.random.seed(_NUMPY_RANDOM_SEED)
    inputs = py_utils.NestedMap(
        act=[
            tf.constant(
                np.random.uniform(size=(3, params.num_input_nodes)),
                tf.float32),
        ],
        padding=tf.zeros([3, 1]))
    state0 = py_utils.NestedMap(
        c=tf.constant(
            np.random.uniform(size=(3, params.num_hidden_nodes)), tf.float32),
        m=tf.constant(
            np.random.uniform(size=(3, params.num_output_nodes)), tf.float32))

    state1, _ = lstm.FPropDefaultTheta(state0, inputs)

    with self.session(
        use_gpu=False, config=py_utils.SessionConfig(inline=False)):
      self.evaluate(tf.global_variables_initializer())

      wts = tf.get_collection('DoubleProjectionLSTMCell_vars')
      if enable_ln_on_c:
        self.assertLen(wts, 2 + 3 * 4 + 2)
      else:
        self.assertLen(wts, 2 + 3 * 4)

      self.assertAllClose(m_expected, state1.m.eval())
      self.assertAllClose(c_expected, state1.c.eval())

  # pyformat: disable
  @parameterized.named_parameters(
      ('', tf.zeros,
       lambda: tf.constant(np.random.uniform(size=(3, 2)), tf.float32),
       [[0.097589, 0.579055], [0.046737, 0.187892], [0.001656, 0.426245]],
       [[0.241993, 0.820267], [0.086863, 0.349722], [0.003176, 0.655448]]),
      ('_Padding', tf.ones, lambda: tf.zeros([3, 2], tf.float32),
       [[0., 0.], [0., 0.], [0., 0.]], [[0., 0.], [0., 0.], [0., 0.]]))
  # pyformat: enable
  def testQuantizedLSTMCell(self, padding_fn, state0_fn, m_expected,
                            c_expected):
    params = rnn_cell.QuantizedLSTMCell.Params().Set(
        name='lstm',
        params_init=py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        num_input_nodes=2,
        num_output_nodes=2,
        cc_schedule=quant_utils.LinearClippingCapSchedule.Params().Set(
            start_step=0, end_step=2, start_cap=5.0, end_cap=1.0))
    params.vn.global_vn = False
    params.vn.per_step_vn = False

    lstm = rnn_cell.QuantizedLSTMCell(params)
    lstm_vars = lstm.vars
    tf.logging.info('lstm vars = %s', lstm_vars)
    self.assertIn('wm', lstm_vars.wm.name)

    wm = lstm.theta.wm
    self.assertEqual(wm.get_shape(), tf.TensorShape([4, 8]))

    np.random.seed(_NUMPY_RANDOM_SEED)
    inputs = py_utils.NestedMap(
        act=[tf.constant(np.random.uniform(size=(3, 2)), tf.float32)],
        padding=padding_fn([3, 1]))
    state0 = py_utils.NestedMap(c=state0_fn(), m=state0_fn())
    state1, _ = lstm.FPropDefaultTheta(state0, inputs)

    with self.session(use_gpu=False):
      self.evaluate(tf.global_variables_initializer())

      self.assertAllClose(m_expected, state1.m.eval())
      self.assertAllClose(c_expected, state1.c.eval())
      self.assertEqual(5.0,
                       lstm.cc_schedule.GetState(lstm.theta.cc_schedule).eval())
      self.evaluate(tf.assign(py_utils.GetOrCreateGlobalStepVar(), 1))
      self.assertEqual(3.0,
                       lstm.cc_schedule.GetState(lstm.theta.cc_schedule).eval())

  def testQuantizedLayerNormalizedLSTMCell(self):
    params = rnn_cell.LayerNormalizedLSTMCell.Params().Set(
        name='lstm',
        params_init=py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        num_input_nodes=2,
        num_output_nodes=2,
        random_seed=_RANDOM_SEED,
        cc_schedule=quant_utils.LinearClippingCapSchedule.Params().Set(
            start_step=0, end_step=2, start_cap=5.0, end_cap=1.0))
    params.vn.global_vn = False
    params.vn.per_step_vn = False

    lstm = rnn_cell.LayerNormalizedLSTMCell(params)
    lstm_vars = lstm.vars
    tf.logging.info('lstm vars = %s', lstm_vars)
    self.assertIn('wm', lstm_vars.wm.name)

    wm = lstm.theta.wm
    self.assertEqual(wm.get_shape(), tf.TensorShape([4, 8]))

    np.random.seed(_NUMPY_RANDOM_SEED)
    inputs = py_utils.NestedMap(
        act=[tf.constant(np.random.uniform(size=(3, 2)), tf.float32)],
        padding=tf.zeros([3, 1]))
    state0 = py_utils.NestedMap(
        c=tf.constant(np.random.uniform(size=(3, 2)), tf.float32),
        m=tf.constant(np.random.uniform(size=(3, 2)), tf.float32))
    state1, _ = lstm.FPropDefaultTheta(state0, inputs)

    with self.session(use_gpu=False):
      self.evaluate(tf.global_variables_initializer())
      m_expected = [[0.03960676, 0.26547235], [-0.00677715, 0.09782403],
                    [-0.00272907, 0.31641623]]
      c_expected = [[0.14834785, 0.3804915], [-0.00927538, 0.38059634],
                    [-0.01014781, 0.46336061]]
      self.assertAllClose(m_expected, state1.m.eval())
      self.assertAllClose(c_expected, state1.c.eval())

      self.assertEqual(5.0,
                       lstm.cc_schedule.GetState(lstm.theta.cc_schedule).eval())
      self.evaluate(tf.assign(py_utils.GetOrCreateGlobalStepVar(), 1))
      self.assertEqual(3.0,
                       lstm.cc_schedule.GetState(lstm.theta.cc_schedule).eval())

  # pyformat: disable
  @parameterized.named_parameters(
      ('_TrainingUnclipped', False, False, 0, None,
       [[0.097589, 0.579055], [0.046737, 0.187892], [0.001656, 0.426245]],
       [[0.241993, 0.820267], [0.086863, 0.349722], [0.003176, 0.655448]]),
      ('_Training', False, True, 0, [[0.0], [0.0], [1.0]],
       [[0.09375, 0.5625], [0.046875, 0.1875], [0.809813, 0.872176]],
       [[0.23288, 0.806], [0.090057, 0.355591], [0.747715, 0.961307]]),
      ('_HiddenNodes', False, True, 4, None,
       [[0.382812, 0.296875], [0.164062, 0.171875], [0.3125, -0.039062]],
       [[-0.160339, 0.795929, 0.449707, 0.347534],
        [-0.049194, 0.548279, -0.060852, -0.106354],
        [-0.464172, 0.345947, 0.407349, 0.430878]]),
      ('_Inference', True, False, 0, None,
       [[0.09375, 0.5625], [0.046875, 0.1875], [0., 0.429688]],
       [[0.23288, 0.806], [0.090057, 0.355591], [-0.003937, 0.662567]]))
  # pyformat: enable
  def testLSTMCellSimpleQuantized(self, is_inference, set_training_step,
                                  num_hidden_nodes, padding, m_expected,
                                  c_expected):
    params = rnn_cell.LSTMCellSimple.Params().Set(
        name='lstm',
        is_inference=is_inference,
        params_init=py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        num_input_nodes=2,
        num_output_nodes=2,
        num_hidden_nodes=num_hidden_nodes,
        output_nonlinearity=False,
        cell_value_cap=None,
        enable_lstm_bias=False)
    params.vn.global_vn = False
    params.vn.per_step_vn = False

    cc_schedule = quant_utils.FakeQuantizationSchedule.Params().Set(
        clip_start_step=1,  # Step 0 is unclipped.
        clip_end_step=2,
        quant_start_step=2,
        start_cap=5.0,
        end_cap=1.0)
    # Default quantization.
    qdomain = quant_utils.SymmetricScheduledClipQDomain.Params().Set(
        cc_schedule=cc_schedule)
    params.qdomain.default = qdomain
    # M state uses the default 8-bit quantziation.
    cc_schedule = cc_schedule.Copy()
    qdomain = quant_utils.SymmetricScheduledClipQDomain.Params().Set(
        cc_schedule=cc_schedule)
    params.qdomain.m_state = qdomain
    # C state uses 16 bit quantization.
    cc_schedule = cc_schedule.Copy().Set(bits=16)
    qdomain = quant_utils.SymmetricScheduledClipQDomain.Params().Set(
        cc_schedule=cc_schedule)
    params.qdomain.c_state = qdomain
    # Fully connected layer clips slightly differently.
    cc_schedule = cc_schedule.Copy().Set(start_cap=64.0, end_cap=8.0)
    qdomain = quant_utils.SymmetricScheduledClipQDomain.Params().Set(
        cc_schedule=cc_schedule)
    params.qdomain.fullyconnected = qdomain

    lstm = rnn_cell.LSTMCellSimple(params)
    lstm_vars = lstm.vars
    tf.logging.info('lstm vars = %s', lstm_vars)
    self.assertIn('wm', lstm_vars.wm.name)
    if num_hidden_nodes:
      self.assertIn('w_proj', lstm_vars.w_proj.name)
    else:
      self.assertNotIn('w_proj', lstm_vars)

    np.random.seed(_NUMPY_RANDOM_SEED)
    if padding is None:
      padding = tf.zeros([3, 1])
    else:
      padding = tf.constant(padding, dtype=tf.float32)
    inputs = py_utils.NestedMap(
        act=[tf.constant(np.random.uniform(size=(3, 2)), tf.float32)],
        padding=padding)
    state0 = py_utils.NestedMap(
        c=tf.constant(
            np.random.uniform(size=(3, lstm.hidden_size)), tf.float32),
        m=tf.constant(
            np.random.uniform(size=(3, lstm.output_size)), tf.float32))
    state1, _ = lstm.FPropDefaultTheta(state0, inputs)

    with self.session(use_gpu=False), self.SetEval(is_inference):
      self.evaluate(tf.global_variables_initializer())
      if set_training_step:
        # Get it into the fully clipped/quantized part of the schedule.
        self.evaluate(tf.assign(py_utils.GetOrCreateGlobalStepVar(), 5))

      # Outputs.
      self.assertAllClose(m_expected, state1.m.eval())
      self.assertAllClose(c_expected, state1.c.eval())

      # Cell reported zeros.
      cell_zero_state = lstm.zero_state(lstm.theta, batch_size=3)
      self.assertAllEqual(cell_zero_state.m.eval(),
                          tf.zeros_like(state0.m).eval())
      self.assertAllEqual(cell_zero_state.c.eval(),
                          tf.zeros_like(state0.c).eval())

  # pyformat: disable
  @parameterized.named_parameters(
      ('_Regular', None, True, 0, False, False, False, False,
       [[0.58657289, 0.70520258], [0.32375532, 0.29133356],
        [0.58900255, 0.58398587]],
       [[0.45297623, 0.88433027], [0.42112729, 0.47023624],
        [0.50483131, 0.89583319]]),
      ('_WithCellCap', 0.4, True, 0, False, False, False, False,
       [[0.569023, 0.472208], [0.314541, 0.260022], [0.543097, 0.379774]],
       [[0.4, 0.4], [0.4, 0.4], [0.4, 0.4]]),
      ('_WithInputGates', None, False, 0, False, False, False, False,
       [[0.18558595, 0.81267989], [0.30404502, 0.35851872],
        [0.51972485, 0.77677751]],
       [[-0.19633068, 0.78370988], [0.29389063, 0.39952573],
        [0.1221304, 0.67298377]]),
      ('_WithProjection', None, True, 3, False, False, False, False,
       [[0.04926362, 0.54914111], [-0.0501487, 0.32742232],
        [-0.19329719, 0.28332305]],
       [[-0.101138, 0.8266117, 0.75368524],
        [0.31730127, 0.58325875, 0.64149243],
        [0.09471729, 0.48504758, 0.53909004]]),
      ('_WithNonZeroBiasInit', None, True, 3, True, False, False, False,
       [[-0.012749, 0.788564], [-0.215866, 0.891125], [-0.420648, 0.713777]],
       [[-0.30021, 0.728136, 0.751739], [0.211836, 0.665222, 0.755629],
        [0.112793, 0.281477, 0.44823]]),
      ('_WithLayerNormalization', None, True, 0, False, True, False, False,
       [[0.301535, 0.744214], [0.38422, -0.524064], [0.328908, 0.658833]],
       [[-1., 1.], [0.999999, -0.999999], [-1., 1.]]),
      ('_Masked', None, True, 0, False, False, True, False,
       [[0.586573, 0.705203], [0.323755, 0.291334], [0.589002, 0.583986]],
       [[0.452976, 0.88433], [0.421127, 0.470236], [0.504831, 0.895833]]),
      ('_MaskedProjection', None, True, 3, False, False, True, True,
       [[0.049264, 0.549141], [-0.050149, 0.327422], [-0.193297, 0.283323]],
       [[-0.10113806, 0.8266117, 0.75368524],
        [0.31730127, 0.58325875, 0.64149243],
        [0.0947173, 0.48504758, 0.53909004]]))
  # pyformat: enable
  def testSRUCell(self, cell_value_cap, couple_input_forget_gates,
                  num_hidden_nodes, enable_bias_init, apply_layer_norm,
                  apply_pruning, apply_pruning_to_projection, m_expected,
                  c_expected):
    params = rnn_cell.SRUCell.Params().Set(
        name='sru',
        couple_input_forget_gates=couple_input_forget_gates,
        params_init=py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        num_input_nodes=2,
        num_hidden_nodes=num_hidden_nodes,
        num_output_nodes=2,
        random_seed=_RANDOM_SEED,
        apply_layer_norm=apply_layer_norm,
        apply_pruning=apply_pruning,
        apply_pruning_to_projection=apply_pruning_to_projection)
    if cell_value_cap is not None:
      params.cell_value_cap = 0.4  # cell cap set to low level
    if enable_bias_init:
      params.bias_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)
    sru = rnn_cell.SRUCell(params)

    self.assertIn('wm', sru.vars.wm.name)
    self.assertIn('b', sru.vars.b.name)
    if apply_pruning:
      self.assertIn('mask', sru.vars.mask.name)
      self.assertIn('threshold', sru.vars.threshold.name)
    if params.apply_pruning_to_projection:
      self.assertIn('w_proj', sru.vars.w_proj.name)
      self.assertIn('proj_mask', sru.vars.proj_mask.name)
      self.assertIn('proj_threshold', sru.vars.proj_threshold.name)

    num_gates = 4 if couple_input_forget_gates else 5
    if num_hidden_nodes > 0:
      num_nodes = num_hidden_nodes
    else:
      num_nodes = params.num_output_nodes
    num_param_vectors = num_gates * num_nodes
    self.assertEqual(sru.theta.wm.get_shape(),
                     tf.TensorShape([2, num_param_vectors]))
    self.assertEqual(sru.theta.b.get_shape(),
                     tf.TensorShape([num_param_vectors]))
    if params.apply_pruning_to_projection:
      self.assertEqual(sru.theta.w_proj.get_shape(), tf.TensorShape([3, 2]))
      self.assertEqual(sru.theta.proj_mask.get_shape(), tf.TensorShape([3, 2]))

    np.random.seed(_NUMPY_RANDOM_SEED)
    inputs = py_utils.NestedMap(
        act=[tf.constant(np.random.uniform(size=(3, 2)), tf.float32)],
        padding=tf.zeros([3, 1]))
    if params.num_hidden_nodes > 0:
      c_dim = params.num_hidden_nodes
    else:
      c_dim = params.num_output_nodes
    state0 = py_utils.NestedMap(
        c=tf.constant(np.random.uniform(size=(3, c_dim)), tf.float32),
        m=tf.constant(np.random.uniform(size=(3, 2)), tf.float32))
    state1, _ = sru.FPropDefaultTheta(state0, inputs)

    with self.session(use_gpu=False):
      self.evaluate(tf.global_variables_initializer())
      m_v = state1.m.eval()
      c_v = state1.c.eval()

    tf.logging.info('m_v = %s', np.array_repr(m_v))
    tf.logging.info('c_v = %s', np.array_repr(c_v))
    self.assertAllClose(m_expected, m_v)
    self.assertAllClose(c_expected, c_v)

  # pyformat: disable
  @parameterized.named_parameters(
      ('', 'quasi_ifo',
       [[0.40564698, 0.32611847], [0.16975531, 0.45970476],
        [0.60286027, 0.24542703]],
       [[0.42057115, 0.49928033], [0.56830668, 0.7003305],
        [1.28926766, 0.75380397]]),
      ('_InSRUMode', 'sru',
       [[0.55884922, 0.40396619], [0.71426249, 0.70820922],
        [0.823457, 0.60304904]],
       [[0.65144706, 0.56252223], [0.75096267, 0.65605044],
        [0.79761195, 0.36905324]]))
  # pyformat: enable
  def testQRNNPoolingCell(self, pooling_formula, m_expected, c_expected):
    params = rnn_cell.QRNNPoolingCell.Params().Set(
        name='QuasiRNN',
        params_init=py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        num_input_nodes=2,
        num_output_nodes=2,
        zo_prob=0.0,
        random_seed=_RANDOM_SEED,
        pooling_formula=pooling_formula)
    qrnn = rnn_cell.QRNNPoolingCell(params)

    np.random.seed(_NUMPY_RANDOM_SEED)
    num_rnn_matrices = 4
    inputs = py_utils.NestedMap(
        act=[
            tf.constant(
                np.random.uniform(
                    size=(3, params.num_input_nodes * num_rnn_matrices)),
                tf.float32)
        ],
        padding=tf.zeros([3, 1]))
    state0 = py_utils.NestedMap(
        c=tf.constant(
            np.random.uniform(size=(3, params.num_output_nodes)), tf.float32),
        m=tf.constant(
            np.random.uniform(size=(3, params.num_output_nodes)), tf.float32))
    state1, _ = qrnn.FPropDefaultTheta(state0, inputs)

    with self.session(use_gpu=False):
      self.evaluate(tf.global_variables_initializer())
      m_v = state1.m.eval()
      c_v = state1.c.eval()

    tf.logging.info('m_v = %s', np.array_repr(m_v))
    tf.logging.info('c_v = %s', np.array_repr(c_v))
    self.assertAllClose(m_expected, m_v)
    self.assertAllClose(c_expected, c_v)

  # pyformat: disable
  @parameterized.named_parameters(
      ('_FnZeros', False, False, [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]),
      ('_FnRandomNormal', True, False, [[-0.630551, -1.208959, -0.348799]],
       [[-0.630551, -1.208959, -0.348799]]),
      ('_FnRandomNormalInEval', True, True, [[0.0, 0.0, 0.0]],
       [[0.0, 0.0, 0.0]]))
  # pyformat: enable
  def testLSTMZeroState(self, random_zero_state, is_eval, m_expected,
                        c_expected):
    if random_zero_state:
      zero_state_init = py_utils.RNNCellStateInit.RandomNormal(seed=12345)
    else:
      zero_state_init = py_utils.RNNCellStateInit.Zeros()
    params = rnn_cell.LSTMCellSimple.Params().Set(
        name='lstm',
        params_init=py_utils.WeightInit.Constant(0.1),
        num_input_nodes=2,
        num_output_nodes=3,
        forget_gate_bias=2.0,
        bias_init=py_utils.WeightInit.Constant(0.1),
        dtype=tf.float64,
        zero_state_init_params=zero_state_init)
    lstm = rnn_cell.LSTMCellSimple(params)

    with self.session(use_gpu=False), self.SetEval(is_eval):
      self.evaluate(tf.global_variables_initializer())
      init_state_value = self.evaluate(lstm.zero_state(lstm.theta, 1))

    tf.logging.info('testLSTMSimpleWithStateInitializationFn m = %s',
                    np.array_repr(init_state_value['m']))
    tf.logging.info('testLSTMSimpleWithStateInitializationFn c = %s',
                    np.array_repr(init_state_value['c']))
    self.assertAllClose(init_state_value['m'], m_expected)
    self.assertAllClose(init_state_value['c'], c_expected)


if __name__ == '__main__':
  tf.test.main()
