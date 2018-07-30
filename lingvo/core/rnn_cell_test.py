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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops

from lingvo.core import cudnn_rnn_utils
from lingvo.core import py_utils
from lingvo.core import quant_utils
from lingvo.core import rnn_cell

UNI_RNN = cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION
BI_RNN = cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION

_INIT_RANDOM_SEED = 429891685
_NUMPY_RANDOM_SEED = 12345
_RANDOM_SEED = 98274


class RNNCellTest(tf.test.TestCase):

  def _testLSTMSimpleHelper(self,
                            inline=False,
                            trainable_zero_state=False,
                            couple_input_forget_gates=False,
                            apply_pruning=False):
    with self.test_session(
        use_gpu=False, config=py_utils.SessionConfig(inline=inline)):
      params = rnn_cell.LSTMCellSimple.Params()
      params.name = 'lstm'
      params.output_nonlinearity = True
      params.params_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)

      params.vn.global_vn = False
      params.vn.per_step_vn = False
      params.num_input_nodes = 2
      params.num_output_nodes = 2
      params.trainable_zero_state = trainable_zero_state
      params.couple_input_forget_gates = couple_input_forget_gates

      lstm = rnn_cell.LSTMCellSimple(params)

      print('lstm vars = ', lstm.vars)
      self.assertTrue('wm' in lstm.vars.wm.name)
      self.assertTrue('b' in lstm.vars.b.name)

      num_param_vectors = 6 if couple_input_forget_gates else 8
      self.assertEqual(lstm.theta.wm.get_shape(),
                       tf.TensorShape([4, num_param_vectors]))
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

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()

      if trainable_zero_state:
        variable_count = 4  # weights, biases, initial states for c and h
      else:
        variable_count = 2
      wts = tf.get_collection('LSTMCellSimple_vars')
      self.assertEqual(variable_count, len(wts))

      # pyformat: disable
      if couple_input_forget_gates:
        m_expected = [
            [0.22088, 0.244225],
            [0.123647, 0.25378],
            [0.163328, 0.214796]]
        c_expected = [
            [0.355682, 0.711696],
            [0.313728, 0.633475],
            [0.485248, 0.961122]]
      else:
        m_expected = [
            [0.095727, 0.476658],
            [0.04662, 0.180589],
            [0.001656, 0.374141]]
        c_expected = [
            [0.241993, 0.820267],
            [0.086863, 0.349722],
            [0.003176, 0.655448]]
      xmw_expected = [
          [-0.74310219, 1.10182762, 0.67478961, 0.62169313, 0.77394271,
           -0.1691505, -0.39185536, 0.87572402],
          [-0.78952235, 0.04464795, 0.00245538, -0.34931657, 0.22463873,
           0.02745318, 0.15253648, 0.14931624],
          [-1.58246589, 0.03950393, 0.18513964, -0.25745165, 0.73317981,
           0.68082684, 0.08576801, 0.62040436]]
      # pyformat: enable
      self.assertAllClose(m_expected, state1.m.eval())
      self.assertAllClose(c_expected, state1.c.eval())
      # print('xmw = ', extras.xmw.eval())
      # self.assertAllClose(xmw_expected, extras.xmw.eval())

  def testLSTMSimple_NoInline(self):
    self._testLSTMSimpleHelper(inline=False)

  def testLSTMSimple_Inline(self):
    self._testLSTMSimpleHelper(inline=True)

  def testLSTMSimple_TrainableZeroState(self):
    self._testLSTMSimpleHelper(inline=False, trainable_zero_state=True)

  def testCifgLSTMSimple_NoInline(self):
    self._testLSTMSimpleHelper(inline=False, couple_input_forget_gates=True)

  def testCifgLSTMSimple_Inline(self):
    self._testLSTMSimpleHelper(inline=True, couple_input_forget_gates=True)

  def testCifgLSTMSimple_TrainableZeroState(self):
    self._testLSTMSimpleHelper(
        inline=False, trainable_zero_state=True, couple_input_forget_gates=True)

  def testLSTMSimple_Masked(self):
    with self.test_session(
        use_gpu=False, config=py_utils.SessionConfig(inline=False)):
      params = rnn_cell.LSTMCellSimple.Params()
      params.name = 'lstm'
      params.output_nonlinearity = True
      params.params_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)

      params.vn.global_vn = False
      params.vn.per_step_vn = False
      params.num_input_nodes = 2
      params.num_output_nodes = 2
      params.apply_pruning = True
      lstm = rnn_cell.LSTMCellSimple(params)

      print('lstm vars = ', lstm.vars)
      self.assertTrue('wm' in lstm.vars.wm.name)
      self.assertTrue('b' in lstm.vars.b.name)
      self.assertTrue('mask' in lstm.vars.mask.name)
      self.assertTrue('threshold' in lstm.vars.threshold.name)

      num_param_vectors = 8
      self.assertEqual(lstm.theta.wm.get_shape(),
                       tf.TensorShape([4, num_param_vectors]))
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

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()

      variable_count = 2
      wts = tf.get_collection('LSTMCellSimple_vars')
      self.assertEqual(variable_count, len(wts))

      masks = tf.get_collection('masks')
      self.assertEqual(1, len(masks))

      threshold = tf.get_collection('thresholds')
      self.assertEqual(1, len(threshold))

      m_expected = [[0.095727, 0.476658], [0.04662, 0.180589],
                    [0.001656, 0.374141]]
      c_expected = [[0.241993, 0.820267], [0.086863, 0.349722],
                    [0.003176, 0.655448]]
      self.assertAllClose(m_expected, state1.m.eval())
      self.assertAllClose(c_expected, state1.c.eval())

  def testLSTMSimple_Projections(self):
    with self.test_session(
        use_gpu=False, config=py_utils.SessionConfig(inline=False)):
      params = rnn_cell.LSTMCellSimple.Params()
      params.name = 'lstm'
      params.output_nonlinearity = True
      params.params_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)

      params.vn.global_vn = False
      params.vn.per_step_vn = False
      params.num_input_nodes = 2
      params.num_output_nodes = 1
      params.num_hidden_nodes = 2
      params.trainable_zero_state = False

      lstm = rnn_cell.LSTMCellSimple(params)

      print('lstm vars = ', lstm.vars)
      self.assertTrue('wm' in lstm.vars.wm.name)
      self.assertTrue('b' in lstm.vars.b.name)

      self.assertEqual(lstm.theta.wm.get_shape(), tf.TensorShape([3, 8]))
      self.assertEqual(lstm.theta.b.get_shape(), tf.TensorShape([8]))

      np.random.seed(_NUMPY_RANDOM_SEED)
      inputs = py_utils.NestedMap(
          act=[tf.constant(np.random.uniform(size=(3, 2)), tf.float32)],
          padding=tf.zeros([3, 1]))
      state0 = py_utils.NestedMap(
          c=tf.constant(np.random.uniform(size=(3, 2)), tf.float32),
          m=tf.constant(np.random.uniform(size=(3, 1)), tf.float32))

      state1, _ = lstm.FPropDefaultTheta(state0, inputs)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()

      variable_count = 3  # weights, biases, projection.
      wts = tf.get_collection('LSTMCellSimple_vars')
      self.assertEqual(variable_count, len(wts))

      m_expected = [[0.414049], [0.076521], [0.356313]]
      c_expected = [[0.270425, 0.840373], [0.349856, 0.440421],
                    [0.261243, 0.889804]]
      self.assertAllClose(m_expected, state1.m.eval())
      self.assertAllClose(c_expected, state1.c.eval())

  def testLSTMSimpleGroupedWithoutShuffling(self):
    self._testLSTMSimpleGrouped(num_shuffle_shards=1)

  def testLSTMSimpleGroupedWithOutputShuffling(self):
    self._testLSTMSimpleGrouped(num_shuffle_shards=2)

  def _testLSTMSimpleGrouped(self, num_shuffle_shards):
    with self.test_session(
        use_gpu=False, config=py_utils.SessionConfig(inline=False)):
      params = rnn_cell.LSTMCellGrouped.Params()
      params.name = 'lstm'
      params.num_input_nodes = 8
      params.num_output_nodes = 8
      params.num_groups = 4
      params.num_shuffle_shards = num_shuffle_shards
      child_p = params.child_lstm_tpl
      child_p.output_nonlinearity = True
      child_p.params_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)
      child_p.vn.global_vn = False
      child_p.vn.per_step_vn = False
      child_p.trainable_zero_state = False

      lstm = params.cls(params)

      print('lstm vars = ', lstm.vars)
      for child_lstm in lstm.groups:
        self.assertTrue('wm' in child_lstm.vars.wm.name)
        self.assertTrue('b' in child_lstm.vars.b.name)

        self.assertEqual(child_lstm.theta.wm.get_shape(), tf.TensorShape([4,
                                                                          8]))
        self.assertEqual(child_lstm.theta.b.get_shape(), tf.TensorShape([8]))

      np.random.seed(_NUMPY_RANDOM_SEED)
      inputs = py_utils.NestedMap(
          act=[tf.constant(np.random.uniform(size=(3, 8)), tf.float32)],
          padding=tf.zeros([3, 1]))
      state0 = py_utils.NestedMap(
          groups=py_utils.SplitRecursively(
              py_utils.NestedMap(
                  c=tf.constant(np.random.uniform(size=(3, 8)), tf.float32),
                  m=tf.constant(np.random.uniform(size=(
                      3, 8)), tf.float32)), params.num_groups))

      state1, _ = lstm.FPropDefaultTheta(state0, inputs)
      self.assertEqual(params.num_groups, len(state1.groups))
      out1 = lstm.GetOutput(state1)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()

      variable_count = 2 * params.num_groups  # one for weights, one for biases.
      wts = tf.get_collection('LSTMCellSimple_vars')
      self.assertEqual(variable_count, len(wts))

      state1 = py_utils.ConcatRecursively(state1.groups)
      m_actual = state1.m.eval()
      c_actual = state1.c.eval()
      out_actual = out1.eval()
      print('m_actual =', np.array_repr(m_actual))
      print('c_actual =', np.array_repr(c_actual))
      print('out_actual =', np.array_repr(out_actual))

      # pylint: disable=bad-whitespace, line-too-long
      m_expected = [[
          -0.07857136, 0.43932292, 0.11373602, 0.16337454, 0.01618987,
          0.09685542, -0.20168062, 0.52612996
      ], [
          0.07929622, 0.18910739, -0.11084013, 0.32307294, 0.03500029,
          -0.05823045, 0.16963124, 0.27039385
      ], [
          0.11623365, 0.38104215, 0.00935007, 0.22124135, -0.17368057,
          0.10859803, -0.06948104, 0.10925373
      ]]
      c_expected = [[
          -0.23670214, 0.66260374, 0.24650344, 0.28946888, 0.03051668,
          0.15143034, -0.52736223, 0.88325077
      ], [
          0.16262427, 0.28568456, -0.19542629, 0.52116692, 0.06872599,
          -0.1123996, 0.31477568, 0.49881396
      ], [
          0.19667494, 0.68746102, 0.02078706, 0.30816019, -0.36376655,
          0.16003416, -0.16141629, 0.16648693
      ]]
      out_expected = m_expected
      # pylint: enable=bad-whitespace, line-too-long
      if num_shuffle_shards > 1:

        def _ShuffleShards(x):
          return [[row[i] for i in (0, 3, 2, 5, 4, 7, 6, 1)] for row in x]

        assert num_shuffle_shards == 2
        out_expected = _ShuffleShards(out_expected)
      self.assertAllClose(m_expected, m_actual)
      self.assertAllClose(c_expected, c_actual)
      self.assertAllClose(out_expected, out_actual)

  def _testLSTMSimple_VN(self):
    with self.test_session(use_gpu=False):
      params = rnn_cell.LSTMCellSimple.Params()
      params.name = 'lstm'
      params.output_nonlinearity = True
      params.params_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)

      params.vn.seed = 8820495
      params.vn.global_vn = True
      params.vn.per_step_vn = False
      params.vn.scale = 0.5
      params.num_input_nodes = 2
      params.num_output_nodes = 2

      lstm = rnn_cell.LSTMCellSimple(params)

      np.random.seed(_NUMPY_RANDOM_SEED)
      inputs = py_utils.NestedMap(
          act=[tf.constant(np.random.uniform(size=(3, 2)), tf.float32)],
          padding=tf.zeros([3, 1]))
      state0 = py_utils.NestedMap(
          c=tf.constant(np.random.uniform(size=(3, 2)), tf.float32),
          m=tf.constant(np.random.uniform(size=(3, 2)), tf.float32))

      state1, _ = lstm.FPropDefaultTheta(state0, inputs)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()

      # pyformat: disable
      m_expected = [
          [0.080182, 0.4585],
          [0.050852, 0.245296],
          [0.023557, 0.382329]]
      c_expected = [
          [0.387777, 0.819644],
          [-0.160634, 0.513716],
          [-0.179584, 0.862915]]
      # pyformat: enable
      self.assertAllClose(m_expected, state1.m.eval())
      self.assertAllClose(c_expected, state1.c.eval())

  def testLSTMSimple_GlobalNoise(self):
    self._testLSTMSimple_VN()

  def testLSTMSimpleDouble(self):
    with self.test_session(use_gpu=False):
      params = rnn_cell.LSTMCellSimple.Params()
      params.name = 'lstm'
      params.output_nonlinearity = True
      params.params_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)

      params.vn.seed = 8820495
      params.vn.global_vn = True
      params.vn.per_step_vn = False
      params.vn.scale = 0.5
      params.num_input_nodes = 2
      params.num_output_nodes = 2
      params.dtype = tf.float64

      lstm = rnn_cell.LSTMCellSimple(params)

      np.random.seed(_NUMPY_RANDOM_SEED)
      inputs = py_utils.NestedMap(
          act=[tf.constant(np.random.uniform(size=(3, 2)), tf.float64)],
          padding=tf.zeros([3, 1], dtype=tf.float64))
      state0 = py_utils.NestedMap(
          c=tf.constant(np.random.uniform(size=(3, 2)), tf.float64),
          m=tf.constant(np.random.uniform(size=(3, 2)), tf.float64))
      state1, _ = lstm.FPropDefaultTheta(state0, inputs)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()

      # pyformat: disable
      m_expected = [
          [0.3472136, 0.11880029],
          [0.14214374, 0.33760977],
          [0.1168568, 0.32053401]]
      c_expected = [
          [1.46477364, 0.43743008],
          [0.57051592, 0.14892339],
          [0.6949858, 0.16326128]]
      # pyformat: enable
      self.assertAllClose(m_expected, state1.m.eval())
      self.assertAllClose(c_expected, state1.c.eval())

  def testLSTMSimpleNoOutputNonlinearity(self):
    with self.test_session(use_gpu=False):
      params = rnn_cell.LSTMCellSimple.Params()
      params.name = 'lstm'
      params.output_nonlinearity = False
      params.dtype = tf.float64
      params.cell_value_cap = 10.0
      params.params_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)

      params.vn.global_vn = False
      params.vn.per_step_vn = False
      params.num_input_nodes = 2
      params.num_output_nodes = 2

      lstm = rnn_cell.LSTMCellSimple(params)

      np.random.seed(_NUMPY_RANDOM_SEED)
      inputs = py_utils.NestedMap(
          act=[tf.constant(np.random.uniform(size=(3, 2)), tf.float64)],
          padding=tf.zeros([3, 1], dtype=tf.float64))
      state0 = py_utils.NestedMap(
          c=tf.constant(np.random.uniform(size=(3, 2)), tf.float64),
          m=tf.constant(np.random.uniform(size=(3, 2)), tf.float64))
      state1, _ = lstm.FPropDefaultTheta(state0, inputs)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()

      wts = tf.get_collection('LSTMCellSimple_vars')
      self.assertEqual(2, len(wts))

      # pyformat: disable
      m_expected = [
          [0.532625, 0.083511],
          [0.118662, 0.110532],
          [0.121542, 0.084161]]
      c_expected = [
          [0.789908, 0.312811],
          [0.192642, 0.207369],
          [0.167591, 0.172713]]
      # pyformat: enable

      self.assertAllClose(m_expected, state1.m.eval())
      self.assertAllClose(c_expected, state1.c.eval())

  def testLSTMSimpleBypass(self):
    with self.test_session(use_gpu=False):
      params = rnn_cell.LSTMCellSimple.Params()
      params.name = 'lstm'
      params.output_nonlinearity = False
      params.dtype = tf.float64
      params.cell_value_cap = 10.0
      params.params_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)

      params.vn.global_vn = False
      params.vn.per_step_vn = False
      params.num_input_nodes = 2
      params.num_output_nodes = 2

      lstm = rnn_cell.LSTMCellSimple(params)

      np.random.seed(_NUMPY_RANDOM_SEED)
      inputs = py_utils.NestedMap(
          act=[tf.constant(np.random.uniform(size=(3, 2)), tf.float64)],
          padding=tf.ones([3, 1], dtype=tf.float64))
      state0 = py_utils.NestedMap(
          c=tf.constant(np.random.uniform(size=(3, 2)), tf.float64),
          m=tf.constant(np.random.uniform(size=(3, 2)), tf.float64))
      state1, _ = lstm.FPropDefaultTheta(state0, inputs)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()

      wts = tf.get_collection('LSTMCellSimple_vars')
      self.assertEqual(2, len(wts))

      # pyformat: enable
      self.assertAllClose(state0.m.eval(), state1.m.eval())
      self.assertAllClose(state0.c.eval(), state1.c.eval())

  def testLSTMCuDNNCompliant(self):
    if not tf.test.is_gpu_available(cuda_only=True):
      return
    batch_size = 32
    input_nodes = 16
    cell_nodes = 8
    dtype = tf.float32

    # LSTM inputs and states constants.
    np.random.seed(_NUMPY_RANDOM_SEED)
    inputs_v = np.random.uniform(size=(batch_size, input_nodes))
    paddings_v = np.zeros((batch_size, 1))
    # The last example is padded.
    paddings_v[-1] = 1.
    h_v = np.zeros((batch_size, cell_nodes))
    c_v = np.zeros((batch_size, cell_nodes))
    cudnn_helper = cudnn_rnn_utils.CuDNNLSTMInitializer(
        input_nodes, cell_nodes, direction=UNI_RNN)

    # tf CuDNN LSTM
    with self.test_session(use_gpu=True, graph=tf.Graph()) as sess:
      inputs = tf.expand_dims(tf.constant(inputs_v, dtype=dtype), 0)
      state_h = tf.expand_dims(tf.constant(h_v, dtype=dtype), 0)
      state_c = tf.expand_dims(tf.constant(c_v, dtype=dtype), 0)
      cudnn_params = tf.get_variable(
          'cudnn_opaque_params',
          initializer=tf.random_uniform(
              shape=[cudnn_helper.OpaqueParamsShape(dtype)], dtype=dtype),
          validate_shape=False)

      outputs, h, c, _ = gen_cudnn_rnn_ops.cudnn_rnn(
          input=inputs,
          input_h=state_h,
          input_c=state_c,
          params=cudnn_params,
          rnn_mode='lstm',
          input_mode='linear_input',
          direction='unidirectional',
          dropout=0.0,
          is_training=True)
      outputs = tf.squeeze(outputs, axis=0) * (1.0 - paddings_v)
      h = tf.squeeze(h, axis=0) * (1.0 - paddings_v)
      c = tf.squeeze(c, axis=0) * (1.0 - paddings_v)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()
      cudnn_params_v = sess.run(cudnn_params)
      cudnn_outputs_v, cudnn_h_v, cudnn_c_v = sess.run([outputs, h, c])

    # LSTMCellCuDNNCompliant
    with self.test_session(use_gpu=False, graph=tf.Graph()) as sess:
      p = rnn_cell.LSTMCellCuDNNCompliant.Params()
      p.name = 'lstm_cell_cudnn'
      p.dtype = dtype
      p.num_input_nodes = input_nodes
      p.num_output_nodes = cell_nodes
      lstm = rnn_cell.LSTMCellCuDNNCompliant(p)

      assign_op = tf.assign(lstm.vars.wb, cudnn_params_v)

      inputs = py_utils.NestedMap(
          act=[tf.constant(inputs_v, dtype=p.dtype)],
          padding=tf.constant(paddings_v, dtype=p.dtype))
      state0 = py_utils.NestedMap(
          m=tf.constant(h_v, dtype=dtype), c=tf.constant(c_v, dtype=dtype))
      state1, _ = lstm.FPropDefaultTheta(state0, inputs)
      state1.m *= 1.0 - paddings_v
      state1.c *= 1.0 - paddings_v

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()
      assign_op.op.run()
      outputs_v, h_v, c_v = sess.run([state1.m, state1.m, state1.c])

    self.assertAllClose(cudnn_outputs_v, outputs_v)
    self.assertAllClose(cudnn_h_v, h_v)
    self.assertAllClose(cudnn_c_v, c_v)

  def _testConvLSTMHelper(self, inline=False):
    with self.test_session(
        use_gpu=False, config=py_utils.SessionConfig(inline=inline)):
      params = rnn_cell.ConvLSTMCell.Params()
      params.name = 'conv_lstm'
      params.output_nonlinearity = True
      params.params_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)

      params.vn.global_vn = False
      params.vn.per_step_vn = False
      params.inputs_shape = [None, 4, 2, 3]
      params.cell_shape = [None, 4, 2, 2]
      params.filter_shape = [3, 2]

      lstm = rnn_cell.ConvLSTMCell(params)

      lstm_vars = lstm.vars
      print('lstm vars = ', lstm_vars)
      self.assertTrue('wm' in lstm_vars.wm.name)
      self.assertTrue('b' in lstm_vars.b.name)

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

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()

      wts = tf.get_collection('ConvLSTMCell_vars')
      self.assertEqual(2, len(wts))

      # pyformat: disable
      m_expected = [
          0.4144063, 0.88831079, 0.56665027, 0.30154669, 0.2818037]
      c_expected = [
          4.72228432, 3.9454143, 3.77556086, 2.76972866, 1.87397099]
      # print('m1.eval', m1.eval())
      # print('c1.eval', c1.eval())
      # pyformat: enable
      self.assertAllClose(m_expected, m1.eval())
      self.assertAllClose(c_expected, c1.eval())

  def testConvLSTM_NoInline(self):
    self._testConvLSTMHelper(inline=False)

  def testConvLSTM_Inline(self):
    self._testConvLSTMHelper(inline=True)

  def _testConvLSTM_VN(self, per_step_vn=False):
    with self.test_session(use_gpu=False):
      params = rnn_cell.ConvLSTMCell.Params()
      params.name = 'conv_lstm'
      params.output_nonlinearity = True
      params.params_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)

      params.vn.seed = 8820495
      params.vn.global_vn = not per_step_vn
      params.vn.per_step_vn = per_step_vn
      params.vn.scale = 0.5

      params.inputs_shape = [None, 4, 2, 3]
      params.cell_shape = [None, 4, 2, 2]
      params.filter_shape = [3, 2]

      lstm = rnn_cell.ConvLSTMCell(params)

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

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()

      # pyformat: disable
      m_expected = [
          0.21634784, 0.40635043, 0.12228709, 0.51806468, 0.02064975]
      c_expected = [
          5.21427298, 4.5560832, 4.24992609, 3.85193706, 2.35372424]
      # print('m1.eval', np.array_repr(m1.eval()))
      # print('c1.eval', np.array_repr(c1.eval()))
      # pyformat: enable
      self.assertAllClose(m_expected, m1.eval())
      self.assertAllClose(c_expected, c1.eval())

  def testConvLSTM_GlobalNoise(self):
    self._testConvLSTM_VN(per_step_vn=False)

  def testLSTMSimpleWithForgetGateBias(self):
    with self.test_session(use_gpu=False):
      params = rnn_cell.LSTMCellSimple.Params()
      params.name = 'lstm'
      params.output_nonlinearity = True
      params.params_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)

      params.vn.seed = 8820495
      params.vn.global_vn = True
      params.vn.per_step_vn = False
      params.vn.scale = 0.5
      params.num_input_nodes = 2
      params.num_output_nodes = 2
      params.forget_gate_bias = -1.0
      params.dtype = tf.float64

      lstm = rnn_cell.LSTMCellSimple(params)

      np.random.seed(_NUMPY_RANDOM_SEED)
      inputs = py_utils.NestedMap(
          act=[tf.constant(np.random.uniform(size=(3, 2)), tf.float64)],
          padding=tf.zeros([3, 1], dtype=tf.float64))
      state0 = py_utils.NestedMap(
          c=tf.constant(np.random.uniform(size=(3, 2)), tf.float64),
          m=tf.constant(np.random.uniform(size=(3, 2)), tf.float64))
      state1, _ = lstm.FPropDefaultTheta(state0, inputs)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()

      # pyformat: disable
      m_expected = [
          [0.19534954, 0.10979363],
          [0.02134449, 0.2821926],
          [-0.02530111, 0.25382254]]
      c_expected = [
          [1.29934979, 0.31769676],
          [0.41655035, 0.05172589],
          [0.58909841, -0.00438461]]
      # pyformat: enable
      self.assertAllClose(m_expected, state1.m.eval())
      self.assertAllClose(c_expected, state1.c.eval())

  def testZoneOut_Disabled(self):
    with self.test_session(use_gpu=False):
      prev_v = [[0.2, 0.0], [0.1, 0.4], [0.1, 0.5]]
      cur_v = [[0.3, 1.0], [0.0, 2.4], [0.2, 3.4]]
      padding_v = [[1.0], [0.0], [0.0]]
      new_v = rnn_cell.ZoneOut(
          prev_v,
          cur_v,
          padding_v,
          zo_prob=0.0,
          is_eval=False,
          random_uniform=None)
      v_expected = [[0.2, 0.], [0., 2.4], [0.2, 3.4]]
      new_v_evaled = new_v.eval()
      print(np.array_repr(new_v_evaled))
      self.assertAllClose(v_expected, new_v_evaled)

  def testZoneOut_Enabled(self):
    with self.test_session(use_gpu=False):
      prev_v = [[0.2, 0.0], [0.1, 0.4], [0.1, 0.5]]
      cur_v = [[0.3, 1.0], [0.0, 2.4], [0.2, 3.4]]
      padding_v = [[1.0], [0.0], [0.0]]
      new_v = rnn_cell.ZoneOut(
          prev_v,
          cur_v,
          padding_v,
          zo_prob=0.5,
          is_eval=False,
          random_uniform=tf.random_uniform([3, 2], seed=98798202))
      v_expected = [[0.2, 0.], [0., 0.4], [0.2, 0.5]]
      new_v_evaled = new_v.eval()
      print(np.array_repr(new_v_evaled))
      self.assertAllClose(v_expected, new_v_evaled)

  def testZoneOut_Eval(self):
    with self.test_session(use_gpu=False):
      prev_v = [[0.2, 0.0], [0.1, 0.4], [0.1, 0.5]]
      cur_v = [[0.3, 1.0], [0.0, 2.4], [0.2, 3.4]]
      padding_v = [[1.0], [0.0], [0.0]]
      new_v = rnn_cell.ZoneOut(
          prev_v,
          cur_v,
          padding_v,
          zo_prob=0.5,
          is_eval=True,
          random_uniform=None)
      # In eval mode, if padding[i] == 1, new_v equals prev_v.
      # Otherwise, new_v = zo_prob * prev_v + (1.0 - zo_prob) * cur_v
      v_expected = [[0.2, 0.], [0.05, 1.4], [0.15, 1.95]]
      new_v_evaled = new_v.eval()
      print(np.array_repr(new_v_evaled))
      self.assertAllClose(v_expected, new_v_evaled)

  def testLSTMSimpleWithZoneOut(self):
    with self.test_session(use_gpu=False):
      params = rnn_cell.LSTMCellSimple.Params()
      params.name = 'lstm'
      params.output_nonlinearity = True
      params.params_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      params.num_input_nodes = 2
      params.num_output_nodes = 2
      params.zo_prob = 0.5
      params.random_seed = _RANDOM_SEED

      lstm = rnn_cell.LSTMCellSimple(params)

      np.random.seed(_NUMPY_RANDOM_SEED)
      inputs = py_utils.NestedMap(
          act=[tf.constant(np.random.uniform(size=(3, 2)), tf.float32)],
          padding=tf.zeros([3, 1]))
      state0 = py_utils.NestedMap(
          c=tf.constant(np.random.uniform(size=(3, 2)), tf.float32),
          m=tf.constant(np.random.uniform(size=(3, 2)), tf.float32))
      state1, _ = lstm.FPropDefaultTheta(state0, inputs)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()

      m_expected = [[0.0083883, 0.10644437], [0.04662009, 0.18058866],
                    [0.0016561, 0.37414068]]
      c_expected = [[0.96451449, 0.65317708], [0.08686253, 0.34972212],
                    [0.00317609, 0.6554482]]
      m_v = state1.m.eval()
      c_v = state1.c.eval()
      print('m_v', np.array_repr(m_v))
      print('c_v', np.array_repr(c_v))
      self.assertAllClose(m_expected, m_v)
      self.assertAllClose(c_expected, c_v)

  def _testLSTMSimpleDeterministicWithZoneOutHelper(self, seed_dtype=tf.int64):
    with self.test_session(use_gpu=False):
      params = rnn_cell.LSTMCellSimpleDeterministic.Params()
      params.name = 'lstm'
      params.output_nonlinearity = True
      params.params_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      params.num_input_nodes = 2
      params.num_output_nodes = 2
      params.zo_prob = 0.5
      params.random_seed = _RANDOM_SEED

      lstm = rnn_cell.LSTMCellSimpleDeterministic(params)

      np.random.seed(_NUMPY_RANDOM_SEED)
      inputs = py_utils.NestedMap(
          act=[tf.constant(np.random.uniform(size=(3, 2)), tf.float32)],
          padding=tf.zeros([3, 1]))
      state0 = lstm.zero_state(3)
      state1, _ = lstm.FPropDefaultTheta(state0, inputs)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()

      m_expected = [[-0.145889, 0.], [-0.008282, 0.073219], [-0.041057, 0.]]
      c_expected = [[0., 0.532332], [-0.016117, 0.13752], [0., 0.]]
      m_v = state1.m.eval()
      c_v = state1.c.eval()
      print('m_v', np.array_repr(m_v))
      print('c_v', np.array_repr(c_v))
      self.assertAllClose(m_expected, m_v)
      self.assertAllClose(c_expected, c_v)

  def testLSTMSimpleDeterministicWithZoneOutInt64(self):
    self._testLSTMSimpleDeterministicWithZoneOutHelper(tf.int64)

  def testLSTMSimpleDeterministicWithZoneOutInt32(self):
    self._testLSTMSimpleDeterministicWithZoneOutHelper(tf.int32)

  def testLNLSTMCell(self):
    m_v, c_v = self._testLNLSTMCell(rnn_cell.LayerNormalizedLSTMCell)
    m_expected = [[0.03960676, 0.26547235], [-0.00677715, 0.09782403],
                  [-0.00272907, 0.31641623]]
    c_expected = [[0.14834785, 0.3804915], [-0.00927538, 0.38059634],
                  [-0.01014781, 0.46336061]]
    self.assertAllClose(m_expected, m_v)
    self.assertAllClose(c_expected, c_v)

  def testLNLSTMCellSimple(self):
    m_v, c_v = self._testLNLSTMCell(rnn_cell.LayerNormalizedLSTMCellSimple)
    m_expected = [[0.03960676, 0.26547235], [-0.00677715, 0.09782403],
                  [-0.00272907, 0.31641623]]
    c_expected = [[0.14834785, 0.3804915], [-0.00927538, 0.38059634],
                  [-0.01014781, 0.46336061]]
    self.assertAllClose(m_expected, m_v)
    self.assertAllClose(c_expected, c_v)

  def testLNLSTMCellProj(self):
    m_v, c_v = self._testLNLSTMCell(
        rnn_cell.LayerNormalizedLSTMCellSimple, num_hidden_nodes=4)
    m_expected = [[0.39790073, 0.28511256], [0.41482946, 0.28972796],
                  [0.47132283, 0.03284446]]
    c_expected = [[-0.3667627, 1.03294277, 0.24229962, 0.43976486],
                  [-0.15832338, 1.22740746, 0.19910297, -0.14970526],
                  [-0.57552528, 0.9139322, 0.41805002, 0.58792269]]
    self.assertAllClose(m_expected, m_v)
    self.assertAllClose(c_expected, c_v)

  def _testLNLSTMCell(self, cell_cls, num_hidden_nodes=0):
    with self.test_session(use_gpu=False):
      params = cell_cls.Params()
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

      lstm = params.cls(params)

      np.random.seed(_NUMPY_RANDOM_SEED)
      inputs = py_utils.NestedMap(
          act=[tf.constant(np.random.uniform(size=(3, 2)), tf.float32)],
          padding=tf.zeros([3, 1]))
      state0 = py_utils.NestedMap(
          c=tf.constant(
              np.random.uniform(size=(3, lstm.hidden_size)), tf.float32),
          m=tf.constant(
              np.random.uniform(size=(3, lstm.output_size)), tf.float32))
      state1, _ = lstm.FPropDefaultTheta(state0, inputs)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()

      m_v = state1.m.eval()
      c_v = state1.c.eval()
      print('m_v', np.array_repr(m_v))
      print('c_v', np.array_repr(c_v))
      return m_v, c_v

  def testQuantizedLSTMCellPadding(self):
    with self.test_session(use_gpu=False) as sess:
      params = rnn_cell.QuantizedLSTMCell.Params()
      params.name = 'lstm'
      params.params_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      params.num_input_nodes = 2
      params.num_output_nodes = 2
      params.cc_schedule.start_step = 0
      params.cc_schedule.end_step = 2
      params.cc_schedule.start_cap = 5.0
      params.cc_schedule.end_cap = 1.0

      lstm = rnn_cell.QuantizedLSTMCell(params)
      lstm_vars = lstm.vars
      print('lstm vars = ', lstm_vars)
      self.assertTrue('wm' in lstm_vars.wm.name)
      self.assertTrue('cap' in lstm_vars.cc_schedule.cap.name)

      wm = lstm.theta.wm
      cap = lstm.theta.cc_schedule.cap
      self.assertEqual(wm.get_shape(), tf.TensorShape([4, 8]))
      self.assertEqual(cap.get_shape(), tf.TensorShape([]))

      np.random.seed(_NUMPY_RANDOM_SEED)
      inputs = py_utils.NestedMap(
          act=[tf.constant(np.random.uniform(size=(3, 2)), tf.float32)],
          padding=tf.ones([3, 1]))
      state0 = py_utils.NestedMap(
          c=tf.zeros([3, 2], tf.float32), m=tf.zeros([3, 2], tf.float32))
      state1, _ = lstm.FPropDefaultTheta(state0, inputs)

      tf.global_variables_initializer().run()

      m_expected = [[0.0, 0.], [0.0, 0.], [0.0, 0.]]
      c_expected = [[0.0, 0.], [0.0, 0.], [0.0, 0.]]
      self.assertAllClose(m_expected, state1.m.eval())
      self.assertAllClose(c_expected, state1.c.eval())
      update_op = lstm.PostTrainingStepUpdate(1)
      sess.run(update_op)

  def testQuantizedLayerNormalizedLSTMCell(self):
    params = rnn_cell.LayerNormalizedLSTMCell.Params()
    params.name = 'lstm'
    params.params_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)
    params.vn.global_vn = False
    params.vn.per_step_vn = False
    params.num_input_nodes = 2
    params.num_output_nodes = 2
    params.zo_prob = 0.0
    params.random_seed = _RANDOM_SEED
    params.cc_schedule = quant_utils.LinearClippingCapSchedule.Params().Set(
        start_step=0, end_step=2, start_cap=5.0, end_cap=1.0)

    lstm = rnn_cell.LayerNormalizedLSTMCell(params)
    lstm_vars = lstm.vars
    print('lstm vars = ', lstm_vars)
    self.assertTrue('wm' in lstm_vars.wm.name)
    self.assertTrue('cap' in lstm_vars.cc_schedule.cap.name)

    wm = lstm.theta.wm
    cap = lstm.theta.cc_schedule.cap
    self.assertEqual(wm.get_shape(), tf.TensorShape([4, 8]))
    self.assertEqual(cap.get_shape(), tf.TensorShape([]))

    np.random.seed(_NUMPY_RANDOM_SEED)
    inputs = py_utils.NestedMap(
        act=[tf.constant(np.random.uniform(size=(3, 2)), tf.float32)],
        padding=tf.zeros([3, 1]))
    state0 = py_utils.NestedMap(
        c=tf.constant(np.random.uniform(size=(3, 2)), tf.float32),
        m=tf.constant(np.random.uniform(size=(3, 2)), tf.float32))
    state1, _ = lstm.FPropDefaultTheta(state0, inputs)

    with self.test_session(use_gpu=False) as sess:
      tf.global_variables_initializer().run()
      # pylint: disable=bad-whitespace
      m_expected = [[0.03960676, 0.26547235], [-0.00677715, 0.09782403],
                    [-0.00272907, 0.31641623]]
      c_expected = [[0.14834785, 0.3804915], [-0.00927538, 0.38059634],
                    [-0.01014781, 0.46336061]]
      # pylint: enable=bad-whitespace
      self.assertAllClose(m_expected, state1.m.eval())
      self.assertAllClose(c_expected, state1.c.eval())

      self.assertEqual(5.0, cap.eval())
      update_op = lstm.PostTrainingStepUpdate(1)
      sess.run(update_op)
      self.assertEqual(3.0, cap.eval())

  def testQuantizedLSTMCell(self):
    with self.test_session(use_gpu=False) as sess:
      params = rnn_cell.QuantizedLSTMCell.Params()
      params.name = 'lstm'
      params.params_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      params.num_input_nodes = 2
      params.num_output_nodes = 2
      params.cc_schedule.start_step = 0
      params.cc_schedule.end_step = 2
      params.cc_schedule.start_cap = 5.0
      params.cc_schedule.end_cap = 1.0

      lstm = rnn_cell.QuantizedLSTMCell(params)
      lstm_vars = lstm.vars
      print('lstm vars = ', lstm_vars)
      self.assertTrue('wm' in lstm_vars.wm.name)
      self.assertTrue('cap' in lstm_vars.cc_schedule.cap.name)

      wm = lstm.theta.wm
      cap = lstm.theta.cc_schedule.cap
      self.assertEqual(wm.get_shape(), tf.TensorShape([4, 8]))
      self.assertEqual(cap.get_shape(), tf.TensorShape([]))

      np.random.seed(_NUMPY_RANDOM_SEED)
      inputs = py_utils.NestedMap(
          act=[tf.constant(np.random.uniform(size=(3, 2)), tf.float32)],
          padding=tf.zeros([3, 1]))
      state0 = py_utils.NestedMap(
          c=tf.constant(np.random.uniform(size=(3, 2)), tf.float32),
          m=tf.constant(np.random.uniform(size=(3, 2)), tf.float32))
      state1, _ = lstm.FPropDefaultTheta(state0, inputs)

      tf.global_variables_initializer().run()

      m_expected = [[0.097589, 0.579055], [0.046737, 0.187892],
                    [0.001656, 0.426245]]
      c_expected = [[0.241993, 0.820267], [0.086863, 0.349722],
                    [0.003176, 0.655448]]
      self.assertAllClose(m_expected, state1.m.eval())
      self.assertAllClose(c_expected, state1.c.eval())

      self.assertEqual(5.0, cap.eval())
      update_op = lstm.PostTrainingStepUpdate(1)
      sess.run(update_op)
      self.assertEqual(3.0, cap.eval())

  def testFakeQuantizedLSTMCellTrainingUnclipped(self):
    m_expected = [[0.097589, 0.579055], [0.046737, 0.187892],
                  [0.001656, 0.426245]]
    c_expected = [[0.241993, 0.820267], [0.086863, 0.349722],
                  [0.003176, 0.655448]]
    self._testFakeQuantizedLSTMCellHelper(
        is_inference=False,
        set_training_step=False,
        m_expected=m_expected,
        c_expected=c_expected)

  def testFakeQuantizedLSTMCellTraining(self):
    padding = [[0.0], [0.0], [1.0]]
    m_expected = [[0.09375, 0.5625], [0.046875, 0.1875], [0.809813, 0.872176]]
    c_expected = [[0.23288, 0.806], [0.090057, 0.355591], [0.747715, 0.961307]]
    self._testFakeQuantizedLSTMCellHelper(
        is_inference=False,
        set_training_step=True,
        m_expected=m_expected,
        c_expected=c_expected,
        padding=padding)

  def testFakeQuantizedLSTMCellHiddenNodes(self):
    m_expected = [[0.382812, 0.296875], [0.164062, 0.171875],
                  [0.3125, -0.039062]]
    c_expected = [[-0.160339, 0.795929, 0.449707,
                   0.347534], [-0.049194, 0.548279, -0.060852, -0.106354],
                  [-0.464172, 0.345947, 0.407349, 0.430878]]
    self._testFakeQuantizedLSTMCellHelper(
        is_inference=False,
        set_training_step=True,
        m_expected=m_expected,
        c_expected=c_expected,
        num_hidden_nodes=4)

  def testFakeQuantizedLSTMCellInference(self):
    # At inference time, quantization/clipping is forced on, so even though
    # we don't set the training step, we should get fully quantized results.
    m_expected = [[0.09375, 0.5625], [0.046875, 0.1875], [0., 0.429688]]
    c_expected = [[0.23288, 0.806], [0.090057, 0.355591], [-0.003937, 0.662567]]
    self._testFakeQuantizedLSTMCellHelper(
        is_inference=True,
        set_training_step=False,
        m_expected=m_expected,
        c_expected=c_expected)

  def _testFakeQuantizedLSTMCellHelper(self,
                                       is_inference,
                                       set_training_step,
                                       m_expected,
                                       c_expected,
                                       num_hidden_nodes=0,
                                       padding=None):
    with self.test_session(use_gpu=False) as sess:
      params = rnn_cell.FakeQuantizedLSTMCell.Params()
      params.name = 'lstm'
      params.is_eval = is_inference
      params.is_inference = is_inference
      params.params_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      params.num_input_nodes = 2
      params.num_output_nodes = 2
      params.num_hidden_nodes = num_hidden_nodes
      params.cc_schedule.clip_start_step = 1  # Step 0 is unclipped.
      params.cc_schedule.clip_end_step = 2
      params.cc_schedule.quant_start_step = 2
      params.cc_schedule.start_cap = 5.0
      params.cc_schedule.end_cap = 1.0

      lstm = rnn_cell.FakeQuantizedLSTMCell(params)
      lstm_vars = lstm.vars
      print('lstm vars = ', lstm_vars)
      self.assertTrue('wm' in lstm_vars.wm.name)
      if num_hidden_nodes:
        self.assertTrue('w_proj' in lstm_vars.w_proj.name)
      else:
        self.assertFalse('w_proj' in lstm_vars)
      if not is_inference:
        self.assertTrue('clip_ratio' in lstm_vars.cc_schedule.clip_ratio.name)
        self.assertTrue('fq_ratio' in lstm_vars.cc_schedule.fq_ratio.name)

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

      tf.global_variables_initializer().run()

      if set_training_step:
        # Get it into the fully clipped/quantized part of the schedule.
        update_op = lstm.PostTrainingStepUpdate(5)
        sess.run(update_op)

      # Outputs.
      self.assertAllClose(m_expected, state1.m.eval())
      self.assertAllClose(c_expected, state1.c.eval())

      # Cell reported zeros.
      cell_zero_state = lstm.zero_state(batch_size=3)
      self.assertAllEqual(cell_zero_state.m.eval(),
                          tf.zeros_like(state0.m).eval())
      self.assertAllEqual(cell_zero_state.c.eval(),
                          tf.zeros_like(state0.c).eval())

  def testSRUCell(self):
    with self.test_session(use_gpu=False):
      params = rnn_cell.SRUCell.Params()
      params.name = 'sru'
      params.params_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)
      params.num_input_nodes = 2
      params.num_output_nodes = 2
      params.zo_prob = 0.0
      params.random_seed = _RANDOM_SEED

      sru = rnn_cell.SRUCell(params)

      np.random.seed(_NUMPY_RANDOM_SEED)
      inputs = py_utils.NestedMap(
          act=[tf.constant(np.random.uniform(size=(3, 2)), tf.float32)],
          padding=tf.zeros([3, 1]))
      state0 = py_utils.NestedMap(
          c=tf.constant(np.random.uniform(size=(3, 2)), tf.float32),
          m=tf.constant(np.random.uniform(size=(3, 2)), tf.float32))
      state1, _ = sru.FPropDefaultTheta(state0, inputs)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()

      m_expected = [[0.58657289, 0.70520258], [0.32375532, 0.29133356],
                    [0.58900255, 0.58398587]]
      c_expected = [[0.45297623, 0.88433027], [0.42112729, 0.47023624],
                    [0.50483131, 0.89583319]]
      m_v = state1.m.eval()
      c_v = state1.c.eval()
      print('m_v', np.array_repr(m_v))
      print('c_v', np.array_repr(c_v))
      self.assertAllClose(m_expected, m_v)
      self.assertAllClose(c_expected, c_v)

  def testQRNNPoolingCell(self):
    num_rnn_matrices = 4
    with self.test_session(use_gpu=False):
      params = rnn_cell.QRNNPoolingCell.Params()
      params.name = 'QuasiRNN'
      params.params_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)
      params.num_input_nodes = 2
      params.num_output_nodes = 2
      params.zo_prob = 0.0
      params.random_seed = _RANDOM_SEED
      params.pooling_formula = 'quasi_ifo'

      qrnn = rnn_cell.QRNNPoolingCell(params)

      np.random.seed(_NUMPY_RANDOM_SEED)
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

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()

      m_expected = [[0.40564698, 0.32611847], [0.16975531, 0.45970476],
                    [0.60286027, 0.24542703]]
      c_expected = [[0.42057115, 0.49928033], [0.56830668, 0.7003305],
                    [1.28926766, 0.75380397]]
      m_v = state1.m.eval()
      c_v = state1.c.eval()
      print('m_v', np.array_repr(m_v))
      print('c_v', np.array_repr(c_v))
      self.assertAllClose(m_expected, m_v)
      self.assertAllClose(c_expected, c_v)

  def testQRNNPoolingCellInSRUMode(self):
    num_rnn_matrices = 4
    with self.test_session(use_gpu=False):
      params = rnn_cell.QRNNPoolingCell.Params()
      params.name = 'SRU'
      params.params_init = py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED)
      params.num_input_nodes = 2
      params.num_output_nodes = 2
      params.zo_prob = 0.0
      params.random_seed = _RANDOM_SEED
      params.pooling_formula = 'sru'

      sru = rnn_cell.QRNNPoolingCell(params)

      np.random.seed(_NUMPY_RANDOM_SEED)
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
      state1, _ = sru.FPropDefaultTheta(state0, inputs)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()

      m_expected = [[0.55884922, 0.40396619], [0.71426249, 0.70820922],
                    [0.823457, 0.60304904]]
      c_expected = [[0.65144706, 0.56252223], [0.75096267, 0.65605044],
                    [0.79761195, 0.36905324]]
      m_v = state1.m.eval()
      c_v = state1.c.eval()
      print('m_v', np.array_repr(m_v))
      print('c_v', np.array_repr(c_v))
      self.assertAllClose(m_expected, m_v)
      self.assertAllClose(c_expected, c_v)


if __name__ == '__main__':
  tf.test.main()
