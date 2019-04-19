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
"""Tests for rnn_layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import types
import unittest

import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow as tf

from tensorflow.python.ops import inplace_ops
from lingvo.core import attention
from lingvo.core import base_layer
from lingvo.core import cluster_factory
from lingvo.core import layers_with_attention
from lingvo.core import py_utils
from lingvo.core import rnn_cell
from lingvo.core import rnn_layers
from lingvo.core import test_utils

FLAGS = tf.flags.FLAGS


class TimestepAccumulator(base_layer.Accumulator):
  """Simple accumulator for counting timesteps."""

  def DefaultValue(self):
    return tf.convert_to_tensor(0.0)

  def Increment(self):
    self.SetValue(self.GetValue() + 1.0)


def AddTimestepAccumulator(layer):
  orig_fprop = layer.FProp

  def WrappedFProp(*args, **kwargs):
    layer.accumulators.ts_count.Increment()
    return orig_fprop(*args, **kwargs)

  layer.FProp = WrappedFProp

  layer.RegisterAccumulator('ts_count', TimestepAccumulator())


class LayersTestBase(test_utils.TestCase):

  def _testStackedFRNNHelper(self,
                             cls,
                             dtype,
                             trailing_pad_len=0,
                             keep_prob=1.0,
                             bi_directional=False,
                             input_dim=-1,
                             output_dim=-1):
    tf.set_random_seed(123456)
    batch = 3
    dims = 16
    slen = 10 + trailing_pad_len
    num_layers = 4
    with tf.Graph().as_default() as g:
      with self.session(use_gpu=True, graph=g) as sess:
        params = rnn_cell.LSTMCellSimple.Params()
        params.name = 'lstm'
        params.output_nonlinearity = True
        params.dtype = dtype
        params.params_init = py_utils.WeightInit.Uniform(1.24, 429891685)
        params.vn.global_vn = True
        params.vn.per_step_vn = False
        params.vn.seed = 2938482
        params.vn.scale = 0.1
        params.num_input_nodes = dims
        params.num_output_nodes = dims // 2 if bi_directional else dims

        sfrnn_params = cls.Params()
        sfrnn_params.name = 'sfrnn'
        sfrnn_params.dtype = dtype
        sfrnn_params.random_seed = 123456
        sfrnn_params.cell_tpl = params
        sfrnn_params.num_layers = num_layers
        sfrnn_params.skip_start = 2
        sfrnn_params.dropout.keep_prob = keep_prob
        sfrnn_params.num_input_nodes = input_dim
        sfrnn_params.num_output_nodes = output_dim
        with tf.name_scope('sfrnn'):
          sfrnn = sfrnn_params.cls(sfrnn_params)

        np.random.seed(12345)
        input_dim = input_dim if input_dim > 0 else dims
        output_dim = output_dim if output_dim > 0 else dims
        inputs = tf.constant(
            np.random.uniform(size=(slen, batch, input_dim)), dtype)
        paddings = np.zeros([slen, batch, 1])
        if trailing_pad_len > 0:
          paddings[-trailing_pad_len:, :] = 1.0
          paddings[-trailing_pad_len - 3:-trailing_pad_len - 1, :] = 1.0
        paddings = tf.constant(paddings, dtype)

        tf.global_variables_initializer().run()
        if bi_directional:
          sfrnn_outputs = sfrnn.FPropFullSequence(sfrnn.theta, inputs, paddings)
          sfrnn_outputs = py_utils.HasShape(sfrnn_outputs,
                                            [slen, batch, output_dim])
          return sess.run(sfrnn_outputs)
        else:
          sfrnn_outputs, sfrnn_final = sfrnn.FPropDefaultTheta(inputs, paddings)
          sfrnn_outputs = py_utils.HasShape(sfrnn_outputs,
                                            [slen, batch, output_dim])
          return sess.run([sfrnn_outputs, sfrnn_final])

  def _testStackedFRNNGradHelper(self, cls, bi_directional=False):
    trailing_pad_len = 2
    dtype = tf.float64
    batch = 3
    dims = 16
    slen = 10 + trailing_pad_len
    num_layers = 4
    with self.session(use_gpu=True, graph=tf.Graph()) as sess:
      params = rnn_cell.LSTMCellSimple.Params()
      params.name = 'lstm'
      params.output_nonlinearity = True
      params.dtype = dtype
      params.params_init = py_utils.WeightInit.Uniform(1.24, 429891685)
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      params.vn.seed = 2938482
      params.vn.scale = 0.1
      params.num_input_nodes = dims
      params.num_output_nodes = dims // 2 if bi_directional else dims

      sfrnn_params = cls.Params()
      sfrnn_params.name = 'sfrnn'
      sfrnn_params.dtype = dtype
      sfrnn_params.cell_tpl = params
      sfrnn_params.num_layers = num_layers
      sfrnn_params.skip_start = 2
      with tf.name_scope('sfrnn'):
        sfrnn = sfrnn_params.cls(sfrnn_params)

      np.random.seed(12345)
      inputs = tf.constant(np.random.uniform(size=(slen, batch, dims)), dtype)
      paddings = np.zeros([slen, batch, 1])
      paddings[-trailing_pad_len:, :] = 1.0
      paddings[-trailing_pad_len - 3:-trailing_pad_len - 1, :] = 1.0
      paddings = tf.constant(paddings, dtype)

      if bi_directional:
        sfrnn_outputs = sfrnn.FPropDefaultTheta(inputs, paddings)
        loss = tf.reduce_sum(sfrnn_outputs)
      else:
        sfrnn_outputs, sfrnn_final = sfrnn.FPropDefaultTheta(inputs, paddings)
        loss = tf.reduce_sum(sfrnn_outputs)
        for fin in sfrnn_final.rnn:
          loss += tf.reduce_sum(fin.m) + tf.reduce_sum(fin.c)
      xs = sfrnn.vars.Flatten() + [inputs]
      dxs = tf.gradients(loss, xs)

      # Compares the sym grad against the numeric grads.
      tf.global_variables_initializer().run()
      grad_step = 17
      sym_grads = sess.run(dxs)
      sym_grads = [test_utils.PickEveryN(_, grad_step) for _ in sym_grads]
      num_grads = [
          test_utils.PickEveryN(
              test_utils.ComputeNumericGradient(
                  sess, loss, v, delta=1e-4, step=grad_step), grad_step)
          for v in xs
      ]
      for (sym, num) in zip(sym_grads, num_grads):
        self.assertFalse(np.any(np.isnan(sym)))
        self.assertFalse(np.any(np.isnan(num)))
        print('max = ', np.max(np.abs(sym)))
        self.assertAllClose(sym, num)


class LayersTest(LayersTestBase):

  def testIdentitySeqLayer(self):
    with self.session(use_gpu=False) as sess:
      rnn_params = rnn_layers.IdentitySeqLayer.Params()
      rnn_params.name = 'no_op'
      rnn = rnn_params.cls(rnn_params)

      np.random.seed(12345)
      inputs_sequence = []
      paddings_sequence = []
      for _ in range(5):
        inputs_sequence.append(
            tf.constant(np.random.uniform(size=(3, 2)), tf.float32))
        paddings_sequence.append(tf.zeros([3, 1]))

      paddings_sequence[-1] = tf.constant([[1.0], [1.0], [1.0]])
      paddings_sequence[-2] = tf.constant([[1.0], [1.0], [1.0]])

      inputs, paddings = tf.stack(inputs_sequence), tf.stack(paddings_sequence)
      outputs = rnn.FPropFullSequence(rnn.theta, inputs, paddings)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()

      inputs_v, outputs_v = sess.run([inputs, outputs])
      self.assertAllEqual(inputs_v, outputs_v)

  def testRNN(self):
    with self.session(use_gpu=False) as sess:
      params = rnn_cell.LSTMCellSimple.Params()
      params.name = 'lstm'
      params.output_nonlinearity = True
      params.params_init = py_utils.WeightInit.Uniform(1.24, 429891685)
      params.num_input_nodes = 2
      params.num_output_nodes = 2

      rnn_params = rnn_layers.RNN.Params()
      rnn_params.name = 'rnn'
      rnn_params.vn.global_vn = True
      rnn_params.vn.per_step_vn = False
      rnn_params.vn.seed = 2938482
      rnn_params.vn.scale = 0.1
      rnn_params.cell = params
      rnn_params.sequence_length = 10
      rnn = rnn_layers.RNN(rnn_params)

      np.random.seed(12345)
      inputs_sequence = []
      paddings_sequence = []
      for _ in range(rnn_params.sequence_length):
        inputs_sequence.append(
            tf.constant(np.random.uniform(size=(3, 2)), tf.float32))
        paddings_sequence.append(tf.zeros([3, 1]))

      paddings_sequence[-1] = tf.constant([[1.0], [1.0], [1.0]])
      paddings_sequence[-2] = tf.constant([[1.0], [1.0], [1.0]])

      inputs, paddings = tf.stack(inputs_sequence), tf.stack(paddings_sequence)
      outputs, final = rnn.FPropDefaultTheta(inputs, paddings)

      outputs *= paddings
      sum_outputs = tf.reduce_sum(outputs, axis=0)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()

      actual = sess.run(py_utils.NestedMap(sum=sum_outputs, **final))

      # In case this test ever breaks, you can uncomment the line below, copy
      # out the golden values, and then comment out the line below again.
      #
      sum_expected = [[-0.46614376, 0.86599183], [-0.63463444, 0.57043159],
                      [-0.64659989, 0.72516292]]
      c_expected = [[-0.63455635, 0.76446551], [-0.59985822, 0.6631192],
                    [-0.63043576, 0.77522433]]
      m_expected = [[-0.23307188, 0.43299592], [-0.31731722, 0.2852158],
                    [-0.32329994, 0.36258146]]
      self.assertAllClose(sum_expected, actual.sum)
      self.assertAllClose(m_expected, actual.m)
      self.assertAllClose(c_expected, actual.c)

  def _CreateCuDNNLSTMParams(self,
                             input_nodes,
                             cell_nodes,
                             dtype=tf.float32,
                             is_eval=False):
    params = rnn_layers.CuDNNLSTM.Params()
    params.name = 'cudnn_lstm'
    params.dtype = dtype
    params.params_init = py_utils.WeightInit.Uniform(1.24, 429891685)
    params.vn.global_vn = False
    params.vn.per_step_vn = False
    params.is_eval = is_eval
    params.cell = rnn_cell.LSTMCellSimple.Params()
    params.cell.name = 'lstm_cell'
    params.cell.num_input_nodes = input_nodes
    params.cell.num_output_nodes = cell_nodes
    params.cell.cell_value_cap = None
    params.cell.forget_gate_bias = 0
    return params

  @unittest.skipUnless(tf.test.is_built_with_cuda(), 'Only available on GPU.')
  def testCuDNNLSTM(self):
    batch_size = 3
    seq_length = 10
    input_nodes = 4
    cell_nodes = 2

    np.random.seed(12345)
    inputs_v = np.random.uniform(size=(seq_length, batch_size, input_nodes))
    paddings_v = np.zeros((seq_length, batch_size, 1))
    model_var_v = np.random.uniform(
        size=((input_nodes + cell_nodes) * 4 * cell_nodes + 8 * cell_nodes))

    def _CreateLayer(is_eval):
      params = self._CreateCuDNNLSTMParams(
          input_nodes, cell_nodes, is_eval=is_eval)
      rnn = rnn_layers.CuDNNLSTM(params)
      inputs = tf.constant(inputs_v, dtype=params.dtype)
      paddings = tf.constant(paddings_v, dtype=params.dtype)
      outputs, final = rnn.FPropDefaultTheta(inputs, paddings)
      # Set outputs of padding inputs to 0. for comparison.
      outputs *= 1 - paddings

      if not is_eval:
        all_vars = tf.get_collection('CuDNNLSTM_vars')
        assert len(all_vars) == 1
        model_var_init = tf.assign(all_vars[0], model_var_v)
        return model_var_init, outputs, final
      else:
        return outputs, final

    # Train graph
    with tf.Graph().as_default() as g:
      with self.session(use_gpu=True, graph=g) as sess:
        tf.set_random_seed(87654321)
        init_op, outputs, final = _CreateLayer(is_eval=False)  # pylint:disable=unbalanced-tuple-unpacking
        saver = tf.train.Saver()

        # Initialize all the variables, and then run one step.
        tf.global_variables_initializer().run()
        init_op.op.run()

        save_path = os.path.join(self.get_temp_dir(), 'cudnn-lstm-test')
        saver.save(sess, save_path)

        (cudnn_outputs_v, cudnn_m_v,
         cudnn_c_v) = sess.run([outputs, final.m, final.c])

    # Eval graph
    with tf.Graph().as_default() as g:
      with self.session(use_gpu=False, graph=g) as sess:
        tf.set_random_seed(87654321)
        outputs, final = _CreateLayer(is_eval=True)  # pylint:disable=unbalanced-tuple-unpacking
        saver = tf.train.Saver()

        # Initialize all the variables, and then run one step.
        tf.global_variables_initializer().run()
        saver.restore(sess, save_path)

        (rnn_outputs_v, rnn_m_v,
         rnn_c_v) = sess.run([outputs, final.m, final.c])

    # CuDNNLSTM train and eval mode are equivalent and its checkpoints can
    # be consumed by FRNN in eval mode.
    self.assertAllClose(rnn_outputs_v, cudnn_outputs_v)
    self.assertAllClose(rnn_m_v, np.squeeze(cudnn_m_v))
    self.assertAllClose(rnn_c_v, np.squeeze(cudnn_c_v))

  @unittest.skipUnless(tf.test.is_built_with_cuda(), 'Only available on GPU.')
  def testCuDNNLSTMGradientChecker(self):
    batch_size = 3
    seq_length = 10
    input_nodes = 4
    cell_nodes = 2
    dtype = tf.float64

    np.random.seed(12345)
    inputs_v = np.random.uniform(size=(seq_length, batch_size, input_nodes))
    paddings_v = np.random.uniform(size=(seq_length, batch_size, 1))
    paddings_v[-1] = np.expand_dims(
        np.expand_dims(np.array([1.] * (batch_size - 1) + [0.]), 0), 2)
    paddings_v[-2] = np.zeros((1, batch_size, 1))

    with tf.Graph().as_default() as g:
      with self.session(use_gpu=True) as sess:
        tf.set_random_seed(87654321)
        params = self._CreateCuDNNLSTMParams(
            input_nodes, cell_nodes, dtype=dtype, is_eval=False)
        rnn = rnn_layers.CuDNNLSTM(params)
        inputs = tf.constant(inputs_v, dtype=params.dtype)
        paddings = tf.constant(paddings_v, dtype=params.dtype)

        with tf.device('/gpu:0'):
          outputs, final = rnn.FPropDefaultTheta(inputs, paddings)
        all_vars = tf.get_collection('CuDNNLSTM_vars')
        assert len(all_vars) == 1

        loss = tf.reduce_sum(outputs * paddings) + tf.add_n(
            [tf.reduce_sum(x) for x in final.Flatten()])
        grads = tf.gradients(loss, all_vars)

        # Initialize all the variables, and then run one step.
        tf.global_variables_initializer().run()

        symbolic_grads = [gd.eval() for gd in grads]
        numerical_grads = []
        for v in all_vars:
          numerical_grads.append(
              test_utils.ComputeNumericGradient(sess, loss, v))
        for x, y in zip(symbolic_grads, numerical_grads):
          self.assertAllClose(x, y)

  def testRNNGradientChecker(self):
    with self.session(use_gpu=False) as sess:
      params = rnn_cell.LSTMCellSimple.Params()
      params.name = 'lstm'
      params.output_nonlinearity = True
      params.params_init = py_utils.WeightInit.Uniform(1.24, 429891685)
      params.num_input_nodes = 2
      params.num_output_nodes = 2

      rnn_params = rnn_layers.RNN.Params()
      rnn_params.name = 'rnn'
      rnn_params.dtype = tf.float64
      rnn_params.vn.scale = 0.1
      rnn_params.vn.global_vn = False
      rnn_params.vn.per_step_vn = False
      rnn_params.cell = params
      rnn_params.sequence_length = 10
      rnn = rnn_layers.RNN(rnn_params)

      np.random.seed(12345)
      inputs_sequence = []
      paddings_sequence = []
      for _ in range(rnn_params.sequence_length):
        inputs_sequence.append(
            tf.constant(np.random.uniform(size=(3, 2)), tf.float64))
        paddings_sequence.append(tf.zeros([3, 1], dtype=tf.float64))

      paddings_sequence[-1] = tf.constant(
          [[1.0], [1.0], [1.0]], dtype=tf.float64)
      paddings_sequence[-2] = tf.constant(
          [[1.0], [1.0], [1.0]], dtype=tf.float64)

      inputs, paddings = tf.stack(inputs_sequence), tf.stack(paddings_sequence)
      outputs, final = rnn.FPropDefaultTheta(inputs, paddings)
      outputs *= paddings
      sum_outputs = tf.reduce_sum(outputs, axis=0)

      loss = tf.reduce_sum(sum_outputs) + tf.reduce_sum(final.m + final.c)
      all_vars = tf.get_collection('LSTMCellSimple_vars')
      assert len(all_vars) == 2

      grads = tf.gradients(loss, all_vars)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()

      symbolic_grads = [gd.eval() for gd in grads]
      numerical_grads = []
      for v in all_vars:
        numerical_grads.append(test_utils.ComputeNumericGradient(sess, loss, v))
      for x, y in zip(symbolic_grads, numerical_grads):
        self.assertAllClose(x, y)

  def testRNNReversed(self):
    """Test an RNN layer with reverse=true.

    This should yield the same output as feeding reversed input into
    the same RNN with reverse=false (except the output is in reversed order).
    """

    timesteps = 10
    padding_steps = 2
    batch_size = 2
    depth = 3
    with self.session(use_gpu=True) as sess:
      lstm_params = rnn_cell.LSTMCellSimple.Params()
      lstm_params.output_nonlinearity = True
      lstm_params.num_input_nodes = depth
      lstm_params.num_output_nodes = depth

      rnn_params = rnn_layers.RNN.Params()
      rnn_params.vn.global_vn = False
      rnn_params.vn.per_step_vn = False
      rnn_params.vn.seed = 2938482
      rnn_params.vn.scale = 0.1
      rnn_params.cell = lstm_params
      rnn_params.sequence_length = timesteps

      fwd_rnn_params = rnn_params.Copy()
      fwd_rnn_params.name = 'fwd'
      fwd_rnn_params.cell.name = 'fwd_lstm'
      fwd_rnn = rnn_layers.RNN(fwd_rnn_params)

      bak_rnn_params = rnn_params.Copy()
      bak_rnn_params.name = 'bak'
      bak_rnn_params.cell.name = 'bak_lstm'
      bak_rnn_params.reverse = True
      bak_rnn = rnn_layers.RNN(bak_rnn_params)

      # Create 8 timesteps of random input, 2 timesteps of zeros, and paddings
      # to match.
      fwd_inputs = tf.constant(
          np.concatenate(
              (np.random.uniform(
                  size=(timesteps - padding_steps, batch_size, depth)),
               np.zeros(shape=(padding_steps, batch_size, depth))),
              axis=0), tf.float32)
      fwd_paddings = tf.concat(
          (tf.zeros(shape=(timesteps - padding_steps, batch_size, depth)),
           tf.ones(shape=(padding_steps, batch_size, depth))),
          axis=0)
      bak_inputs = tf.reverse(fwd_inputs, [0])
      bak_paddings = tf.reverse(fwd_paddings, [0])

      # Run the forward rnn with reversed inputs
      reversed_outputs, _ = fwd_rnn.FProp(fwd_rnn.theta, bak_inputs,
                                          bak_paddings)
      reversed_outputs = tf.reverse(reversed_outputs, [0])

      # Run the backward rnn with forward inputs.  Note we reuse the fwd_rnn
      # theta so the results should match
      bak_outputs, _ = bak_rnn.FProp(fwd_rnn.theta, fwd_inputs, fwd_paddings)

      tf.global_variables_initializer().run()
      actual_reversed_outputs, actual_bak_outputs = sess.run(
          [reversed_outputs, bak_outputs])
      self.assertAllClose(actual_reversed_outputs, actual_bak_outputs)

  def testRNNWithConvLSTMCell(self):
    with self.session(use_gpu=False) as sess:
      params = rnn_cell.ConvLSTMCell.Params()
      params.name = 'conv_lstm'
      params.output_nonlinearity = True
      params.params_init = py_utils.WeightInit.Uniform(1.24, 429891685)
      params.inputs_shape = [None, 4, 2, 3]
      params.cell_shape = [None, 4, 2, 2]
      params.filter_shape = [3, 2]

      rnn_params = rnn_layers.RNN.Params()
      rnn_params.name = 'rnn'
      rnn_params.vn.global_vn = True
      rnn_params.vn.per_step_vn = False
      rnn_params.vn.seed = 2938482
      rnn_params.vn.scale = 0.1
      rnn_params.cell = params
      rnn_params.sequence_length = 10
      rnn = rnn_layers.RNN(rnn_params)

      np.random.seed(12345)
      inputs_sequence = []
      paddings_sequence = []
      for _ in range(rnn_params.sequence_length):
        inputs_sequence.append(
            tf.constant(np.random.uniform(size=(3, 4, 2, 3)), tf.float32))
        paddings_sequence.append(tf.zeros([3, 1]))

      paddings_sequence[-1] = tf.constant([[1.0], [1.0], [1.0]])
      paddings_sequence[-2] = tf.constant([[1.0], [1.0], [1.0]])

      inputs, paddings = tf.stack(inputs_sequence), tf.stack(paddings_sequence)
      outputs, final = rnn.FPropDefaultTheta(inputs, paddings)
      outputs *= tf.reshape(1.0 - paddings,
                            [rnn_params.sequence_length, -1, 1, 1, 1])

      sum_outputs = tf.reduce_sum(tf.reduce_sum(outputs, axis=0), [1, 2, 3])
      sum_final_m = tf.reduce_sum(final.m, [1, 2, 3])
      sum_final_c = tf.reduce_sum(final.c, [1, 2, 3])

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()
      actual = sess.run(
          py_utils.NestedMap(sum=sum_outputs, m=sum_final_m, c=sum_final_c))

      # In case this test ever breaks, you can uncomment the line below, copy
      # out the golden values, and then comment out the line below again.
      #
      # print('sum_outputs', np.array_repr(actual.sum))
      # print('final_m', np.array_repr(actual.m))
      # print('final_c', np.array_repr(actual.c))

      sum_expected = [1.59282923, 0.85964835, -1.09797788]
      m_expected = [0.32634005, 0.38098311, 0.23331133]
      c_expected = [2.72942424, 2.72087693, 2.97179723]
      self.assertAllClose(sum_expected, actual.sum)
      self.assertAllClose(m_expected, actual.m)
      self.assertAllClose(c_expected, actual.c)

  def _testFRNNWithConvLSTMCell(self):
    with self.session(use_gpu=True) as sess:
      params = rnn_cell.ConvLSTMCell.Params()
      params.name = 'conv_lstm'
      params.output_nonlinearity = True
      params.params_init = py_utils.WeightInit.Uniform(1.24, 429891685)
      params.inputs_shape = [None, 4, 2, 3]
      params.cell_shape = [None, 4, 2, 2]
      params.filter_shape = [3, 2]

      rnn_params = rnn_layers.RNN.Params()
      rnn_params.name = 'rnn'
      rnn_params.vn.global_vn = True
      rnn_params.vn.per_step_vn = False
      rnn_params.vn.seed = 2938482
      rnn_params.vn.scale = 0.1
      rnn_params.cell = params
      rnn_params.sequence_length = 10

      with tf.variable_scope('rnn'):
        rnn = rnn_layers.RNN(rnn_params)

      np.random.seed(12345)
      inputs_sequence = []
      paddings_sequence = []
      for _ in range(rnn_params.sequence_length):
        inputs_sequence.append(
            tf.constant(np.random.uniform(size=(3, 4, 2, 3)), tf.float32))
        paddings_sequence.append(tf.zeros([3, 1]))

      paddings_sequence[-1] = tf.constant([[1.0], [1.0], [1.0]])
      paddings_sequence[-2] = tf.constant([[1.0], [1.0], [1.0]])

      inputs, paddings = tf.stack(inputs_sequence), tf.stack(paddings_sequence)
      outputs, final = rnn.FPropDefaultTheta(inputs, paddings)
      outputs *= tf.reshape(1.0 - paddings,
                            [rnn_params.sequence_length, -1, 1, 1, 1])
      rnn_outputs = outputs

      frnn_params = rnn_layers.FRNN.Params()
      frnn_params.name = 'frnn'
      frnn_params.cell = params
      frnn_params.vn = rnn_params.vn

      with tf.variable_scope('frnn'):
        frnn = frnn_params.cls(frnn_params)

      frnn_outputs, frnn_final = frnn.FPropDefaultTheta(
          tf.stack(inputs_sequence), tf.stack(paddings_sequence))
      paddings = tf.stack(paddings_sequence)
      frnn_outputs *= tf.reshape(1.0 - paddings,
                                 tf.concat([tf.shape(paddings), [1, 1]], 0))

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()

      (rnn_outputs_v, rnn_final_v, frnn_outputs_v,
       frnn_final_v) = sess.run([rnn_outputs, final, frnn_outputs, frnn_final])

      self.assertAllClose(rnn_outputs_v, frnn_outputs_v)
      self.assertAllClose(rnn_final_v.m, frnn_final_v.m)
      self.assertAllClose(rnn_final_v.c, frnn_final_v.c)

  def testFRNNWithConvLSTMCell4(self):
    self._testFRNNWithConvLSTMCell()

  def testRNNWithConvLSTMCellGradientChecker(self):
    with self.session(use_gpu=True) as sess:
      params = rnn_cell.ConvLSTMCell.Params()
      params.name = 'conv_lstm'
      params.output_nonlinearity = True
      params.params_init = py_utils.WeightInit.Uniform(1.24, 429891685)
      params.inputs_shape = [None, 4, 2, 3]
      params.cell_shape = [None, 4, 2, 2]
      params.filter_shape = [3, 2]

      rnn_params = rnn_layers.RNN.Params()
      rnn_params.name = 'rnn'
      rnn_params.vn.global_vn = False
      rnn_params.vn.per_step_vn = False
      rnn_params.vn.seed = 2938482
      rnn_params.vn.scale = 0.1
      rnn_params.cell = params
      rnn_params.sequence_length = 10
      rnn = rnn_layers.RNN(rnn_params)

      np.random.seed(12345)
      inputs_sequence = []
      paddings_sequence = []
      for _ in range(rnn_params.sequence_length):
        inputs_sequence.append(
            tf.constant(np.random.uniform(size=(3, 4, 2, 3)), tf.float32))
        paddings_sequence.append(tf.zeros([3, 1]))

      paddings_sequence[-1] = tf.constant([[1.0], [1.0], [1.0]])
      paddings_sequence[-2] = tf.constant([[1.0], [1.0], [1.0]])

      inputs, paddings = tf.stack(inputs_sequence), tf.stack(paddings_sequence)
      outputs, final = rnn.FPropDefaultTheta(inputs, paddings)
      outputs *= tf.reshape(1.0 - paddings,
                            [rnn_params.sequence_length, -1, 1, 1, 1])
      loss = tf.reduce_sum(tf.reduce_sum(
          outputs, axis=0)) + tf.reduce_sum(final.m + final.c)
      all_vars = tf.trainable_variables()
      assert len(all_vars) == 2

      grads = tf.gradients(loss, all_vars)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()

      symbolic_grads = [gd.eval() for gd in grads]
      numerical_grads = []
      for v in all_vars:
        numerical_grads.append(test_utils.ComputeNumericGradient(sess, loss, v))
      for x, y in zip(symbolic_grads, numerical_grads):
        self.assertAllClose(x, y, rtol=0.1, atol=0.1)

  def _testFRNNWithConvLSTMCellGradientChecker(self):
    with self.session(use_gpu=True) as sess:
      params = rnn_cell.ConvLSTMCell.Params()
      params.name = 'conv_lstm'
      params.output_nonlinearity = True
      params.params_init = py_utils.WeightInit.Uniform(1.24, 429891685)
      params.inputs_shape = [None, 4, 2, 3]
      params.cell_shape = [None, 4, 2, 2]
      params.filter_shape = [3, 2]

      frnn_params = rnn_layers.FRNN.Params()
      frnn_params.name = 'rnn'
      frnn_params.vn.global_vn = False
      frnn_params.vn.per_step_vn = False
      frnn_params.vn.seed = 2938482
      frnn_params.vn.scale = 0.1
      frnn_params.cell = params
      frnn = rnn_layers.FRNN(frnn_params)

      np.random.seed(12345)
      inputs_sequence = tf.constant(
          np.random.uniform(size=(10, 3, 4, 2, 3)), tf.float32)
      paddings = inplace_ops.inplace_update(
          tf.zeros([10, 3, 1]), 1, [[1.0], [0.0], [1.0]])

      outputs, final = frnn.FPropDefaultTheta(inputs_sequence, paddings)
      outputs *= tf.reshape(paddings, [10, 3, 1, 1, 1])
      loss = tf.reduce_sum(outputs) + tf.reduce_sum(final.m + final.c)
      all_vars = tf.trainable_variables()
      assert len(all_vars) == 2

      grads = tf.gradients(loss, all_vars)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()

      symbolic_grads = [gd.eval() for gd in grads]
      numerical_grads = []
      for v in all_vars:
        numerical_grads.append(test_utils.ComputeNumericGradient(sess, loss, v))
      for x, y in zip(symbolic_grads, numerical_grads):
        self.assertAllClose(x, y, rtol=0.1, atol=0.1)

  def testFRNNWithConvLSTMCellGradientChecker(self):
    self._testFRNNWithConvLSTMCellGradientChecker()

  def testFRNNWithLSTMCellSimpleDeterministicGradientChecker(self):
    with self.session(use_gpu=True) as sess:
      params = rnn_cell.LSTMCellSimpleDeterministic.Params()
      params.name = 'conv_lstm'
      params.output_nonlinearity = True
      params.params_init = py_utils.WeightInit.Uniform(1.24, 429891685)
      params.zo_prob = 0.25
      params.num_input_nodes = 4
      params.num_output_nodes = 6
      params.dtype = tf.float64

      frnn_params = rnn_layers.FRNN.Params()
      frnn_params.dtype = tf.float64
      frnn_params.name = 'rnn'
      frnn_params.cell = params
      old_enable_asserts = FLAGS.enable_asserts
      FLAGS.enable_asserts = False
      frnn = rnn_layers.FRNN(frnn_params)
      FLAGS.enable_asserts = old_enable_asserts

      np.random.seed(12345)
      inputs_sequence = tf.constant(
          np.random.uniform(size=(10, 3, 4)), tf.float64)
      paddings = inplace_ops.inplace_update(
          tf.zeros([10, 3, 1], tf.float64), 1, [[1.0], [0.0], [1.0]])

      outputs, _ = frnn.FPropDefaultTheta(inputs_sequence, paddings)
      outputs *= (1.0 - tf.reshape(paddings, [10, 3, 1]))
      loss = tf.reduce_sum(outputs)

      all_vars = tf.trainable_variables()
      assert len(all_vars) == 2

      grads = tf.gradients(loss, all_vars)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()

      symbolic_grads = [gd.eval() for gd in grads]
      numerical_grads = []
      for v in all_vars:
        numerical_grads.append(test_utils.ComputeNumericGradient(sess, loss, v))
      for x, y in zip(symbolic_grads, numerical_grads):
        self.assertAllClose(x, y, rtol=0.00001, atol=0.00001)

  def _testFRNNHelper(self, config=None):
    dtype = tf.float32
    batch = 3
    dims = 16
    slen = 10
    with self.session(use_gpu=True, config=config) as sess:
      params = rnn_cell.LSTMCellSimple.Params()
      params.name = 'lstm'
      params.output_nonlinearity = True
      params.dtype = dtype
      params.params_init = py_utils.WeightInit.Uniform(1.24, 429891685)
      params.vn.global_vn = True
      params.vn.per_step_vn = False
      params.vn.seed = 2938482
      params.vn.scale = 0.1
      params.num_input_nodes = dims
      params.num_output_nodes = dims

      frnn_params = rnn_layers.FRNN.Params()
      frnn_params.name = 'frnn'
      frnn_params.dtype = dtype
      frnn_params.cell = params
      with tf.variable_scope('frnn'):
        frnn = rnn_layers.FRNN(frnn_params)
        AddTimestepAccumulator(frnn.cell)

      rnn_params = rnn_layers.RNN.Params()
      rnn_params.name = 'rnn'
      rnn_params.dtype = dtype
      rnn_params.sequence_length = slen
      rnn_params.cell = params
      with tf.variable_scope('rnn'):
        rnn = rnn_layers.RNN(rnn_params)

      np.random.seed(12345)
      inputs = tf.constant(
          np.random.uniform(size=(slen, batch, dims)), tf.float32)
      paddings = np.zeros([slen, batch, 1])
      paddings[-3:-1, :] = 1.0
      paddings = tf.constant(paddings, tf.float32)

      frnn_outputs, frnn_final = frnn.FPropDefaultTheta(inputs, paddings)
      rnn_outputs, rnn_final = rnn.FPropDefaultTheta(
          tf.unstack(inputs), tf.unstack(paddings))

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()
      frnn_out, frnn_final, rnn_out, rnn_final = sess.run(
          [frnn_outputs, frnn_final, rnn_outputs, rnn_final])
      self.assertAllClose(frnn_out, rnn_out)
      self.assertAllClose(frnn_final.m, rnn_final.m)
      self.assertAllClose(frnn_final.c, rnn_final.c)

  def testFRNNNoInline(self):
    self._testFRNNHelper(py_utils.SessionConfig(inline=False))

  def testFRNNInline(self):
    self._testFRNNHelper(py_utils.SessionConfig(inline=True))

  def _testFRNNGradHelper(self, config):
    dtype = tf.float64  # More stable using float64.
    batch = 3
    dims = 16
    slen = 10
    with self.session(use_gpu=True, config=config) as sess:
      params = rnn_cell.LSTMCellSimple.Params()
      params.name = 'lstm'
      params.output_nonlinearity = True
      params.params_init = py_utils.WeightInit.Uniform(0.02, 429891685)
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      params.num_input_nodes = dims
      params.num_output_nodes = dims
      params.dtype = dtype

      frnn_params = rnn_layers.FRNN.Params()
      frnn_params.name = 'frnn'
      frnn_params.dtype = dtype
      frnn_params.cell = params
      frnn = rnn_layers.FRNN(frnn_params)
      AddTimestepAccumulator(frnn.cell)
      w, b = frnn.theta.cell.wm, frnn.theta.cell.b

      np.random.seed(12345)
      inputs = tf.constant(
          np.random.uniform(-0.02, 0.02, size=(slen, batch, dims)), dtype)
      paddings = np.zeros([slen, batch, 1])
      paddings[-3:-1, :] = 1.0
      paddings = tf.constant(paddings, dtype)
      frnn_outputs, frnn_final = frnn.FPropDefaultTheta(inputs, paddings)
      loss = tf.reduce_sum(frnn_outputs) + tf.reduce_sum(
          frnn_final.m) + tf.reduce_sum(frnn_final.c)
      dw, db, dinputs = tf.gradients(loss, [w, b, inputs])

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()
      grad_step = 7
      sym_grads = sess.run([db, dw, dinputs])
      sym_grads = [test_utils.PickEveryN(_, grad_step) for _ in sym_grads]
      num_grads = [
          test_utils.PickEveryN(
              test_utils.ComputeNumericGradient(
                  sess, loss, v, delta=1e-4, step=grad_step), grad_step)
          for v in [b, w, inputs]
      ]
      for (sym, num) in zip(sym_grads, num_grads):
        self.assertFalse(np.any(np.isnan(sym)))
        self.assertFalse(np.any(np.isnan(num)))
        print('max = ', np.max(np.abs(sym)))
        self.assertAllClose(sym, num)

  def testFRNNGradNoInline(self):
    self._testFRNNGradHelper(py_utils.SessionConfig(inline=False))

  def testFRNNGradInline(self):
    self._testFRNNGradHelper(py_utils.SessionConfig(inline=True))

  def testStackedFRNNDropout(self):
    v1_out, _ = self._testStackedFRNNHelper(
        rnn_layers.StackedFRNNLayerByLayer,
        tf.float32,
        trailing_pad_len=0,
        keep_prob=0.5)
    if tf.test.is_gpu_available():
      rtol = 1e-5
    else:
      rtol = 1e-6
    self.assertAllClose([175.974121], [np.sum(v1_out * v1_out)], rtol=rtol)

  def testStackedFRNNInputOutputDims(self):
    v1_out, _ = self._testStackedFRNNHelper(
        rnn_layers.StackedFRNNLayerByLayer,
        tf.float32,
        trailing_pad_len=0,
        keep_prob=0.5,
        input_dim=5,
        output_dim=7)
    if tf.test.is_gpu_available():
      rtol = 1e-5
    else:
      rtol = 1e-6
    self.assertAllClose([32.743263], [np.sum(v1_out * v1_out)], rtol=rtol)

  def testStackedFRNNLayerByLayerGrad(self):
    self._testStackedFRNNGradHelper(rnn_layers.StackedFRNNLayerByLayer)

  def testStackedBiFRNNDropout(self):
    v1_out = self._testStackedFRNNHelper(
        rnn_layers.StackedBiFRNNLayerByLayer,
        tf.float32,
        trailing_pad_len=0,
        keep_prob=0.5,
        bi_directional=True)
    if tf.test.is_gpu_available():
      rtol = 1e-5
    else:
      rtol = 1e-6
    self.assertAllClose([305.774384], [np.sum(v1_out * v1_out)], rtol=rtol)

  def testStackedBiFRNNInputOutputDims(self):
    v1_out = self._testStackedFRNNHelper(
        rnn_layers.StackedBiFRNNLayerByLayer,
        tf.float32,
        trailing_pad_len=0,
        keep_prob=0.5,
        bi_directional=True,
        input_dim=5,
        output_dim=8)
    if tf.test.is_gpu_available():
      rtol = 1e-5
    else:
      rtol = 1e-6
    self.assertAllClose([8.116007], [np.sum(v1_out * v1_out)], rtol=rtol)

  def testStackedBiFRNNLayerByLayerGrad(self):
    self._testStackedFRNNGradHelper(
        rnn_layers.StackedBiFRNNLayerByLayer, bi_directional=True)

  def _testBidirectionalFRNNHelper(self,
                                   trailing_pad_len=0,
                                   cluster_params=None):
    batch = 3
    dims = 16
    slen = 10 + trailing_pad_len
    with self.session(
        use_gpu=True, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      params = rnn_cell.LSTMCellSimple.Params()
      params.name = 'lstm_forward'
      params.output_nonlinearity = True
      params.params_init = py_utils.WeightInit.Uniform(0.02, 429891685)
      params.vn.global_vn = True
      params.vn.per_step_vn = False
      params.vn.seed = 2938482
      params.vn.scale = 0.1
      params.num_input_nodes = dims
      params.num_output_nodes = dims
      lstm_forward = params.Copy()
      params.name = 'lstm_backward'
      params.params_init = py_utils.WeightInit.Uniform(0.02, 83820209838)
      lstm_backward = params.Copy()

      frnn_params = rnn_layers.BidirectionalFRNN.Params()
      frnn_params.name = 'bifrnn'
      frnn_params.fwd = lstm_forward.Copy()
      frnn_params.bak = lstm_backward.Copy()
      with tf.variable_scope('frnn'):
        frnn = rnn_layers.BidirectionalFRNN(frnn_params)

      rnn_params = rnn_layers.BidirectionalRNN.Params()
      rnn_params.name = 'rnn'
      rnn_params.fwd = lstm_forward.Copy()
      rnn_params.bak = lstm_backward.Copy()
      rnn_params.sequence_length = slen

      with cluster_factory.Cluster(cluster_params if cluster_params else
                                   cluster_factory.Cluster.Params()):
        with tf.variable_scope('rnn'):
          rnn = rnn_layers.BidirectionalRNN(rnn_params)

          np.random.seed(12345)
          inputs = tf.constant(
              np.random.uniform(size=(slen, batch, dims)), tf.float32)
          paddings = np.zeros([slen, batch, 1])
          paddings[-trailing_pad_len:, :] = 1.0
          paddings[-trailing_pad_len - 3:-trailing_pad_len - 1, :] = 1.0
          paddings = tf.constant(paddings, tf.float32)

          frnn_outputs = frnn.FPropDefaultTheta(inputs, paddings)
          rnn_outputs = rnn.FPropDefaultTheta(
              tf.unstack(inputs), tf.unstack(paddings))

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()
      rnn_outputs_val, frnn_outputs_val = [
          x[:-trailing_pad_len] for x in sess.run([rnn_outputs, frnn_outputs])
      ]
      self.assertAllClose(rnn_outputs_val, frnn_outputs_val)

  def testBidirectionalFRNN(self):
    self._testBidirectionalFRNNHelper()

  def testBidirectionalFRNNTrailingPadding(self):
    self._testBidirectionalFRNNHelper(trailing_pad_len=2)

  def testBidirectionalFRNNSplit(self):
    cluster_params = cluster_factory.Cluster.Params()
    cluster_params.worker.Set(
        gpus_per_replica=2, devices_per_split=2, name='/job:localhost')
    self._testBidirectionalFRNNHelper(cluster_params=cluster_params)

  def _testBidirectionalFRNNGradientHelper(self):
    dtype = tf.float64  # More stable using float64.
    batch = 3
    dims = 16
    slen = 10
    with self.session(use_gpu=True) as sess:
      params = rnn_cell.LSTMCellSimple.Params()
      params.name = 'lstm_forward'
      params.output_nonlinearity = True
      params.params_init = py_utils.WeightInit.Uniform(0.02, 429891685)
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      params.vn.scale = 0.1
      params.dtype = dtype
      params.num_input_nodes = dims
      params.num_output_nodes = dims
      lstm_forward = params.Copy()
      params.name = 'lstm_backward'
      params.params_init = py_utils.WeightInit.Uniform(0.02, 83820209838)
      params.dtype = dtype
      lstm_backward = params.Copy()

      frnn_params = rnn_layers.BidirectionalFRNN.Params()
      frnn_params.name = 'bifrnn'
      frnn_params.dtype = dtype
      frnn_params.fwd = lstm_forward.Copy()
      frnn_params.bak = lstm_backward.Copy()
      frnn = rnn_layers.BidirectionalFRNN(frnn_params)
      w0, b0 = (frnn.theta.fwd_rnn.cell.wm, frnn.theta.fwd_rnn.cell.b)
      w1, b1 = (frnn.theta.bak_rnn.cell.wm, frnn.theta.bak_rnn.cell.b)

      np.random.seed(12345)
      inputs = tf.constant(np.random.uniform(size=(slen, batch, dims)), dtype)
      paddings = np.zeros([slen, batch, 1])
      paddings[-3:-1, :] = 1.0
      paddings = tf.constant(paddings, dtype)
      frnn_outputs = frnn.FPropDefaultTheta(inputs, paddings)
      loss = tf.reduce_sum(frnn_outputs)

      dw0, db0, dw1, db1, dinputs = tf.gradients(loss, [w0, b0, w1, b1, inputs])

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()
      grad_step = 13
      sym_grads = sess.run([dw0, db0, dw1, db1, dinputs])
      sym_grads = [test_utils.PickEveryN(_, grad_step) for _ in sym_grads]
      num_grads = [
          test_utils.PickEveryN(
              test_utils.ComputeNumericGradient(
                  sess, loss, v, delta=1e-4, step=grad_step), grad_step)
          for v in [w0, b0, w1, b1, inputs]
      ]
      for (sym, num) in zip(sym_grads, num_grads):
        self.assertFalse(np.any(np.isnan(sym)))
        self.assertFalse(np.any(np.isnan(num)))
        print('max = ', np.max(np.abs(sym)))
        self.assertAllClose(sym, num)

  def testBidirectionalFRNNGrad(self):
    self._testBidirectionalFRNNGradientHelper()

  def _MultiSourceFRNNWithAttentionInputs(self,
                                          single_source=False,
                                          single_source_length=True,
                                          dtype=tf.float32):
    np.random.seed(12345)
    if single_source:
      src_names = ['en']
      slens = [10]
      sdepths = [4]
    elif single_source_length:
      src_names = ['en1', 'en2', 'de']
      slens = [11, 10, 9]
      sdepths = [4, 4, 4]
    else:
      src_names = ['en1', 'en2', 'de']
      slens = [11, 10, 9]
      sdepths = [4, 4, 3]
    sbatch = 3
    tlen = 7
    tbatch = 6
    dims = 4

    src_encs = py_utils.NestedMap()
    src_paddings = py_utils.NestedMap()
    for sdepth, slen, sname in zip(sdepths, slens, src_names):
      src_encs[sname] = tf.constant(
          np.random.uniform(size=[slen, sbatch, sdepth]), dtype)
      src_paddings[sname] = tf.constant(np.zeros([slen, sbatch]), dtype)
    inputs = tf.constant(np.random.uniform(size=(tlen, tbatch, dims)), dtype)
    paddings = tf.constant(np.zeros([tlen, tbatch, 1]), dtype)
    return (src_encs, src_paddings, inputs, paddings)

  def _MultiSourceFRNNWithAttentionParams(self,
                                          single_source=False,
                                          single_source_length=True,
                                          dtype=tf.float32):
    dims = 4
    alt_depth = 3
    if single_source:
      src_names = ['en']
    else:
      src_names = ['en1', 'en2', 'de']

    p = rnn_cell.LSTMCellSimple.Params()
    p.name = 'lstm'
    p.dtype = dtype
    p.output_nonlinearity = True
    p.params_init = py_utils.WeightInit.Uniform(0.02, 429891685)
    p.vn.global_vn = False
    p.vn.per_step_vn = False
    p.num_input_nodes = dims * 2
    p.num_output_nodes = dims
    lstm_params = p

    p = attention.AdditiveAttention.Params()
    p.name = 'atten'
    p.dtype = dtype
    p.params_init = py_utils.WeightInit.Gaussian(0.1, 12345)
    p.source_dim = dims
    p.query_dim = dims
    p.hidden_dim = dims
    p.vn.global_vn = False
    p.vn.per_step_vn = False
    attention_tpl = p

    p = layers_with_attention.MergerLayer.Params()
    p.name = 'merger'
    p.dtype = dtype
    p.merger_op = ('mean' if single_source else 'atten')
    p.source_dim = dims
    p.query_dim = dims
    p.hidden_dim = dims
    merger_tpl = p

    p = rnn_layers.MultiSourceFRNNWithAttention.Params()
    p.name = 'msrc_frnn_with_atten'
    p.dtype = dtype
    p.cell = lstm_params
    p.attention_tpl = attention_tpl
    p.atten_merger = merger_tpl
    p.source_names = src_names

    if not single_source_length:
      de_atten = attention_tpl.Copy()
      de_atten.source_dim = alt_depth
      p.source_name_to_attention_params = {'de': de_atten}
      merger_tpl.pre_proj_input_dims = [dims, dims, alt_depth]
      merger_tpl.pre_proj_output_dim = dims
      merger_tpl.proj_tpl.batch_norm = False
      merger_tpl.proj_tpl.weight_norm = True

    return p

  def testMultiSourceFRNNWithAttention(self):
    with self.session(use_gpu=True) as sess:
      p = self._MultiSourceFRNNWithAttentionParams()
      msrc_frnn = p.cls(p)

      (src_encs, src_paddings, inputs,
       paddings) = self._MultiSourceFRNNWithAttentionInputs()
      a, m = msrc_frnn.FPropDefaultTheta(src_encs, src_paddings, inputs,
                                         paddings)
      msrc_frnn_out = tf.concat([a, m], 2)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()
      ys = sess.run([msrc_frnn_out])[0]
      self.assertEqual(ys.shape, (7, 6, 8))
      print(np.sum(ys, axis=(1, 2)), np.sum(ys, axis=(0, 1)),
            np.sum(ys, axis=(0, 2)))
      # pyformat: disable
      # pylint: disable=bad-whitespace
      self.assertAllClose(
          np.sum(ys, axis=(1, 2)), [
              11.87568951,  11.8436203 ,  11.80368233,  11.80167198,
              11.82034779,  11.80246162,  11.80818748
          ])
      self.assertAllClose(
          np.sum(ys, axis=(0, 1)), [
              21.41802788,  20.86244965,  21.48164749,  19.95701981,
              -0.54706949,   0.07046284,  -0.50449395,   0.0176318
          ])
      self.assertAllClose(
          np.sum(ys, axis=(0, 2)), [
              13.29822254,  14.01552773,  14.04851151,  13.28098106,
              14.05391502,  14.0585041
          ])
      # pyformat: enable
      # pylint: enable=bad-whitespace

  def testMultiSourceFRNNWithAttentionMultiDepth(self):
    with self.session(use_gpu=True) as sess:
      p = self._MultiSourceFRNNWithAttentionParams(single_source_length=False)
      msrc_frnn = p.cls(p)

      (src_encs, src_paddings, inputs, paddings
      ) = self._MultiSourceFRNNWithAttentionInputs(single_source_length=False)
      a, m = msrc_frnn.FPropDefaultTheta(src_encs, src_paddings, inputs,
                                         paddings)
      msrc_frnn_out = tf.concat([a, m], 2)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()
      ys = sess.run([msrc_frnn_out])[0]
      self.assertEqual(ys.shape, (7, 6, 8))
      print(np.sum(ys, axis=(1, 2)), np.sum(ys, axis=(0, 1)),
            np.sum(ys, axis=(0, 2)))
      # pyformat: disable
      # pylint: disable=bad-whitespace
      self.assertAllClose(
          np.sum(ys, axis=(1, 2)), [
              5.976197,  5.932313,  5.917447,  5.907898,  5.907385,  5.90272 ,
              5.890248
          ])
      self.assertAllClose(
          np.sum(ys, axis=(0, 1)), [
              2.635296e+01,   3.177989e+00,   1.024462e+01,   2.403777e+00,
              -4.908564e-01,   1.006475e-01,  -3.303704e-01,  -2.455414e-02
          ])
      self.assertAllClose(
          np.sum(ys, axis=(0, 2)), [
              6.610287,  6.657996,  7.452699,  6.626875,  6.60216 ,  7.484191
          ])
      # pyformat: enable
      # pylint: enable=bad-whitespace

  def testMultiSourceFRNNWithAttentionSingleSource(self, dtype=tf.float32):
    with self.session(
        use_gpu=True, config=py_utils.SessionConfig(inline=False)) as sess:
      p = self._MultiSourceFRNNWithAttentionParams(
          single_source=True, dtype=dtype)
      frnn = p.cls(p)

      (src_encs, src_paddings, inputs,
       paddings) = self._MultiSourceFRNNWithAttentionInputs(
           single_source=True, dtype=dtype)

      a, m = frnn.FPropDefaultTheta(src_encs, src_paddings, inputs, paddings)
      frnn_out = tf.concat([a, m], 2)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()
      ys, = sess.run([frnn_out])
      self.assertEqual(ys.shape, (7, 6, 8))
      print(np.sum(ys, axis=(1, 2)), np.sum(ys, axis=(0, 1)),
            np.sum(ys, axis=(0, 2)))

      # These values are identical with FRNNWithAttention.
      expected_sum12 = [
          13.07380962, 13.03321552, 12.99956226, 13.00612164, 13.01202011,
          12.99347878, 12.98680687
      ]
      expected_sum01 = [
          2.41238327e+01, 2.11899853e+01, 2.45926647e+01, 2.22827835e+01,
          -5.62886238e-01, 2.42760777e-02, -5.79716980e-01, 3.40666063e-02
      ]
      expected_sum02 = [
          12.74695969, 16.13114548, 16.66101837, 12.74922562, 16.16581345,
          16.65085411
      ]

      self.assertAllClose(np.sum(ys, axis=(1, 2)), expected_sum12)
      self.assertAllClose(np.sum(ys, axis=(0, 1)), expected_sum01)
      self.assertAllClose(np.sum(ys, axis=(0, 2)), expected_sum02)

  def testMultiSourceFRNNWithAttentionGradSingleSource(self, dtype=tf.float64):
    with self.session(
        use_gpu=True, config=py_utils.SessionConfig(inline=False)) as sess:

      p = self._MultiSourceFRNNWithAttentionParams(
          single_source=True, dtype=dtype)
      frnn = p.cls(p)

      (src_encs, src_paddings, inputs,
       paddings) = self._MultiSourceFRNNWithAttentionInputs(
           single_source=True, dtype=dtype)

      # Fetch all the parameters.
      w0, b0 = (frnn.theta.cell.wm, frnn.theta.cell.b)
      att0h, att0q, att0s = (frnn.theta.attentions[0].hidden_var,
                             frnn.theta.attentions[0].query_var,
                             frnn.theta.attentions[0].source_var)

      out, _ = frnn.FPropDefaultTheta(src_encs, src_paddings, inputs, paddings)
      loss = tf.reduce_sum(out)

      parameters = [w0, b0, inputs, att0h, att0q, att0s]
      grads = tf.gradients(loss, parameters)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()
      sym_grads = sess.run(grads)
      num_grads = [
          test_utils.ComputeNumericGradient(sess, loss, v, delta=1e-5)
          for v in parameters
      ]
      for i, (sym, num) in enumerate(zip(sym_grads, num_grads)):
        print([
            i, sym.shape, num.shape,
            np.max(np.abs(sym)),
            np.max(np.abs(sym - num)),
            np.max(np.abs(sym - num) / np.abs(sym))
        ])

      def Compare(name, sym, num, rtol=1e-5):
        print(['name = ', name])
        self.assertFalse(np.any(np.isnan(sym)))
        self.assertFalse(np.any(np.isnan(num)))
        self.assertAllClose(sym, num, rtol=rtol, atol=1e-8)

      for i, (sym, num) in enumerate(zip(sym_grads, num_grads)):
        Compare(parameters[i].name, sym, num)

  def testMultiSourceFRNNWithAttentionGrad(self, dtype=tf.float64):
    with self.session(
        use_gpu=True, config=py_utils.SessionConfig(inline=False)) as sess:

      p = self._MultiSourceFRNNWithAttentionParams(dtype=dtype)
      frnn = p.cls(p)

      # Fetch all the parameters.
      w0, b0 = (frnn.theta.cell.wm, frnn.theta.cell.b)
      mh, mq, ms = (frnn.theta.atten_merger.atten.hidden_var,
                    frnn.theta.atten_merger.atten.query_var,
                    frnn.theta.atten_merger.atten.source_var)
      att0h, att0q, att0s = (frnn.theta.attentions[0].hidden_var,
                             frnn.theta.attentions[0].query_var,
                             frnn.theta.attentions[0].source_var)
      att1h, att1q, att1s = (frnn.theta.attentions[1].hidden_var,
                             frnn.theta.attentions[1].query_var,
                             frnn.theta.attentions[1].source_var)
      att2h, att2q, att2s = (frnn.theta.attentions[2].hidden_var,
                             frnn.theta.attentions[2].query_var,
                             frnn.theta.attentions[2].source_var)

      (src_encs, src_paddings, inputs,
       paddings) = self._MultiSourceFRNNWithAttentionInputs(dtype=dtype)

      out, _ = frnn.FPropDefaultTheta(src_encs, src_paddings, inputs, paddings)
      loss = tf.reduce_sum(out)

      parameters = [
          w0, b0, inputs, mh, mq, ms, att0h, att0q, att0s, att1h, att1q, att1s,
          att2h, att2q, att2s
      ]
      grads = tf.gradients(loss, parameters)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()
      sym_grads = sess.run(grads)
      num_grads = [
          test_utils.ComputeNumericGradient(sess, loss, v, delta=1e-5)
          for v in parameters
      ]
      for i, (sym, num) in enumerate(zip(sym_grads, num_grads)):
        print([
            i, sym.shape, num.shape,
            np.max(np.abs(sym)),
            np.max(np.abs(sym - num)),
            np.max(np.abs(sym - num) / np.abs(sym))
        ])

      def Compare(name, sym, num, rtol=1e-5):
        print(['name = ', name])
        self.assertFalse(np.any(np.isnan(sym)))
        self.assertFalse(np.any(np.isnan(num)))
        self.assertAllClose(sym, num, rtol=rtol, atol=1e-8)

      for i, (sym, num) in enumerate(zip(sym_grads, num_grads)):
        Compare(parameters[i].name, sym, num)

  def testMultiSourceFRNNWithAttentionGradMultiDepth(self, dtype=tf.float64):
    with self.session(
        use_gpu=True, config=py_utils.SessionConfig(inline=False)) as sess:

      p = self._MultiSourceFRNNWithAttentionParams(
          single_source_length=False, dtype=dtype)
      frnn = p.cls(p)

      # Fetch all the parameters.
      w0, b0 = (frnn.theta.cell.wm, frnn.theta.cell.b)
      mh, mq, ms, mw0, mw1, mw2 = (frnn.theta.atten_merger.atten.hidden_var,
                                   frnn.theta.atten_merger.atten.query_var,
                                   frnn.theta.atten_merger.atten.source_var,
                                   frnn.theta.atten_merger.pre_proj[0].w,
                                   frnn.theta.atten_merger.pre_proj[1].w,
                                   frnn.theta.atten_merger.pre_proj[2].w)
      att0h, att0q, att0s = (frnn.theta.attentions[0].hidden_var,
                             frnn.theta.attentions[0].query_var,
                             frnn.theta.attentions[0].source_var)
      att1h, att1q, att1s = (frnn.theta.attentions[1].hidden_var,
                             frnn.theta.attentions[1].query_var,
                             frnn.theta.attentions[1].source_var)
      att2h, att2q, att2s = (frnn.theta.attentions[2].hidden_var,
                             frnn.theta.attentions[2].query_var,
                             frnn.theta.attentions[2].source_var)

      (src_encs, src_paddings, inputs,
       paddings) = self._MultiSourceFRNNWithAttentionInputs(
           single_source_length=False, dtype=dtype)

      out, _ = frnn.FPropDefaultTheta(src_encs, src_paddings, inputs, paddings)
      loss = tf.reduce_sum(out)

      parameters = [
          w0, b0, inputs, mh, mq, ms, att0h, att0q, att0s, att1h, att1q, att1s,
          att2h, att2q, att2s, mw0, mw1, mw2
      ]
      grads = tf.gradients(loss, parameters)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()
      sym_grads = sess.run(grads)
      num_grads = [
          test_utils.ComputeNumericGradient(sess, loss, v, delta=1e-5)
          for v in parameters
      ]
      for i, (sym, num) in enumerate(zip(sym_grads, num_grads)):
        print([
            i, sym.shape, num.shape,
            np.max(np.abs(sym)),
            np.max(np.abs(sym - num)),
            np.max(np.abs(sym - num) / np.abs(sym))
        ])

      def Compare(name, sym, num, rtol=1e-5):
        print(['name = ', name])
        self.assertFalse(np.any(np.isnan(sym)))
        self.assertFalse(np.any(np.isnan(num)))
        self.assertAllClose(sym, num, rtol=rtol, atol=1e-8)

      for i, (sym, num) in enumerate(zip(sym_grads, num_grads)):
        Compare(parameters[i].name, sym, num)

  def _CreateFRNNWithAttentionParams(self, dtype, dims, slen, sbatch, tlen,
                                     tbatch):
    # Create RNN Layer.
    p = rnn_cell.LSTMCellSimple.Params()
    p.name = 'lstm'
    p.dtype = dtype
    p.output_nonlinearity = True
    p.params_init = py_utils.WeightInit.Uniform(0.02, 429891685)
    p.vn.global_vn = False
    p.vn.per_step_vn = False
    p.num_input_nodes = dims * 2
    p.num_output_nodes = dims
    lstm_params = p

    # Create Attention Layer.
    p = attention.AdditiveAttention.Params()
    p.name = 'atten'
    p.dtype = dtype
    p.params_init = py_utils.WeightInit.Gaussian(0.1, 12345)
    p.source_dim = dims
    p.query_dim = dims
    p.hidden_dim = dims
    p.vn.global_vn = False
    p.vn.per_step_vn = False
    atten = p

    p = rnn_layers.FRNNWithAttention.Params()
    p.name = 'frnn_with_atten'
    p.dtype = dtype
    p.cell = lstm_params
    p.attention = atten
    p.output_prev_atten_ctx = False
    return p

  def testFRNNWithAttentionSeparateSourceContextIdenticalToSourceEnc(self):
    dtype = tf.float32
    dims = 4
    slen = 10
    sbatch = 3
    tlen = 7
    tbatch = 6

    with self.session(
        use_gpu=True, config=py_utils.SessionConfig(inline=True)) as sess:
      np.random.seed(12345)
      p = self._CreateFRNNWithAttentionParams(
          dtype=dtype,
          dims=dims,
          slen=slen,
          sbatch=sbatch,
          tlen=tlen,
          tbatch=tbatch)

      frnn = p.cls(p)

      src_encs = tf.constant(
          np.random.uniform(size=[slen, sbatch, dims]), dtype)
      src_paddings = tf.constant(np.zeros([slen, sbatch]), dtype)

      inputs = tf.constant(np.random.uniform(size=(tlen, tbatch, dims)), dtype)
      paddings = tf.constant(np.zeros([tlen, tbatch, 1]), dtype)

      # Run without specifying source context vectors.
      atten_ctx, rnn_out, atten_prob, _ = frnn.FPropDefaultTheta(
          src_encs, src_paddings, inputs, paddings)
      frnn_out = tf.concat([atten_ctx, rnn_out, atten_prob], 2)

      # Run after providing separate source context vectors set to the src_encs
      # should provide the same answer.
      (atten_ctx_src_ctx, rnn_out_src_ctx, atten_prob_src_ctx,
       _) = frnn.FPropDefaultTheta(
           src_encs, src_paddings, inputs, paddings, src_contexts=src_encs)
      frnn_out_src_ctx = tf.concat(
          [atten_ctx_src_ctx, rnn_out_src_ctx, atten_prob_src_ctx], 2)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()
      frnn_out_v, frnn_out_src_ctx_v = sess.run([frnn_out, frnn_out_src_ctx])

      # Expected last dimensions for atten_ctx_src_ctx, rnn_out_src_ctx,
      # atten_prob_src_ctx are respectively, (dims, dims, slen).
      self.assertEqual(frnn_out_v.shape, (tlen, tbatch, 2 * dims + slen))
      self.assertEqual(frnn_out_src_ctx_v.shape, frnn_out_v.shape)

      self.assertAllClose(frnn_out_v, frnn_out_src_ctx_v)

  def testFRNNWithAttentionSeparateSourceContextDifferentFromSourceEnc(self):
    dtype = tf.float32
    dims = 4
    slen = 10
    sbatch = 3
    tlen = 7
    tbatch = 6

    with self.session(
        use_gpu=True, config=py_utils.SessionConfig(inline=True)) as sess:
      np.random.seed(12345)
      p = self._CreateFRNNWithAttentionParams(
          dtype=dtype,
          dims=dims,
          slen=slen,
          sbatch=sbatch,
          tlen=tlen,
          tbatch=tbatch)

      frnn = p.cls(p)

      src_encs = tf.constant(
          np.random.uniform(size=[slen, sbatch, dims]), dtype)
      src_paddings = tf.constant(np.zeros([slen, sbatch]), dtype)

      # We create src_contexts with even dimensions (0, 2) set to all zero, the
      # rest are set randomly.
      src_contexts = np.random.uniform(size=[slen, sbatch, dims])
      src_contexts[:, :, 0:dims:2] = 0.0
      src_contexts = tf.constant(src_contexts, dtype=dtype)

      inputs = tf.constant(np.random.uniform(size=(tlen, tbatch, dims)), dtype)
      paddings = tf.constant(np.zeros([tlen, tbatch, 1]), dtype)

      # Run after providing separate source context vectors set to the src_encs
      # should provide the same answer.
      atten_ctx, _, _, _ = frnn.FPropDefaultTheta(
          src_encs, src_paddings, inputs, paddings, src_contexts=src_contexts)

      # Initialize all the variables, and then run one step.
      tf.global_variables_initializer().run()
      atten_ctx_v = sess.run(atten_ctx)

      self.assertEqual(atten_ctx_v.shape, (tlen, tbatch, dims))
      # Verify that the output also has zeros in the locations that the
      # source context has zeros.
      self.assertAllClose(
          np.zeros(shape=(tlen, tbatch, dims // 2)),
          atten_ctx_v[:, :, 0:dims:2])

  def _testFRNNWithAttentionUseZeroAttenState(self, zero_atten_state_fn):
    dtype = tf.float32
    dims = 5
    slen = 4
    tlen = 3
    sbatch = 2
    tbatch = 6

    with self.session(use_gpu=True) as sess:
      p = self._CreateFRNNWithAttentionParams(
          dtype=dtype,
          dims=dims,
          slen=slen,
          sbatch=sbatch,
          tlen=tlen,
          tbatch=tbatch)
      p.use_zero_atten_state = True
      p.atten_context_dim = dims
      frnn = p.cls(p)

      # Override the ZeroAttentionState to have the desired output type
      frnn.atten.ZeroAttentionState = types.MethodType(zero_atten_state_fn,
                                                       frnn.atten)

      src_encs = tf.constant(
          np.random.uniform(size=[slen, sbatch, dims]), dtype)
      src_paddings = tf.constant(np.zeros([slen, sbatch]), dtype)

      inputs = tf.constant(np.random.uniform(size=(tlen, tbatch, dims)), dtype)
      paddings = tf.constant(np.zeros([tlen, tbatch, 1]), dtype)

      atten_ctx, rnn_out, atten_prob, _ = frnn.FPropDefaultTheta(
          src_encs, src_paddings, inputs, paddings)

      tf.global_variables_initializer().run()
      atten_ctx, rnn_out, atten_prob = sess.run(
          [atten_ctx, rnn_out, atten_prob])

      # Check shapes
      self.assertEqual(atten_ctx.shape, (tlen, tbatch, dims))
      self.assertEqual(rnn_out.shape, (tlen, tbatch, dims))
      self.assertEqual(atten_prob.shape, (tlen, tbatch, slen))

  def testFRNNWithAttentionUseZeroAttenStateTensor(self):

    def _TensorZeroAttenState(self, source_seq_length, decoder_batch_size):
      del source_seq_length
      p = self.params
      zs = tf.zeros([decoder_batch_size, 1], dtype=py_utils.FPropDtype(p))
      return zs

    self._testFRNNWithAttentionUseZeroAttenState(_TensorZeroAttenState)

  def testFRNNWithAttentionUseZeroAttenStateNestedMap(self):

    def _NestedMapZeroAttenState(self, source_seq_length, decoder_batch_size):
      del source_seq_length
      p = self.params
      zs = tf.zeros([decoder_batch_size, 1], dtype=py_utils.FPropDtype(p))
      return py_utils.NestedMap(z=zs)

    self._testFRNNWithAttentionUseZeroAttenState(_NestedMapZeroAttenState)


if __name__ == '__main__':
  tf.test.main()
