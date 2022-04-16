# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for lingvo Jax rnn_cell layers."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from lingvo.core import rnn_cell
from lingvo.jax import py_utils
from lingvo.jax import test_utils
from lingvo.jax.layers import rnn_cell as jax_rnn_cell
import numpy as np
import tensorflow.compat.v2 as tf

NestedMap = py_utils.NestedMap

_INIT_RANDOM_SEED = 429891685
_NUMPY_RANDOM_SEED = 12345


class RnnCellTest(test_utils.TestCase):

  @parameterized.parameters(
      (jax_rnn_cell.LSTMCellSimple, False, False),
      (jax_rnn_cell.LSTMCellSimple, False, True),
      (jax_rnn_cell.CIFGLSTMCellSimple, True, False),
      (jax_rnn_cell.CIFGLSTMCellSimple, True, True),
  )
  def test_LSTMSimple(self, jax_cell_class, cifg, output_nonlinearity):
    np.random.seed(_NUMPY_RANDOM_SEED)
    inputs = py_utils.NestedMap(
        act=[np.random.uniform(size=(3, 2))], padding=jnp.zeros([3, 1]))
    state0 = py_utils.NestedMap(
        c=np.random.uniform(size=(3, 2)), m=np.random.uniform(size=(3, 2)))
    tf_inputs = py_utils.NestedMap(
        act=[tf.constant(inputs.act[0], tf.float32)], padding=tf.zeros([3, 1]))
    tf_state0 = py_utils.NestedMap(
        c=tf.constant(state0.c, tf.float32),
        m=tf.constant(state0.m, tf.float32))

    params = rnn_cell.LSTMCellSimple.Params().Set(
        name='lstm',
        params_init=py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        bias_init=py_utils.WeightInit.Uniform(1.24, _INIT_RANDOM_SEED),
        num_input_nodes=2,
        num_output_nodes=2,
        couple_input_forget_gates=cifg,
        enable_lstm_bias=True,
        output_nonlinearity=output_nonlinearity)
    lstm = rnn_cell.LSTMCellSimple(params)
    res, _ = lstm.FPropDefaultTheta(tf_state0, tf_inputs)
    m_expected = res.m.numpy()
    c_expected = res.c.numpy()

    p = jax_cell_class.Params().Set(
        num_input_nodes=2,
        num_output_nodes=2,
        name='lstm',
        output_nonlinearity=output_nonlinearity,
    )
    model = p.Instantiate()

    theta = model.instantiate_variables(jax.random.PRNGKey(5678))
    theta.wm = lstm.vars['wm'].numpy()
    theta.b = lstm.vars['b'].numpy()

    output, _ = test_utils.apply(model, model.vars_to_flax_vars(theta),
                                 model.fprop, state0, inputs)
    self.assertAllClose(m_expected, output.m)
    self.assertAllClose(c_expected, output.c)


if __name__ == '__main__':
  absltest.main()
