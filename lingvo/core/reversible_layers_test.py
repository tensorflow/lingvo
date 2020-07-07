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
# ==============================================================================
"""Tests for reversible layers."""

from absl.testing import parameterized
from lingvo import compat as tf
from lingvo.core import layers
from lingvo.core import py_utils
from lingvo.core import reversible_layers
from lingvo.core import test_utils

import numpy as np

FLAGS = tf.flags.FLAGS


class RevNetLayerTest(test_utils.TestCase):
  """Test single revnet layer."""

  def testRevNetLayerFProp(self):
    with self.session():
      tf.random.set_seed(321)
      input_1 = tf.random.normal([5, 3], seed=89122)
      input_2 = tf.random.normal([5, 3], seed=19438)
      p = reversible_layers.RevNetLayer.Params()
      p.name = 'revnet_simple'
      p.f_params = layers.FCLayer.Params().Set(input_dim=3, output_dim=3)
      p.g_params = layers.FCLayer.Params().Set(input_dim=3, output_dim=3)
      revnet_layer = p.Instantiate()

      h, _, _ = revnet_layer.FPropDefaultTheta(
          py_utils.NestedMap(split1=input_1, split2=input_2))
      self.evaluate(tf.global_variables_initializer())
      actual_layer_output = self.evaluate(h)
      expected_split1 = np.array([[-0.7262997, 0.9276514, -0.20907314],
                                  [-0.7089523, 0.24923629, 2.5001974],
                                  [1.6766014, 0.26847264, -0.2510258],
                                  [0.9629222, -0.57908165, 0.0485389],
                                  [2.7580009, 0.17540382, 1.6282884]],
                                 dtype=np.float32)
      expected_split2 = np.array([[1.1282716, 1.4266306, -0.16530532],
                                  [-0.3836313, 0.04922554, 0.25543338],
                                  [0.03718817, 1.5488712, 2.1594636],
                                  [-2.1252284, 3.2059612, 0.1118355],
                                  [3.4058936, -0.63690275, -0.95291173]],
                                 dtype=np.float32)

      self.assertAllClose(expected_split1, actual_layer_output.split1)
      self.assertAllClose(expected_split2, actual_layer_output.split2)

  def testRevNetLayerReverseGrad(self):
    with self.session() as sess:
      tf.random.set_seed(321)
      input_1 = np.random.normal(size=[5, 3])
      input_2 = np.random.normal(size=[5, 3])
      x1 = tf.placeholder(dtype=tf.float32)
      x2 = tf.placeholder(dtype=tf.float32)
      p = reversible_layers.RevNetLayer.Params()
      p.name = 'revnet_simple'
      p.f_params = layers.FCLayer.Params().Set(input_dim=3, output_dim=3)
      p.g_params = layers.FCLayer.Params().Set(input_dim=3, output_dim=3)
      revnet_layer = p.Instantiate()

      outputs, f_seed, g_seed = revnet_layer.FPropDefaultTheta(
          py_utils.NestedMap(split1=x1, split2=x2))
      loss = tf.reduce_sum(outputs.split1 + outputs.split2)

      # Computes tensorflow gradients.
      dy1, dy2 = tf.gradients(loss, [outputs.split1, outputs.split2])
      dx1, dx2 = tf.gradients(loss, [x1, x2])
      dw = tf.gradients(
          loss,
          revnet_layer.theta.Flatten(),
          unconnected_gradients=tf.UnconnectedGradients.ZERO)

      # Computes custom gradients.
      inputs_reconstruct, dinputs, d_theta = revnet_layer.ReverseAndGrad(
          revnet_layer.theta,
          outputs,
          py_utils.NestedMap(split1=dy1, split2=dy2),
          f_seed,
          g_seed,
      )

      self.evaluate(tf.global_variables_initializer())

      # Tests the reverse.
      x1r, x2r = sess.run(
          [inputs_reconstruct.split1, inputs_reconstruct.split2], {
              x1: input_1,
              x2: input_2
          })
      self.assertAllClose(input_1, x1r)
      self.assertAllClose(input_2, x2r)

      # Tests the gradient.
      dx1_tf, dx2_tf, dx1_custom, dx2_custom, dw_tf, dw_custom = sess.run(
          [dx1, dx2, dinputs.split1, dinputs.split2, dw,
           d_theta.Flatten()], {
               x1: input_1,
               x2: input_2
           })
      self.assertAllClose(dx1_tf, dx1_custom)
      self.assertAllClose(dx2_tf, dx2_custom)
      self.assertAllClose(dw_tf, dw_custom)


class StackedRevNetLayerTest(test_utils.TestCase, parameterized.TestCase):
  """Test stacked layers."""

  def _SimpleRevNetParams(self, name, dropout, custom_gradient):
    """Construct a simple 3-layers RevNet."""
    layer_tpl = reversible_layers.RevNetLayer.Params()
    layer_tpl.f_params = layers.FeedForwardNet.Params().Set(
        input_dim=3, hidden_layer_dims=[3, 3])
    layer_tpl.g_params = layers.FeedForwardNet.Params().Set(
        input_dim=3, hidden_layer_dims=[3, 3])
    if dropout:
      layer_tpl.f_params.dropout = layers.DeterministicDropoutLayer.Params()
      layer_tpl.f_params.dropout.keep_prob = 0.7
      layer_tpl.g_params.dropout = layers.DeterministicDropoutLayer.Params()
      layer_tpl.g_params.dropout.keep_prob = 0.8

    stacked_p = reversible_layers.StackedRevNetLayer.Params()
    stacked_p.name = name
    stacked_p.custom_gradient = custom_gradient
    for idx in range(3):
      layer_p = layer_tpl.Copy()
      layer_p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      layer_p.name = 'layer_{}'.format(idx)
      stacked_p.sub_layer_params.append(layer_p)
    return stacked_p

  def _RunModel(self, input_1_val, input_2_val, dropout, custom_grad):
    self._ClearCachedSession()
    tf.reset_default_graph()
    with self.session() as sess:
      tf.random.set_seed(321)
      input_1 = tf.placeholder(tf.float32)
      input_2 = tf.placeholder(tf.float32)

      revnet_params = self._SimpleRevNetParams('revnet', dropout, custom_grad)
      revnet_params.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      revnet = revnet_params.Instantiate()

      h = revnet.FPropDefaultTheta(
          py_utils.NestedMap(split1=input_1, split2=input_2))

      dfx = tf.gradients(h.Flatten(), [input_1, input_2])
      dfw = tf.gradients(
          h.Flatten(),
          revnet.theta.Flatten(),
          unconnected_gradients=tf.UnconnectedGradients.ZERO)

      self.evaluate(tf.global_variables_initializer())
      dfx_val, dfw_val, h_val = sess.run([dfx, dfw, h],
                                         feed_dict={
                                             input_1: input_1_val,
                                             input_2: input_2_val,
                                         })
      return h_val, dfx_val, dfw_val

  @parameterized.named_parameters(('nodropout', False), ('dropout', True))
  def testStackedRevNetLayer(self, dropout):
    input_1_val = np.random.normal(size=[5, 3])
    input_2_val = np.random.normal(size=[5, 3])
    # input_padding_val = np.random.randint(low=0, high=2, size=[5, 1])

    h, dfx, dfw = self._RunModel(input_1_val, input_2_val, dropout, True)
    h2, dfx2, dfw2 = self._RunModel(input_1_val, input_2_val, dropout, False)

    self.assertAllClose(h, h2)
    self.assertAllClose(dfx, dfx2)
    self.assertAllClose(dfw, dfw2)


if __name__ == '__main__':
  tf.test.main()
