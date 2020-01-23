# Lint as: python2, python3
# coding=utf-8
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
"""Reversible layers for memory efficient Backpropagation.

[1] The Reversible Residual Network: Backpropagation Without Storing Activations
    https://arxiv.org/abs/1707.04585
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils

from six.moves import zip


class RevNetLayer(base_layer.BaseLayer):
  """A reversible residual block.

  Computes y1 = x1 + f(x2), y2 = x2 + g(y1), where f and g can be arbitrary
  functions that retain the input tensor shape.
  """

  @classmethod
  def Params(cls):
    p = super(RevNetLayer, cls).Params()
    p.Define('f_params', None, 'Layer params for the f block.')
    p.Define('g_params', None, 'Layer params for the g block.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(RevNetLayer, self).__init__(params)
    p = params
    assert p.name
    assert p.f_params
    assert p.g_params
    with tf.variable_scope(p.name):
      self.CreateChild('f_block', p.f_params)
      self.CreateChild('g_block', p.g_params)

  def ReverseAndGrad(self, theta, outputs, d_outputs, f_seed, g_seed,
                     *extra_inputs):
    """Implements Algorithm 1 in the revnet paper.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      outputs: A NestedMap: .split1 and .split2 corresponding to y1 and y2.
      d_outputs: A NestedMap: .split1 and .split2 corresponding to dy1 and dy2,
        the total derivatives.
      f_seed: Scalar tensor. The step seed used in forward for the f block.
      g_seed: Scalar tensor. The step seed used in forward for the g block. The
        step seeds are needed for deterministic randomness, e.g. to ensure
        dropout generate the same random mask in forward and reverse_grad.
      *extra_inputs: additional inputs that will be passed to both f and g. No
        gradient will be computed for these inputs.

    Returns:
      A tuple of NestedMaps

      - inputs: .split1 and .split2 corresponding to x1 and x2.
      - d_inputs: .split1 and .split2 corresponding to dx1 and dx2, the total
        derivatives with respect to inputs.
      - d_theta: has the same structure as theta. The total derivatives with
        respect to weights.

    """

    # Stop gradient on the outputs to avoid circular symbolic dependency.
    y1 = tf.stop_gradient(outputs.split1)
    y2 = tf.stop_gradient(outputs.split2)
    dy1 = d_outputs.split1
    dy2 = d_outputs.split2

    # Computes the reverse.
    z1 = y1
    py_utils.ResetStepSeed(g_seed)
    gz1 = self.g_block.FProp(theta.g_block, z1, *extra_inputs)
    x2 = y2 - gz1
    py_utils.ResetStepSeed(f_seed)
    fx2 = self.f_block.FProp(theta.f_block, x2, *extra_inputs)
    x1 = z1 - fx2

    # Computes the gradients.
    dz1 = dy1 + tf.gradients(gz1, z1, dy2)[0]
    dx2 = dy2 + tf.gradients(fx2, x2, dz1)[0]

    dgw = tf.gradients(
        gz1,
        theta.g_block.Flatten(),
        dy2,
        unconnected_gradients=tf.UnconnectedGradients.ZERO)
    dgw = theta.g_block.Pack(dgw)

    dfw = tf.gradients(
        fx2,
        theta.f_block.Flatten(),
        dz1,
        unconnected_gradients=tf.UnconnectedGradients.ZERO)
    dfw = theta.f_block.Pack(dfw)

    return (py_utils.NestedMap(split1=x1, split2=x2),
            py_utils.NestedMap(split1=dz1, split2=dx2),
            py_utils.NestedMap(
                f_block=dfw,
                g_block=dgw,
                global_step=tf.zeros_like(theta.global_step)))

  def FProp(self, theta, inputs, *extra_inputs):
    """Forward pass.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      inputs: A NestedMap: .split1 and .split2 corresponding to x1 and x2.
      *extra_inputs: additional inputs that will be passed to both f and g. No
        gradient will be computed for these inputs.

    Returns:
      outputs: A NestedMap: .split1 and .split2 corresponding to y1 and y2.
      f_seed: Scalar tensor. The step seed used in forward for the f block.
      g_seed: Scalar tensor. The step seed used in forward for the g block.

    """

    f_seed = py_utils.GetStepSeed()
    f_out = self.f_block.FProp(theta.f_block, inputs.split2, *extra_inputs)
    z1 = inputs.split1 + f_out
    g_seed = py_utils.GetStepSeed()
    g_out = self.g_block.FProp(theta.g_block, z1, *extra_inputs)
    y2 = inputs.split2 + g_out
    # This is essential to make dy1 independent to y2.
    y1 = tf.identity(z1)
    return py_utils.NestedMap(split1=y1, split2=y2), f_seed, g_seed


class StackedRevNetLayer(base_layer.BaseLayer):
  """Stacked RevNet layers with custom gradient.

  The standard backpropagation has peak memory footprint of

    Θ(num_layers x (activation_size_per_layer + param_size_per_layer)),

  which is reduced to

    Θ(activation_size_per_layer + num_layers x param_size_per_layer)),

  in the custom gradient implmentation, at the cost of extra computation for the
  reverse and potential accumulated numerical errors. See Section 3.2 in the
  revnet paper for the full discussion.
  """

  @classmethod
  def Params(cls):
    p = super(StackedRevNetLayer, cls).Params()
    p.Define('sub_layer_params', [], 'A list of RevNetLayer params.')
    p.Define(
        'custom_gradient', True, 'If True, use the custom gradient over'
        'the standard TF gradient. Useful for unit test and benchmarks')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(StackedRevNetLayer, self).__init__(params)
    p = params
    assert p.name
    assert p.sub_layer_params
    with tf.variable_scope(p.name):
      self.CreateChildren('sub_layers', p.sub_layer_params)

  def FProp(self, theta, inputs, *extra_inputs):

    initial_step_seed = py_utils.GetStepSeed()
    final_step_seed = py_utils.GenerateSeedFromName(
        tf.no_op(name='new_step_seed').name)
    num_layers = len(self.sub_layers)

    def Bak(inputs, outputs, d_outputs):
      """Backward step."""
      del inputs  # unused
      output_acts, step_seeds = outputs
      d_outputs = d_outputs[0]

      d_layer_thetas = []
      for layer_idx in reversed(range(num_layers)):
        f_seed, g_seed = step_seeds[layer_idx]
        layer = self.sub_layers[layer_idx]
        layer_theta = theta.sub_layers[layer_idx]

        input_acts, d_inputs, d_theta = layer.ReverseAndGrad(
            layer_theta, output_acts, d_outputs, f_seed, g_seed, *extra_inputs)

        d_layer_thetas.append(d_theta)
        # Passes reconstructed inputs to the previous layer.
        output_acts = input_acts
        d_outputs = d_inputs
      py_utils.ResetStepSeed(final_step_seed)
      d_theta = py_utils.NestedMap(global_step=tf.zeros_like(initial_step_seed))
      d_theta.sub_layers = list(reversed(d_layer_thetas))

      extra_grads = [tf.zeros_like(t) for t in extra_inputs]
      return [tf.zeros_like(initial_step_seed), d_theta, d_inputs, extra_grads]

    def Fwd(xs):
      """Forward pass."""
      initial_step_seed, theta, acts, extra_inputs = xs

      py_utils.ResetStepSeed(initial_step_seed)
      layer_step_seeds = []

      for layer_theta, layer in zip(theta.sub_layers, self.sub_layers):
        acts, f_seed, g_seed = layer.FProp(layer_theta, acts, *extra_inputs)
        layer_step_seeds += [(f_seed, g_seed)]
      return [acts, layer_step_seeds]

    if self.params.custom_gradient:
      acts, _ = py_utils.CallDefun(
          Fwd, Bak, [initial_step_seed, theta, inputs, extra_inputs])
      py_utils.ResetStepSeed(final_step_seed)
      return acts
    else:
      acts = inputs
      for layer_theta, layer in zip(theta.sub_layers, self.sub_layers):
        acts, _, _ = layer.FProp(layer_theta, acts, *extra_inputs)
      return acts
