# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Linear layers."""

from jax import numpy as jnp
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax.layers import activations

NestedMap = py_utils.NestedMap
WeightInit = py_utils.WeightInit
WeightParams = py_utils.WeightParams

InstantiableParams = py_utils.InstantiableParams
JTensor = pytypes.JTensor


def ProjectLastDim(inputs: JTensor, weight: JTensor) -> JTensor:
  """Linear projection on the last dim of the input JTensor.

  This is a TPU efficient implementation to avoid reshaping inputs to Rank-2
  JTensor by using Einsum for the compute.

  Args:
    inputs: An input JTensor, the last dimension of which is input_dim.
    weight: A weight matrix with shape [input_dim, output_dim].

  Returns:
    An output JTensor of the same rank as inputs, the last dimension is
    output_dim.
  """
  input_shape = inputs.shape
  assert len(input_shape) >= 2
  weight_shape = weight.shape
  assert len(weight_shape) == 2
  assert input_shape[-1] == weight_shape[0]
  if len(input_shape) == 2:
    return jnp.matmul(inputs, weight)
  else:
    # This is equivalent to:
    #   outputs = tf.einsum('...y,yz->...z', inputs, weight)
    # Unfortunately ... in einsum() leads to extra HBM usage.
    s = ''.join([chr(x) for x in range(97, 123)])  # abc...xyz
    r = len(input_shape)
    return jnp.einsum('{0}y,yz->{0}z'.format(s[:r - 1]), inputs, weight)


class LinearLayer(base_layer.BaseLayer):
  """Linear layer without bias."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('input_dims', 0, 'Depth of the input.')
    p.Define('output_dims', 0, 'Depth of the output.')
    return p

  def CreateLayerVariables(self) -> None:
    super().CreateLayerVariables()
    p = self.params
    wp = p.weight_split_dims_mapping
    self.CreateVariable(
        'w',
        WeightParams(
            shape=[p.input_dims, p.output_dims],
            init=p.params_init,
            dtype=p.dtype,
            device_mesh=p.device_mesh,
            tensor_split_dims_mapping=wp.wt))

  def FProp(self, theta: NestedMap, inputs: JTensor) -> JTensor:
    """Apply projection to inputs.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      inputs: The inputs JTensor.  Shaped [..., input_dims].

    Returns:
      Projected inputs.
    """
    p = self.params
    ap = p.activation_split_dims_mapping
    out = ProjectLastDim(inputs, theta.w)
    out = base_layer.MaybeShard(out, ap.out, p.mesh_axis_names)
    return out


class BiasLayer(base_layer.BaseLayer):
  """Bias layer."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('dims', 0, 'Depth of the input.')
    return p

  def CreateLayerVariables(self) -> None:
    super().CreateLayerVariables()
    p = self.params
    wp = p.weight_split_dims_mapping
    self.CreateVariable(
        'b',
        WeightParams(
            shape=[p.dims],
            init=WeightInit.Constant(0.0),
            dtype=p.dtype,
            device_mesh=p.device_mesh,
            tensor_split_dims_mapping=wp.wt))

  def FProp(self, theta: NestedMap, inputs: JTensor) -> JTensor:
    """Adds bias to inputs.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      inputs: The inputs JTensor.  Shaped [..., dims].

    Returns:
      Inputs plus bias.
    """
    return inputs + theta.b


class FeedForwardLayer(base_layer.BaseLayer):
  """Feedforward layer with activation."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('input_dims', 0, 'Depth of the input.')
    p.Define('output_dims', 0, 'Depth of the output.')
    p.Define(
        'activation', 'RELU', 'Activation function to use.'
        'Options are RELU, RELU6, RELU^2, RELU^3, SIGMOID, TANH, GELU, NONE.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    wp = p.weight_split_dims_mapping
    ap = p.activation_split_dims_mapping
    linear_layer_p = LinearLayer.Params().Set(
        input_dims=p.input_dims,
        output_dims=p.output_dims,
        weight_split_dims_mapping=wp.Copy(),
        activation_split_dims_mapping=ap.Copy())
    self.CreateChild('linear', linear_layer_p)
    bias_layer_p = BiasLayer.Params().Set(dims=p.output_dims)
    if p.device_mesh is not None and wp.wt is not None:
      assert len(wp.wt) == 2
      wp_bias = [wp.wt[1]]
      bias_layer_p.weight_split_dims_mapping.wt = wp_bias
    self.CreateChild('bias', bias_layer_p)
    act_p = activations.ActivationLayer.Params().Set(activation=p.activation)
    self.CreateChild('activation', act_p)

  def FProp(self, theta: NestedMap, inputs: JTensor) -> JTensor:
    projected_inputs = self.bias.FProp(theta.bias,
                                       self.linear.FProp(theta.linear, inputs))
    output = self.activation.FProp(theta.activation, projected_inputs)
    return output
