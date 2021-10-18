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
"""A few utility layers to facilitate writing unit-tests."""

from typing import Any, Tuple

import flax.linen as flax_nn
import jax
from jax import numpy as jnp
from jax import random as jrandom
from lingvo.jax import base_layer
from lingvo.jax import model
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax.layers import flax_wrapper
from lingvo.jax.layers import linears
from lingvo.jax.layers import normalizations
from lingvo.jax.layers import transformers

InstantiableParams = py_utils.InstantiableParams
NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor


class ProjectionLayer(base_layer.BaseLayer):
  """A simple projection layer."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('input_dims', 0, 'Depth of the input.')
    p.Define('output_dims', 0, 'Depth of the output.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    linear_layer_p = linears.LinearLayer.Params().Set(
        input_dims=p.input_dims, output_dims=p.output_dims)
    self.CreateChild('linear', linear_layer_p)
    bias_layer_p = linears.BiasLayer.Params().Set(dims=p.output_dims)
    self.CreateChild('bias', bias_layer_p)

  def FProp(self, theta: NestedMap, inputs: JTensor) -> JTensor:
    return self.bias.FProp(theta.bias, self.linear.FProp(theta.linear, inputs))


class AddOneLayer(base_layer.BaseLayer):
  """A layers without any variables."""

  def FProp(self, theta: NestedMap, inputs: JTensor) -> JTensor:
    return inputs + 1.0


class TestLayer(base_layer.BaseLayer):
  """A test layer which is a composite of multiple layers."""

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    linear_layer_p01 = linears.LinearLayer.Params().Set(
        input_dims=2, output_dims=3)
    linear_layer_p02 = linears.LinearLayer.Params().Set(
        input_dims=3, output_dims=4)
    self.CreateChildren('linear', {
        'linear01': linear_layer_p01,
        'linear02': linear_layer_p02
    })
    bias_layer_p01 = linears.BiasLayer.Params().Set(dims=3)
    bias_layer_p02 = linears.BiasLayer.Params().Set(dims=4)
    self.CreateChildren('bias', [bias_layer_p01, bias_layer_p02])
    add_one_layer_p = AddOneLayer.Params()
    self.CreateChild('add_one', add_one_layer_p)

  def CreateLayerVariables(self) -> None:
    super().CreateLayerVariables()
    p = self.params
    self.CreateVariable(
        'final_proj',
        base_layer.WeightParams(
            shape=[4, 5], init=p.params_init, dtype=p.dtype))

  def FProp(self, theta: NestedMap, inputs: JTensor) -> JTensor:
    x1 = self.linear.linear01.FProp(theta.linear.linear01, inputs)
    x2 = self.bias[0].FProp(theta.bias[0], x1)
    x3 = self.linear.linear02.FProp(theta.linear.linear02, x2)
    x4 = self.bias[1].FProp(theta.bias[1], x3)
    x5 = linears.ProjectLastDim(x4, theta.final_proj)
    x6 = self.add_one.FProp(theta.add_one, x5)
    return x6


class CNN(flax_nn.Module):
  """A simple CNN model."""

  @flax_nn.compact
  def __call__(self, x: JTensor) -> JTensor:
    x = flax_nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = flax_nn.relu(x)
    x = flax_nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = flax_nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = flax_nn.relu(x)
    x = flax_nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = flax_nn.Dense(features=256)(x)
    x = flax_nn.relu(x)
    x = flax_nn.Dense(features=10)(x)
    x = flax_nn.log_softmax(x)
    return x


class MnistCnnLayer(flax_wrapper.FlaxModuleLayer):
  """A wrapper of the CNN layer above."""

  def _CreateFlaxModule(self) -> flax_nn.Module:
    return CNN()

  def _InitModuleStates(self, prng_key: JTensor) -> NestedMap:
    prng_key, sub_key1 = jrandom.split(prng_key)
    prng_key, sub_key2 = jrandom.split(prng_key)
    jit_init = jax.jit(self._module.init)
    initial_vars = jit_init({
        'params': sub_key1,
        'dropout': sub_key2
    }, jnp.ones([2, 32, 32, 1], dtype=jnp.float32))
    return initial_vars

  def FProp(self, theta: NestedMap, *args: Any, **kwargs: Any) -> JTensor:
    prng_key1 = base_layer.NextPrngKey()
    prng_key2 = base_layer.NextPrngKey()
    out = self._module.apply(
        theta,
        *args,
        rngs={
            'params': prng_key1,
            'dropout': prng_key2
        },
        **kwargs)
    return out


class FlaxTestLayer(base_layer.BaseLayer):
  """A composite layer that consists of a flax module."""

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    cnn_p = MnistCnnLayer.Params()
    self.CreateChild('cnn_p1', cnn_p.Copy())
    self.CreateChild('cnn_p2', cnn_p.Copy())
    bn_p = normalizations.BatchNormLayer.Params().Set(dim=10)
    self.CreateChild('bn', bn_p)

  def FProp(self,
            theta: NestedMap,
            x: JTensor) -> Tuple[JTensor, JTensor, JTensor]:
    out1 = self.cnn_p1.FProp(theta.cnn_p1, x)
    out2 = self.cnn_p2.FProp(theta.cnn_p2, x)
    out = self.bn.FProp(theta.bn, out1 + out2)
    return out1, out2, out


class VarUnusedLayer(base_layer.BaseLayer):
  """A test where some of the vars are not used in FProp."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('input_dims', 0, 'Depth of the input.')
    p.Define('output_dims', 0, 'Depth of the output.')
    return p

  def CreateLayerVariables(self) -> None:
    super().CreateLayerVariables()
    p = self.params
    self.CreateVariable(
        'var01',
        base_layer.WeightParams(
            shape=[p.input_dims, p.output_dims],
            init=p.params_init,
            dtype=p.dtype))
    # var02 is not used.
    self.CreateVariable(
        'var02',
        base_layer.WeightParams(
            shape=[p.input_dims, p.output_dims],
            init=p.params_init,
            dtype=p.dtype))

  def FProp(self, theta: NestedMap, inputs: JTensor) -> JTensor:
    out = jnp.einsum('bi,io->bo', inputs, theta.var01)
    loss = jnp.sum(out)
    return loss


class TestModel01(model.BaseTask):
  """Simple model for testing."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('input_dims', 0, 'Depth of the input.')
    p.Define('output_dims', 0, 'Depth of the output.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    bn_params = normalizations.BatchNormLayer.Params().Set(
        name='bn', dim=p.input_dims)
    self.CreateChild('bn', bn_params)

  def CreateLayerVariables(self) -> None:
    super().CreateLayerVariables()
    p = self.params
    self.CreateVariable(
        'var01',
        base_layer.WeightParams(
            shape=[p.input_dims, p.output_dims],
            init=p.params_init,
            dtype=p.dtype))
    # var02 is not used.
    self.CreateVariable(
        'var02',
        base_layer.WeightParams(
            shape=[p.input_dims, p.output_dims],
            init=p.params_init,
            dtype=p.dtype))

  def ComputePredictions(self, theta: NestedMap, inputs: JTensor) -> JTensor:
    in_normed = self.bn.FProp(theta.bn, inputs)
    return jnp.einsum('bi,io->bo', in_normed, theta.var01)

  def ComputeLoss(self,
                  theta: NestedMap,
                  predictions: JTensor,
                  inputs: JTensor) -> Tuple[NestedMap, NestedMap]:
    del inputs
    loss = jnp.sum(predictions)
    loss02 = jnp.sum(predictions * predictions)
    # Here loss is the main loss to back-prop into, and loss02 is an eval
    # metric.
    per_example_out = NestedMap()
    return NestedMap(
        loss=(loss, jnp.array(1.0, loss.dtype)),
        loss02=(loss02, jnp.array(1.0, loss02.dtype))), per_example_out


class TestLinearRegressionModel(model.BaseTask):
  """Linear regression model."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('input_dims', 0, 'Depth of the input.')
    p.Define('output_dims', 0, 'Depth of the output.')
    p.Define('linear_p', linears.LinearLayer.Params(),
             'Params for the linear layer.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    params = p.linear_p
    params.input_dims = p.input_dims
    params.output_dims = p.output_dims
    self.CreateChild('linear', params)

  def ComputePredictions(self,
                         theta: NestedMap,
                         input_batch: NestedMap) -> JTensor:
    return self.linear.FProp(theta.linear, input_batch.inputs)

  def ComputeLoss(self, theta, predictions, input_batch):
    targets = input_batch.targets
    error = predictions - targets
    loss = jnp.mean(jnp.square(error))
    per_example_out = NestedMap(predictions=predictions)
    return NestedMap(
        loss=(loss, jnp.array(1.0, loss.dtype))), per_example_out


class TestBatchNormalizationModel(model.BaseTask):
  """Test batch normalization correctness using a regression task."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('input_dims', 0, 'Depth of the input.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    bn_params = normalizations.BatchNormLayer.Params().Set(
        name='bn', dim=p.input_dims)
    self.CreateChild('bn', bn_params)

  def ComputePredictions(self,
                         theta: NestedMap,
                         input_batch: NestedMap) -> JTensor:
    return self.bn.FProp(theta.bn, input_batch.inputs)

  def ComputeLoss(self,
                  theta: NestedMap,
                  predictions: JTensor,
                  input_batch: NestedMap) -> Tuple[NestedMap, NestedMap]:
    targets = input_batch.targets
    error = predictions - targets
    loss = jnp.mean(jnp.square(error))
    per_example_out = NestedMap(predictions=predictions)
    return NestedMap(
        loss=(loss, jnp.array(1.0, loss.dtype))), per_example_out


class TestSpmdModel(model.BaseTask):
  """A simple spmd model for testing purposes."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('xformer_ffw', transformers.TransformerFeedForwardLayer.Params(),
             'Xformer feedforward layer params.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self.CreateChild('ffwd', p.xformer_ffw)

  def ComputePredictions(self, theta: NestedMap, inputs: NestedMap) -> JTensor:
    return self.ffwd.FProp(theta.ffwd, inputs)

  def ComputeLoss(self, theta: NestedMap, predictions: JTensor,
                  input_batch: NestedMap) -> Tuple[NestedMap, NestedMap]:
    loss = jnp.mean(jnp.square(predictions))
    per_example_out = NestedMap(predictions=predictions)
    return NestedMap(loss=(loss, jnp.array(1.0, loss.dtype))), per_example_out
