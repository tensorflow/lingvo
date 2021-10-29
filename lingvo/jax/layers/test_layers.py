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
    linear_layer_p = linears.Linear.Params().Set(
        input_dims=p.input_dims, output_dims=p.output_dims)
    self.create_child('linear', linear_layer_p)
    bias_layer_p = linears.Bias.Params().Set(dims=p.output_dims)
    self.create_child('bias', bias_layer_p)

  def fprop(self, theta: NestedMap, inputs: JTensor) -> JTensor:
    return self.bias.fprop(theta.bias, self.linear.fprop(theta.linear, inputs))


class AddOneLayer(base_layer.BaseLayer):
  """A layers without any variables."""

  def fprop(self, theta: NestedMap, inputs: JTensor) -> JTensor:
    return inputs + 1.0


class TestLayer(base_layer.BaseLayer):
  """A test layer which is a composite of multiple layers."""

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    linear_layer_p01 = linears.Linear.Params().Set(input_dims=2, output_dims=3)
    linear_layer_p02 = linears.Linear.Params().Set(input_dims=3, output_dims=4)
    self.create_children('linear', {
        'linear01': linear_layer_p01,
        'linear02': linear_layer_p02
    })
    bias_layer_p01 = linears.Bias.Params().Set(dims=3)
    bias_layer_p02 = linears.Bias.Params().Set(dims=4)
    self.create_children('bias', [bias_layer_p01, bias_layer_p02])
    add_one_layer_p = AddOneLayer.Params()
    self.create_child('add_one', add_one_layer_p)

  def create_layer_variables(self) -> None:
    super().create_layer_variables()
    p = self.params
    self.create_variable(
        'final_proj',
        base_layer.weight_params(
            shape=[4, 5], init=p.params_init, dtype=p.dtype))

  def fprop(self, theta: NestedMap, inputs: JTensor) -> JTensor:
    x1 = self.linear.linear01.fprop(theta.linear.linear01, inputs)
    x2 = self.bias[0].fprop(theta.bias[0], x1)
    x3 = self.linear.linear02.fprop(theta.linear.linear02, x2)
    x4 = self.bias[1].fprop(theta.bias[1], x3)
    x5 = linears.project_last_dim(x4, theta.final_proj)
    x6 = self.add_one.fprop(theta.add_one, x5)
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


class MnistCnnLayer(flax_wrapper.FlaxModule):
  """A wrapper of the CNN layer above."""

  def _create_flax_module(self) -> flax_nn.Module:
    return CNN()

  def _init_module_states(self, prng_key: JTensor) -> NestedMap:
    prng_key, sub_key1 = jrandom.split(prng_key)
    prng_key, sub_key2 = jrandom.split(prng_key)
    jit_init = jax.jit(self._module.init)
    initial_vars = jit_init({
        'params': sub_key1,
        'dropout': sub_key2
    }, jnp.ones([2, 32, 32, 1], dtype=jnp.float32))
    return initial_vars

  def fprop(self, theta: NestedMap, *args: Any, **kwargs: Any) -> JTensor:
    prng_key1 = base_layer.next_prng_key()
    prng_key2 = base_layer.next_prng_key()
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
    self.create_child('cnn_p1', cnn_p.Copy())
    self.create_child('cnn_p2', cnn_p.Copy())
    bn_p = normalizations.BatchNorm.Params().Set(dim=10)
    self.create_child('bn', bn_p)

  def fprop(self, theta: NestedMap,
            x: JTensor) -> Tuple[JTensor, JTensor, JTensor]:
    out1 = self.cnn_p1.fprop(theta.cnn_p1, x)
    out2 = self.cnn_p2.fprop(theta.cnn_p2, x)
    out = self.bn.fprop(theta.bn, out1 + out2)
    return out1, out2, out


class VarUnusedLayer(base_layer.BaseLayer):
  """A test where some of the vars are not used in fprop."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('input_dims', 0, 'Depth of the input.')
    p.Define('output_dims', 0, 'Depth of the output.')
    return p

  def create_layer_variables(self) -> None:
    super().create_layer_variables()
    p = self.params
    self.create_variable(
        'var01',
        base_layer.weight_params(
            shape=[p.input_dims, p.output_dims],
            init=p.params_init,
            dtype=p.dtype))
    # var02 is not used.
    self.create_variable(
        'var02',
        base_layer.weight_params(
            shape=[p.input_dims, p.output_dims],
            init=p.params_init,
            dtype=p.dtype))

  def fprop(self, theta: NestedMap, inputs: JTensor) -> JTensor:
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
    bn_params = normalizations.BatchNorm.Params().Set(
        name='bn', dim=p.input_dims)
    self.create_child('bn', bn_params)

  def create_layer_variables(self) -> None:
    super().create_layer_variables()
    p = self.params
    self.create_variable(
        'var01',
        base_layer.weight_params(
            shape=[p.input_dims, p.output_dims],
            init=p.params_init,
            dtype=p.dtype))
    # var02 is not used.
    self.create_variable(
        'var02',
        base_layer.weight_params(
            shape=[p.input_dims, p.output_dims],
            init=p.params_init,
            dtype=p.dtype))

  def compute_predictions(self, theta: NestedMap, inputs: JTensor) -> JTensor:
    in_normed = self.bn.fprop(theta.bn, inputs)
    return jnp.einsum('bi,io->bo', in_normed, theta.var01)

  def compute_loss(self, theta: NestedMap, predictions: JTensor,
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
    p.Define('linear_p', linears.Linear.Params(),
             'Params for the linear layer.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    params = p.linear_p
    params.input_dims = p.input_dims
    params.output_dims = p.output_dims
    self.create_child('linear', params)

  def compute_predictions(self, theta: NestedMap,
                          input_batch: NestedMap) -> JTensor:
    return self.linear.fprop(theta.linear, input_batch.inputs)

  def compute_loss(self, theta, predictions, input_batch):
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
    bn_params = normalizations.BatchNorm.Params().Set(
        name='bn', dim=p.input_dims)
    self.create_child('bn', bn_params)

  def compute_predictions(self, theta: NestedMap,
                          input_batch: NestedMap) -> JTensor:
    return self.bn.fprop(theta.bn, input_batch.inputs)

  def compute_loss(self, theta: NestedMap, predictions: JTensor,
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
    p.Define('xformer_ffw', transformers.TransformerFeedForward.Params(),
             'Xformer feedforward layer params.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self.create_child('ffwd', p.xformer_ffw)

  def compute_predictions(self, theta: NestedMap, inputs: NestedMap) -> JTensor:
    return self.ffwd.fprop(theta.ffwd, inputs)

  def compute_loss(self, theta: NestedMap, predictions: JTensor,
                   input_batch: NestedMap) -> Tuple[NestedMap, NestedMap]:
    loss = jnp.mean(jnp.square(predictions))
    per_example_out = NestedMap(predictions=predictions)
    return NestedMap(loss=(loss, jnp.array(1.0, loss.dtype))), per_example_out
