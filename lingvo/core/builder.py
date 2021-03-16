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
# ==============================================================================
"""A library to build composite layers.

WARNING:
  The builder pattern is still experimental and we need to gain experience
  on when to use and when not to use.
  Please discuss w/ teammates before using it to build complicated
  layers.
"""

import functools
from lingvo.core import activations
from lingvo.core import builder_layers
from lingvo.core import hyperparams
from lingvo.core import layers
from lingvo.core import py_utils
from lingvo.core import tshape


class Base:
  """Model builder with commonly used layers.

  A method in a builder class constructs a layer param.  FProp of a layer
  constructed by a builder takes a tuple of tf.Tensor (one or more) and returns
  a tuple of tf.Tensor (one or more). Even though certain layers support FProp
  argument being None (e.g., Conv2DLayer), builder should not depend on such a
  support.

  The constructed layer is often a composition of multiple sub-layers connected
  in certain patterns. We expect to have a few methods to facilitate building
  these patterns. For example, _Seq() helps to build a sequential layer that
  calls its sub-layer one after another.

  TODO(zhifengc): Adds a more concrete example.
  """

  @classmethod
  def Params(cls):
    """The params of this layer."""
    p = hyperparams.InstantiableParams(cls)
    p.Define('deterministic_dropout', False,
             'Used deterministic dropout or not.')
    p.Define(
        'fprop_dtype', None,
        'Activations datatype to use. To enable bfloat16 activations for '
        'layers built using model builder, set fprop_dtype to '
        'tf.bfloat16, which will be propagated to layers that support '
        'bfloat16 activations. Default is None, which will use float32 '
        'activations.')
    # SPMD partition related params.
    p.Define(
        'device_mesh', None,
        'A numpy.ndarray specifying the topology of a device mesh to place the '
        'computations onto. If device_mesh is None, it is assumed to be a '
        'single device. Here are some examples: '
        'np.array([0, 1, 2, 3, 4, 5, 6, 7]) which is a 1d mesh with 8 devices, '
        'np.array([[0, 1, 2, 3], [4, 5, 6, 7]]) which is 2d matrix of 8 '
        'devices.')
    p.Define(
        'weight_split_dims_mapping', None,
        'Relevant only if device_mesh above is not None. If not None, it '
        'specifies how weight of this layer or those of the sublayers should '
        'be sharded over device mesh. ')
    p.Define(
        'activation_split_dims_mapping', None,
        'Relevant only if device_mesh above is not None. If not None, it '
        'specifies how activation of this layer or those of the sublayers '
        'should be sharded over device mesh. ')
    return p

  @property
  def params(self):
    """Returns the params upon which this layer is built."""
    return self._params

  def __init__(self, params):
    # Sub-classes should put some options common to many layers in __init__.
    self._params = params.Copy()

  ######################################################################
  # Layers to compose multiple layers.
  #
  # Sub-classes are discouraged to override these composition method.
  ######################################################################
  def _Rep(self, name, repeat, *subs):
    r"""Connects sub-layers sequentially and repeat multiple times.

    E.g., _Rep('foo', 2, sa, sb, sc) constructs a layer with 6 layers
    sequentially connected: [sa1, sb1, sc1, sa2, sb2, sc2].  sa1 and sa2 have
    the same structure as the given sa, but sa1 and sa2 do not share the same
    weight.

    Args:
      name: The layer name.
      repeat: Repeat \*subs this many times in the compose layer.
      *subs: A list of sub-layers.

    Returns:
      The param for the composed layer.
    """
    iterations = []
    for i in range(repeat):
      iterations.append(self._Seq('iter_%03d' % i, *[p.Copy() for p in subs]))
    return self._Seq(name, *iterations)

  def _Seq(self, name, *subs):
    """Connects sub-layers sequentially."""
    return builder_layers.SequentialLayer.Params().Set(
        name=name, sub=list(subs))

  def _Graph(self, name, input_endpoints, output_endpoints,
             *signature_sub_param_list):
    """Connects sub-layers into a data flow graph."""
    return builder_layers.GraphLayer.Params().Set(
        name=name,
        input_endpoints=input_endpoints,
        output_endpoints=output_endpoints,
        sub=list(signature_sub_param_list))

  def _Id(self, name):
    """Identity. (t_1, ..., t_n) -> (t1, ..., t_n)."""
    return self._Seq(name)

  def _Arg(self, name, index):
    """Picks index-th element. (t_1, ..., t_n) -> (t_{index},)."""
    return builder_layers.ArgIndexLayer.Params().Set(name=name, idx=[index])

  def _Par(self, name, *subs):
    """y = (f1, f2, ..., fn)(x).

    We feed the input tuple to all sub-layers and concatenates their output
    tuples into one tuple.

    Args:
      name: The layer name.
      *subs: A list of sub-layers.

    Returns:
      The param for the composed layer.
    """

    def ConcatTuples(tuples):
      # tuples is a list of tuples.
      return tuple(functools.reduce(lambda x, y: x + list(y), tuples, []))

    def ConcatMeta(tuples):
      return py_utils.NestedMap(
          flops=0,
          out_shapes=tuple(
              functools.reduce(lambda x, y: x + list(y), tuples, [])))

    return builder_layers.ParallelLayer.Params().Set(
        name=name, sub=list(subs), merge=ConcatTuples, merge_meta=ConcatMeta)

  def _Fn(self, name, fn, fn_out=None, fn_flops=None):
    """y = fn(x).

    Applies a fn: tuple(Tensor) -> a single Tensor or tuple(Tensor) to the input
    tuple.  Typically, fn is a very simple python function. This layer can be
    used for prototyping but we advice to implement the logic as a sub-class of
    BaseLayer for all established layers as FnLayer can't be serialized.

    Args:
      name: The layer name.
      fn: A lambda tuple(Tensor) -> tuple(Tensor).
      fn_out: A lambda tuple(tshape.Shape) -> output tuple(tshape.Shape)
      fn_flops: A lambda tuple(tshape.Shape) -> estimated flops of fn.
        If None, we assume flops == sum of elements in the inputs.

    Returns:
      The param for the composed layer.
    """

    def FnMeta(*shapes):
      """A lambda tuple(tshape.Shape) -> NestedMap{flops, out_shapes}."""
      if fn_out:
        out_shapes = fn_out(*shapes)
        if isinstance(out_shapes, tshape.Shape):
          out_shapes = (out_shapes,)
      else:
        out_shapes = shapes
      if fn_flops:
        flops = fn_flops(*shapes)
      else:
        flops = sum([s.size for s in shapes])
      return py_utils.NestedMap(flops=flops, out_shapes=out_shapes)

    return builder_layers.FnLayer.Params().Set(name=name, fn=fn, fn_meta=FnMeta)

  def _Save(self, name):
    """Returns a layer from which the activation and gradient can be accessed."""
    return layers.FetchLayer.Params().Set(name=name)

  def _AddFetches(self, name, body, fetches):
    """Fetches saved activations in the body sub-layer.

    E.g.:
    _AddFetches('foo', _Seq( 'stack', _Layer('layer1', ...),
    _Save('layer1_out', ...), _Layer('layer2', ...), _Save('layer2_out', ...),
    _Output('output', ...)), ['layer1_out', 'layer2_out'])

    The layer returns the stack's final output together with intermediate
    activations from layer1_out and layer2_out.

    Args:
      name: This layer's name.
      body: The sub-layer.
      fetches: A list of fetch names inside the sub-layer body.

    Returns:
      A layer whose outputs correspond to the activations of fetch points
      in the sub-layer body. [input1, input2, ..., inputN, fetch1, ..., fetchM].
    """
    return builder_layers.BranchLayer.Params().Set(
        name=name, body=body, fetches=fetches)

  def _Rematerialize(self, name, body):
    """Forces rematerialization on FProp of the body layer."""
    return builder_layers.RematerializationLayer.Params().Set(
        name=name, body=body)

  def _BatchParallel(self, name, sub):
    """Splits the batch and compute the forward pass on multiple devices.

    Args:
      name: This layer's name.
      sub: The sub-layer.

    Returns:
      A BatchParallel layer which splits the batch and computes the forward pass
      on multiple devices.
    """
    return builder_layers.BatchParallelLayer.Params().Set(name=name, sub=sub)

  def _PrintShape(self, name):
    """Print FProp input shape information."""
    return builder_layers.PrintShapeLayer.Params().Set(name=name)

  def _CreateNestedMap(self, name, keys):
    """Returns a NestedMap with keys from fprop args."""
    return builder_layers.CreateNestedMapLayer.Params().Set(
        name=name, keys=keys)

  ###########################################################################
  # Basic nn layers.
  #
  # The following method returns a layer param, whose FProp takes a single
  # Tensor and returns a single Tensor.
  #
  # These methods are designed to have minimal knobs. Sub-classes which needs to
  # be flexible can override these methods with different options. E.g., a
  # sub-class builder can override _BN() to tune the decay option.
  ###########################################################################
  def _BN(self, name, dims):
    """Batch norm."""
    return layers.BatchNormLayer.Params().Set(name=name, dim=dims, decay=0.99)

  def _LN(self, name, dims, use_fused_layernorm=False):
    """Layer norm."""
    return layers.LayerNorm.Params().Set(
        name=name,
        input_dim=dims,
        use_fused_layernorm=use_fused_layernorm,
        fprop_dtype=self.params.fprop_dtype)

  def _Dropout(self, name, keep_prob, noise_shape_broadcast_dims=None):
    """Returns a DropoutLayer Params."""
    if self.params.deterministic_dropout:
      return layers.DeterministicDropoutLayer.Params().Set(
          name=name,
          keep_prob=keep_prob,
          noise_shape_broadcast_dims=noise_shape_broadcast_dims)
    return layers.DropoutLayer.Params().Set(
        name=name,
        keep_prob=keep_prob,
        noise_shape_broadcast_dims=noise_shape_broadcast_dims,
        fprop_dtype=self.params.fprop_dtype)

  def _Linear(self,
              name,
              idims,
              odims,
              device_mesh=None,
              weight_split_dims_mapping=None):
    """Linear layer. y = matmul([..., idims], [idims, odims])."""
    return builder_layers.LinearLayer.Params().Set(
        name=name,
        input_dims=idims,
        output_dims=odims,
        fprop_dtype=self.params.fprop_dtype,
        device_mesh=device_mesh,
        weight_split_dims_mapping=weight_split_dims_mapping)

  def _Bias(self, name, dims, device_mesh=None, weight_split_dims_mapping=None):
    """Bias layer. The bias is added to the last dimension of the input."""
    return builder_layers.BiasLayer.Params().Set(
        name=name,
        dims=dims,
        fprop_dtype=self.params.fprop_dtype,
        device_mesh=device_mesh,
        weight_split_dims_mapping=weight_split_dims_mapping)

  def _Activation(self, name, fn='RELU'):
    """Activation layer."""
    return activations.ActivationLayer.Params().Set(activation=fn, name=name)

  def _FC(self, name, idims, odims):
    """Feed-forward fully connected. y = relu(matmul(x, w) + b)."""
    # pyformat: disable
    return self._Seq(
        name,
        self._Linear('linear', idims, odims),
        self._Bias('bias', odims),
        self._Activation('act'))

  def _MLP(self, name, dims):
    """Multiple layers of feed-forward fully connected.

    Args:
      name: The layer name.
      dims: A list of int. i-th layer has dims[i] as its input dimension, and
        dims[i+1] as its output dimensions.

    Returns:
      The param for the composed layer.
    """
    l = []
    for n, (i, o) in enumerate(zip(dims[:-1], dims[1:])):
      l += [self._FC('l%03d' % n, i, o)]
    return self._Seq(name, *l)

  def _Conv2D(self, name, filter_shape, filter_stride):
    """Conv2D layer."""
    return layers.Conv2DLayerNoPadding.Params().Set(
        name=name, filter_shape=filter_shape, filter_stride=filter_stride,
        fprop_dtype=self.params.fprop_dtype)
