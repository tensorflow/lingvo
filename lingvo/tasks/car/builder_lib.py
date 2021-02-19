# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""A collection of helper functions to build lingvo layer params."""

import functools
import math

from lingvo import compat as tf
from lingvo.core import builder_layers
from lingvo.core import layers
from lingvo.core import py_utils
from lingvo.tasks.car import car_layers

import numpy as np

# Keys for NestedMap for points: points, features, and padding.
POINTS_KEY = 'points'
FEATURES_KEY = 'features'
PADDING_KEY = 'padding'

# Keys for NestedMap for range images: xyz, features, and mask.
XYZ_KEY = 'xyz'
MASK_KEY = 'mask'


################################################################################
# Initializations
################################################################################
def TruncatedGaussianInit(filter_shape):
  factor = 2.0
  trunc_stddev = math.sqrt(1.3 * factor / filter_shape[-1])
  return py_utils.WeightInit.TruncatedGaussian(scale=trunc_stddev)


def KaimingUniformFanInRelu(shape):
  del shape
  return py_utils.WeightInit.KaimingUniformFanInRelu()


################################################################################
# Lingvo Layer Builders
################################################################################
# pyformat: disable
class ModelBuilderBase:
  """Model builder with commonly used layers."""

  def __init__(self):
    self.conv_init_method = TruncatedGaussianInit
    self.linear_params_init = None
    self.bn_params_init = None
    self.activation_fn = tf.nn.relu
    self.fc_bn_after_linear = False

  def _Rep(self, name, repeat, *subs):
    """Helper to construct a sequential layer repeated several time."""
    return builder_layers.SequentialLayer.Params().Set(
        name=name, repeat=repeat, sub=list(subs))

  def _Seq(self, name, *subs):
    return builder_layers.SequentialLayer.Params().Set(
        name=name, repeat=1, sub=list(subs))

  def _Branch(self, name, body, fetches):
    return builder_layers.BranchLayer.Params().Set(
        name=name, body=body, fetches=fetches)

  def _BN(self, name, dims):
    bn_params = layers.BatchNormLayer.Params().Set(
        name=name,
        dim=dims,
        decay=0.99,
        # TODO(b/148537111): consider setting this to True.
        add_stats_to_moving_average_variables=False)
    if self.bn_params_init:
      bn_params = bn_params.Set(params_init=self.bn_params_init)
    return bn_params

  def _Linear(self, name, idims, odims, params_init=None):
    linear_params = builder_layers.LinearLayer.Params().Set(
        name=name, input_dims=idims, output_dims=odims)

    params_init = params_init or self.linear_params_init
    if params_init is not None:
      linear_params = linear_params.Set(params_init=params_init)

    return linear_params

  def _Bias(self, name, dims, params_init=None):
    bias_params = builder_layers.BiasLayer.Params().Set(name=name, dims=dims)
    if params_init is not None:
      bias_params = bias_params.Set(params_init=params_init)
    return bias_params

  # TODO(tilarids): Consider using lingvo.core.activations.
  def _Activation(self, name, activation_fn):
    return builder_layers.MapLayer.Params().Set(name=name, fn=activation_fn)

  def _Relu(self, name):
    return self._Activation(name, activation_fn=tf.nn.relu)

  def _Swish(self, name):
    return self._Activation(name, activation_fn=tf.nn.swish)

  def _Sigmoid(self, name):
    return self._Activation(name, activation_fn=tf.nn.sigmoid)

  def _FC(self, name, idims, odims, use_bn=True, activation_fn=None):
    """Fully connected layer, with optional batch norm."""
    activation_fn = activation_fn or self.activation_fn
    if isinstance(idims, (list, tuple)):
      idims = idims[0]
    fc_layers = [self._Linear('linear', idims, odims)]
    if use_bn and self.fc_bn_after_linear:
      # Note that bn should use odims, since after linear.
      fc_layers = fc_layers + [self._BN('bn', odims)]
    elif use_bn and (not self.fc_bn_after_linear):
      # Note that bn should use idims, since before linear.
      fc_layers = [self._BN('bn', idims)] + fc_layers
    else:
      # Add bias since no batch norm that folds in bias.
      fc_layers = fc_layers + [self._Bias('bias', odims)]
    fc_layers += [self._Activation('activation', activation_fn=activation_fn)]

    return self._Seq(name, *fc_layers)

  def _MLP(self, name, dims, use_bn=True, activation_fn=None):
    l = []
    for n, (i, o) in enumerate(zip(dims[:-1], dims[1:])):
      l += [self._FC('l%03d' % n, i, o, use_bn=use_bn,
                     activation_fn=activation_fn)]
    return self._Seq(name, *l)

  def _Map(self, name, fn, **kwargs):
    return builder_layers.MapLayer.Params().Set(name=name, fn=fn, **kwargs)

  def _Max(self, name):
    return self._Map(name=name, fn=tf.reduce_max, kwargs={'axis': -2})

  def _Reshape(self, name, shape):
    return builder_layers.MapLayer.Params().Set(
        name=name, fn=tf.reshape, kwargs={'shape': shape})

  def _Matmul(self, name, *subs):
    def ParFn(*xs):
      result = xs[0]
      for v in xs[1:]:
        result = tf.matmul(result, v)
      return result
    return self._Par(name, ParFn, *subs)

  def _GLU(self, name, idims, odims):

    def Gate(x):
      u, v = tf.split(x, 2, axis=-1)
      return u * tf.sigmoid(v)

    return self._Seq(
        name,
        self._Linear('linear', idims, odims * 2),
        self._Bias('bias', odims * 2),
        builder_layers.MapLayer.Params().Set(name='gate', fn=Gate))

  def _Dropout(self, name, keep_prob):
    return layers.DropoutLayer.Params().Set(name=name, keep_prob=keep_prob)

  def _Fetch(self, name):
    return layers.FetchLayer.Params().Set(name=name)

  def _FirstN(self, name, n):
    """Return the first n args."""
    return builder_layers.FirstNLayer.Params().Set(name=name, n=n)

  def _ArgIdx(self, name, index):
    return builder_layers.ArgIndexLayer.Params().Set(name=name, idx=index)

  def _Join(self, name, *subs):
    r"""Juxtapose outputs from \*subs in a single output tuple."""

    def Join(xs):
      arg_lists = [list(x) for x in xs]
      result = arg_lists[0]
      for v in arg_lists[1:]:
        result = result + v
      return tuple(result)

    return builder_layers.ParallelLayer.Params().Set(
        name=name, sub=list(subs), merge=Join)

  def _Par(self, name, fn, *subs):
    """Helper to construct a parallel layer which merge branches."""

    def _Merge(xs):
      rets = []
      for ys in zip(*xs):
        rets.append(fn(*ys))
      return tuple(rets)

    # TODO(zhifengc): fill merge_meta to compute flops and output shape, etc.
    return builder_layers.ParallelLayer.Params().Set(
        name=name, sub=list(subs), merge=_Merge)

  def _Concat(self, name, *subs):
    r"""Concatenate outputs from \*subs along the last dimensions."""
    return self._Par(name, lambda *xs: tf.concat(xs, axis=-1), *subs)

  def _BroadcastConcat(self, name, *subs):
    r"""Concatenate outputs from \*subs, broadcasting to match leading shape."""

    def _Merge(*xs):
      """Broadcast all dimensions except the last, and concat on last dim."""

      # Stack all shapes and take max on each dimension to get leading shape.
      leading_shape = tf.stack([tf.shape(x)[:-1] for x in xs])
      leading_shape = tf.reduce_max(leading_shape, axis=0)
      # Broadcast each x.
      broadcast_xs = []
      for x in xs:
        broadcast_shape = tf.concat([leading_shape, tf.shape(x)[-1:]], axis=0)
        broadcast_xs.append(tf.broadcast_to(x, broadcast_shape))

      # Concat on last dimension.
      concat_xs = tf.concat(broadcast_xs, axis=-1)

      return concat_xs

    return self._Par(name, _Merge, *subs)

  def _ApplyFnMulti(self, name, fn, *subs):
    """A common use case where every branch produces a single output tensor."""
    return self._Par(name, fn, *subs)

  def _ApplyInParallelAndMerge(self, name, merge_fn, *subs):
    """Applies the subs in parallel to the given input and merges their outputs.

    Args:
      name: String layer name.
      merge_fn: Function to merge the outputs of `subs`, which will be a
        flattened list of subs outputs. For example, if there were 3 subs with
        outputs (out1,), (out21, out22), (out31, out32), the output will be
        [out1, out21, out22, out31, out32].
      *subs: Modules to be applied in parallel to the input

    Returns:
      Params for this layer.
    """

    return self._Seq(
        name,
        self._Join('parallel_subs', *subs),
        self._ApplyFn('merge', merge_fn))

  def _ApplyFn(self, name, fn):
    """Apply fn over the input tuple."""
    return builder_layers.ParallelLayer.Params().Set(
        name=name, sub=[self._Seq('id')], merge=lambda xs: (fn(*(xs[0])),))

  def _MakeInputFeatureFromPoints(self, name):
    """Transforms points to features using the points with constant padding.

    Layer input/output shapes: [N x P x 3] -> ([N x P x 3], [N x P x 4])

    Args:
      name: String layer name.

    Returns:
      Params for this layer.
    """

    def PadOne(inp):
      inp = py_utils.HasShape(inp, [-1, -1, 3])
      return tf.pad(inp, [[0, 0], [0, 0], [0, 1]], constant_values=1.0)

    return self._Join(name, self._Seq('id'), self._ApplyFn('pad', fn=PadOne))

  def _Squeeze(self, name, axis=None):
    def _SqueezeFn(x):
      return tf.squeeze(x, axis=axis)
    return self._ApplyFn(name, fn=_SqueezeFn)

  def _ConvPlain(self, name, filter_shape, filter_stride=(1, 1),
                 padding='SAME', conv_init_method=None):
    conv_init_method = conv_init_method or self.conv_init_method(filter_shape)
    return layers.Conv2DLayerNoPadding.Params().Set(
        name=name,
        filter_shape=filter_shape,
        filter_stride=filter_stride,
        padding=padding,
        params_init=conv_init_method)

  def _DeconvPlain(self, name, filter_shape, filter_stride=(1, 1)):
    return layers.DeconvLayer.Params().Set(
        name=name,
        filter_shape=filter_shape,
        filter_stride=filter_stride,
        params_init=self.conv_init_method(filter_shape))

  def _Conv(self, name, filter_shape, stride=(1, 1), padding='SAME',
            use_bn=True):
    """Helper to construct a conv/normalize/actvation layer."""
    # TODO(zhifengc): Revisit whether BatchNormLayer should apply gamma when the
    # following activation is a relu.
    if isinstance(stride, tuple):
      filter_stride = stride
    elif isinstance(stride, int):
      filter_stride = (stride, stride)
    else:
      raise ValueError(
          'Input stride not a tuple or int. Is a {}'.format(type(stride)))
    norm = self._BN('bn', filter_shape[3]) if use_bn else self._Identity(name)
    return self._Seq(
        name,
        self._ConvPlain('conv', filter_shape, filter_stride, padding),
        norm,
        self._Relu('relu'))

  def _Identity(self, name):
    """Apply identity transformation."""
    return layers.IdentityLayer.Params().Set(name=name)

  def _Shortcut(self, name, idims, odims, stride):
    """Apply ResNet shortcut transformation."""
    if idims != odims or stride != 1:
      return self._ConvPlain(name, (1, 1, idims, odims), stride)
    else:
      return self._Identity(name)

  def _ResidualLayer(self, name, filter_size, stride):
    """ResNet basic layer, as in https://arxiv.org/pdf/1512.03385.pdf Fig. 2.

    Args:
      name: string layer name.
      filter_size: tuple of integers (filter_height, filter_width, idims, odims)
        , which represent filter height, filter width, input channel size,
        output channel size.
      stride: integer of tuple of integers to apply to all dimensions when using
        2D convolution. This will be applied to the shortcut layer and the
        first convolution layer.

    Returns:
      Params for a residual layer.
    """
    filter_height, filter_width, idims, odims = filter_size
    repeated_filter_size = (filter_height, filter_width, odims, odims)
    return self._Seq(
        name,
        self._Add(
            'add',
            self._Shortcut('shortcut', idims, odims, stride),
            self._Seq(
                'residual',
                self._Conv('conv_a', filter_size, stride),
                self._ConvPlain('conv_b', repeated_filter_size),
                self._BN('residual_bn', odims))),
        self._Relu('relu_add'))

  def _ResidualBlock(self, name, filter_size, stride, repeats):
    """ResNet block that downsamples at the beginning.

    Args:
      name: string block name.
      filter_size: tuple of integers (filter_height, filter_width, idims, odims)
        , which represent filter height, filter width, input channel size,
        output channel size.
      stride: integer of tuple of integers to apply to all dimensions when using
        2D convolution. It is applied only to the first downsampling layer for
        efficiency.
      repeats: integer number of residual layers to stack.

    Returns:
      Params for a residual block.
    """
    filter_height, filter_width, _, odims = filter_size
    repeated_filter_size = (filter_height, filter_width, odims, odims)
    return self._Seq(
        name,
        self._ResidualLayer('res3x3_0', filter_size, stride),
        self._Rep(
            'rep',
            repeats,
            self._ResidualLayer('res3x3_1', repeated_filter_size, 1)),
        self._Fetch('final'))

  ##########################################
  # Self-attention stack.
  #
  # NOTE: layers_with_attention has TransformerLayers.  We do not use that
  # directly because those layers assumes input tensors are in [time, batch,
  # dims] so that it's convenient for left-to-right decoding use case.  Point
  # cloud modeling is unlikely need that.
  #
  # We also restrict the attention type to the simplest dot-attention for now.
  ##########################################
  def _SelfAttenStack(self, name, depth, dims, hdims, heads, keep_prob):
    """A self-attentional stack.

    Args:
      name: string layer name.
      depth: int. The number of transformer layers.
      dims: int. The dimension of each transformer layer.
      hdims: int. The hidden dimension of each transformer layer.
        Typically, hdims >= dims.
      heads: int. The number of attention heads.
      keep_prob: Dropout keep probability.

    Returns:
      Params for this layer.
    """
    return self._Rep(
        name, depth, self._Atten('atten', dims, hdims, heads, keep_prob))

  def _Atten(self, name, dims, hdims, heads, keep_prob=1.):
    """Transformer self-attention layer."""
    return self._Seq(
        name,
        self._AttenSelf('self', dims, hdims, heads, keep_prob),
        self._AttenFF('ff', dims, hdims, keep_prob))

  def _LN(self, name, dims):
    """Layer norm."""
    return layers.LayerNorm.Params().Set(name=name, input_dim=dims)

  def _Project(self, name, idims, odims):
    """Project layer. Simply X*W+b."""
    return self._Seq(
        name,
        self._Linear('linear', idims, odims),
        self._Bias('bias', odims))

  def _Add(self, name, lhs, rhs):
    """Add two branches. out = lhs(in) + rhs(in)."""
    return self._ApplyFnMulti(name, lambda x, y: x + y, lhs, rhs)

  def _Multiply(self, name, lhs, rhs):
    """Multiply two branches. out = lhs(in) * rhs(in)."""
    return self._ApplyFnMulti(name, lambda x, y: x * y, lhs, rhs)

  def _AttenFF(self, name, dims, hdims, keep_prob=1.):
    """Transformer feed-forward layer."""
    return self._Seq(
        name,
        self._Add(
            'residual',
            self._Seq('id'),
            self._Seq(
                'ff',
                self._LN('ln', dims),
                self._FC('fc', dims, hdims, use_bn=False),
                self._Project('proj', hdims, dims),
                self._Dropout('dropout', keep_prob))))

  def _AttenSelf(self, name, dims, hdims, heads, keep_prob=1.):
    """Dot-attention, multiple heads, self-attention."""
    assert hdims % heads == 0, 'hdims={} heads={}'.format(hdims, heads)

    def _Atten(query, key, val):
      """Returns weighted val based on dot-attention between query and key."""
      b, n, _ = py_utils.GetShape(query)

      # Query.
      query = py_utils.HasShape(query, [b, n, hdims])
      query = tf.reshape(query, [b, n, heads, hdims // heads])
      query = tf.transpose(query, [0, 2, 1, 3])

      # Key.
      key = py_utils.HasShape(key, [b, n, hdims])
      key = tf.reshape(key, [b, n, heads, hdims // heads])
      key = tf.transpose(key, [0, 2, 1, 3])

      # query:[b, heads, n, hdims // heads]
      # key:  [b, heads, n, hdims // heads]^T
      dotp = tf.matmul(query, key, transpose_b=True)
      probs = tf.nn.softmax(dotp)
      probs = py_utils.HasShape(probs, [b, heads, n, n])

      # value (aka. context)
      val = py_utils.HasShape(val, [b, n, hdims])
      val = tf.reshape(val, [b, n, heads, hdims // heads])
      val = tf.transpose(val, [0, 2, 1, 3])
      val = py_utils.HasShape(val, [b, heads, n, hdims // heads])

      # Weighted average of value (context). [b, heads, n, hdims // heads]
      out = tf.matmul(probs, val)
      out = tf.transpose(out, [0, 2, 1, 3])
      out = tf.reshape(out, [b, n, hdims])
      return out

    return self._Seq(
        name,
        self._Add(
            'residual',
            self._Seq('id'),
            self._Seq(
                'lapd',
                self._LN('ln', dims),
                self._ApplyFnMulti(
                    'atten',
                    _Atten,
                    self._Project('query', dims, hdims),
                    self._Project('key', dims, hdims),
                    self._Project('value', dims, hdims)),
                self._Project('post', hdims, dims),
                self._Dropout('dropout', keep_prob))))

  ##########################################
  # Location and Padding Aware Layers and Helpers
  #
  # These layers expect as input a NestedMap containing the following keys:
  #
  #   points: tensor with shape [..., 3] containing xyz coordinates
  #   features: tensor with shape [..., C] containing features for each point
  #   padding: tensor with same leading shape containing 0/1 with
  #     1 representing a padded point, and 0 representing a real point.
  #
  ##########################################
  class _Decorators:
    """Internal decorators for builder functions."""

    @classmethod
    def ExpectsNestedMapTensor(cls, expected_keys=()):
      """Adds a validation layer before the layer produced by builder_fn.

      Args:
        expected_keys: A string, or an iterable of strings that are expected to
          be in the input NestedMap.

      Returns:
        A decorator function that can be applied on builder functions.
      """

      if isinstance(expected_keys, str):
        expected_keys = [expected_keys]
      expected_keys = set(expected_keys)

      def _ValidateFn(inp):
        """Validate that inp contains the expected keys."""
        if not isinstance(inp, py_utils.NestedMap):
          raise ValueError('Input not a `NestedMap`. Is a {}'.format(type(inp)))
        input_keys = set(inp.keys())
        if not expected_keys.issubset(input_keys):
          missing_keys = expected_keys - input_keys
          raise ValueError('Input is missing key(s): {}'.format(
              ', '.join(missing_keys)))
        return inp

      def _Decorator(builder_fn):
        def _DecoratedFn(self, *args, **kwargs):
          p = builder_fn(self, *args, **kwargs)
          # pylint: disable=protected-access
          return self._Seq('validated_%s' % p.name,
                           self._ApplyFn('validate', fn=_ValidateFn), p)
          # pylint: enable=protected-access
        return _DecoratedFn

      return _Decorator

    @classmethod
    def ExpectsNestedMapPointsTensor(cls, builder_fn):
      """Adds a validation layer before the layer produced by builder_fn."""
      return cls.ExpectsNestedMapTensor(
          expected_keys=(POINTS_KEY, FEATURES_KEY, PADDING_KEY))(builder_fn)

    @classmethod
    def ExpectsNestedMapRangeImage(cls, builder_fn):
      """Adds a validation layer before the layer produced by builder_fn."""
      return cls.ExpectsNestedMapTensor(
          expected_keys=(XYZ_KEY, FEATURES_KEY, MASK_KEY))(builder_fn)

  @_Decorators.ExpectsNestedMapTensor()
  def _GetValue(self, name, key, default=None):
    """Expects a NestedMap as input and produces the value for the given key."""

    def GetValueFn(inp):
      if not isinstance(inp, py_utils.NestedMap):
        raise ValueError('At layer with name={}. Unable to retrieve key {}, '
                         'input not a `NestedMap`. Is a {}'.format(
                             name, key, type(inp)))
      return inp.get(key, default)

    return self._Map(name, fn=GetValueFn)

  @_Decorators.ExpectsNestedMapTensor()
  def _ParMap(self, name, key_to_sub):
    """Perform parallel layers and create a NestedMap from the outputs.

    Parallel branches on an input `NestedMap`. Each branch should expect the
    same `NestedMap` as input; each branch's output will be mapped to the
    specified key in key_to_sub.

    Args:
      name: String layer name.
      key_to_sub: Dictionary mapping keys to sub params. Each sub should expect
        a NestedMap input.

    Returns:
      Params for this layer.
    """
    sorted_keys = sorted(key_to_sub.keys())
    sorted_subs = [key_to_sub[k] for k in sorted_keys]

    def _MakeNestedMap(*vals):
      return py_utils.NestedMap(dict(zip(sorted_keys, vals)))

    return self._ApplyFnMulti(name, _MakeNestedMap, *sorted_subs)

  @_Decorators.ExpectsNestedMapTensor()
  def _SeqToKey(self, name, key, *subs):
    """Apply sequence to only update the value at a specific key.

    Note: This is the more general version of SeqOnKey. SeqOnKey automatically
    extracts the key value before running subs, while SeqToKey passes the entire
    input through.

    This function expects a NestedMap as an input, and applies the provided
    subs on the entire NestedMap, but only updates the value at the specified
    key. The resulting value (tensor or NestedMap) from these subs is placed as
    the new value for the specified key in the original NestedMap. The rest of
    the key, value pairs are returned untouched.

    Args:
      name: string layer name.
      key: string key to update after running sequence on.
      *subs: A list of sub layer Params that should expect the input NestedMap.
        The output of the subs will be placed back to the specified key.
    Returns:
      Params for this layer.
    """

    def _Merge(outputs):
      """Merges the outputs from ParallelLayer.

      Args:
        outputs: A tuple of two elements: (a) the original input NestedMap and
          (b) the outputs from applying the sub layers in subs.

      Returns:
        A new NestedMap map with the specified key's value replaced with the
        result from subs.
      """

      input_map = outputs[0][0]
      seq_result = outputs[1][0]
      new_map = input_map.DeepCopy()
      new_map[key] = seq_result

      return (new_map,)

    return builder_layers.ParallelLayer.Params().Set(
        name=name, sub=[
            self._Seq('id'),
            self._Seq('key_seq',
                      *subs)
        ], merge=_Merge)

  def _SeqOnKey(self, name, key, *subs):
    """Sequential layers operating on a specific key only."""
    return self._SeqToKey(name, key, self._GetValue('get_{}'.format(key), key),
                          *subs)

  @_Decorators.ExpectsNestedMapTensor(FEATURES_KEY)
  def _SeqOnFeatures(self, name, *subs):
    """Sequential layers operating on the features only."""
    return self._SeqOnKey(name, FEATURES_KEY, *subs)

  # TODO(jngiam): Padded* layers should all return nested outputs with padding.
  # Downstream callers should be fixed to handle the returned padding.
  @_Decorators.ExpectsNestedMapTensor((FEATURES_KEY, PADDING_KEY))
  def _PaddedMax(self, name, nested_output=False):
    """Padding aware max pooling layer, emits either a single tensor or map."""
    def _PaddedMaxFn(inp):
      """Apply padded max using reduce_max with paddings replaced by neginf."""
      # Replace all padded features with -inf.
      neginf_padding = tf.where(
          inp.padding > 0, -np.inf * inp.padding, inp.padding)
      features = inp.features + neginf_padding[..., tf.newaxis]
      features = tf.reduce_max(features, axis=-2)

      # Replace features of all padded points by zeros. If a batch of points are
      # all padded, then reduce_min over the padding will be 1. We set the
      # features to be zero, so that we don't get any downstream issue with
      # NaNs. Note that inf * 0 = NaN.
      padding = tf.reduce_min(inp.padding, axis=-1)
      features = tf.where_v2(tf.cast(padding[..., tf.newaxis], tf.bool),
                             tf.zeros_like(features), features)
      features = py_utils.CheckNumerics(features)

      if nested_output:
        return py_utils.NestedMap(features=features, padding=padding)
      else:
        return features

    return self._Map(name=name, fn=_PaddedMaxFn)

  @_Decorators.ExpectsNestedMapPointsTensor
  def _PaddedMean(self, name):
    """Padding aware mean pooling layer, emits a single tensor."""

    def _PaddedMeanFn(inp):
      """Apply padded mean using reduce_sum and dividing by # real points."""
      # Replace all padded features with 0 by masking the padded features out.
      mask = 1 - inp.padding
      features = inp.features * mask[..., tf.newaxis]
      features = tf.reduce_sum(features, axis=-2)
      num_real_points = tf.reduce_sum(mask, axis=-1, keepdims=True)
      # Prevent the divisor of our padded mean from ever being 0, so that
      # the gradient flowing back through this op doesn't give us NaNs.
      num_real_points = tf.maximum(num_real_points, 1)
      features = features / num_real_points

      # Replace features of all padded points by zeros. If a batch of points are
      # all padded, then num_real_points will be zero. We set the features to be
      # zero, so that we don't get any downstream issue with NaNs.
      # Note that inf * 0 = NaN.
      all_padded = tf.equal(num_real_points, 0.)
      all_padded = tf.broadcast_to(all_padded, py_utils.GetShape(features))
      features = tf.where(all_padded, tf.zeros_like(features), features)
      return py_utils.CheckNumerics(features)

    return self._Map(name=name, fn=_PaddedMeanFn)

  @_Decorators.ExpectsNestedMapPointsTensor
  def _PaddedSum(self, name):
    """Padding aware sum pooling layer, emits a single tensor."""

    def _PaddedSumFn(inp):
      # Replace all padded features with 0 by masking the padded features out.
      mask = 1 - inp.padding
      features = inp.features * mask[..., tf.newaxis]
      features = tf.reduce_sum(features, axis=-2)
      return features

    return self._Map(name=name, fn=_PaddedSumFn)

  @_Decorators.ExpectsNestedMapTensor(FEATURES_KEY)
  def _FeaturesFC(self, name, idims, odims, use_bn=True, activation_fn=None):
    """Applies a FC layer to `features` key, emits a `NestedMap`."""
    activation_fn = activation_fn or self.activation_fn
    bn_join_layer = self._Join(
        'join_features_padding',
        self._GetValue('get_features', FEATURES_KEY),
        self._Seq('expand_padding',
                  self._GetValue('get_padding', 'padding'),
                  self._ApplyFn('expand',
                                fn=lambda t: t[..., tf.newaxis])))
    if use_bn and self.fc_bn_after_linear:
      # Note that bn should use odims, since after linear.
      seq_p = [
          self._SeqToKey('linear_seq',
                         FEATURES_KEY,
                         self._GetValue('get_features', FEATURES_KEY),
                         self._Linear('linear', idims, odims)),
          bn_join_layer,
          self._BN('bn', odims),
      ]
    elif use_bn and (not self.fc_bn_after_linear):
      # Note that bn should use idims, since before linear.
      seq_p = [
          bn_join_layer,
          self._BN('bn', idims),
          self._Linear('linear', idims, odims),
      ]
    else:
      # Add bias since no batch norm that folds in bias.
      seq_p = [
          self._GetValue('get_features', FEATURES_KEY),
          self._Linear('linear', idims, odims),
          self._Bias('bias', odims),
      ]
    seq_p += [self._Activation('activation', activation_fn=activation_fn)]
    return self._SeqToKey(name, FEATURES_KEY, *seq_p)

  @_Decorators.ExpectsNestedMapTensor(FEATURES_KEY)
  def _FeaturesMLP(self, name, dims, use_bn=True, activation_fn=None):
    l = []
    for n, (i, o) in enumerate(zip(dims[:-1], dims[1:])):
      l += [self._FeaturesFC('l%03d' % n, i, o, use_bn=use_bn,
                             activation_fn=activation_fn)]
    return self._Seq(name, *l)

  def _CondFC(self, name, idims, adims, odims, use_bn=True, activation_fn=None):
    """Conditional FC layer to use with GIN as a combiner.

    ([..., P, idims], [..., 1, adims]) -> [..., P, odims]

    This layer expects an input tuple (features, aggregate), where features
    contains per-point features, and aggregate is a global feature (e.g.,
    computed using max-pooling over all the points). The aggregate tensor
    is used to compute a linear transformation that is used in this FC layer.
    Each example in the batch has a different linear transformation. This layer
    is similar to the T-Net transformation in PointNet.

    Args:
      name: String name for this layer.
      idims: Dimension for last axis of features tensor.
      adims: Dimension for last axis of aggregate tensor.
      odims: Number of output dimensions.
      use_bn: Whether to enable batch norm.
      activation_fn: Optional override for the activation function. If None,
        defaults to the activation fn that the builder is initialized with.

    Returns:
      Params for a layer.
    """
    activation_fn = activation_fn or self.activation_fn

    def _ReshapeTransform(inp):
      """Reshape the transformation tensor to [..., idims, odims]."""
      base_shape = py_utils.GetShape(inp)[:-1]
      out_shape = list(base_shape) + [idims, odims]
      return tf.reshape(inp, out_shape)

    transform_net = self._Matmul(
        'cond_transform',
        self._Seq(
            'prep_features',
            self._ArgIdx('arg0', index=[0]),
            self._BN('bn', idims) if use_bn else self._Seq('id')),
        self._Seq(
            'compute_linear_transform',
            self._ArgIdx('arg1', index=[1]),
            self._Squeeze('squeeze', axis=-2),
            # TODO(jngiam): Consider other configurations for FC dims here.
            self._FC('fc0', adims, adims * 2, use_bn=use_bn,
                     activation_fn=activation_fn),
            # We use identity for the output since we want the transformation
            # matrix to have negative values too.
            self._FC('fc1', adims * 2, idims * odims, use_bn=use_bn,
                     activation_fn=tf.identity),
            self._ApplyFn('reshape', fn=_ReshapeTransform)))

    return self._Seq(
        name,
        transform_net,
        self._Seq('id') if use_bn else self._Bias('bias', odims),
        self._Activation('activation', activation_fn=activation_fn))

  def _GINCondFC(self, name, lhs, rhs, idims, adims, odims, use_bn=True,
                 activation_fn=None):
    return self._Seq(
        name,
        self._Join('join', lhs, rhs),
        self._CondFC('cond_fc', idims, adims, odims, use_bn=use_bn,
                     activation_fn=activation_fn))

  @_Decorators.ExpectsNestedMapPointsTensor
  def _GIN(self,
           name,
           mlp_dims,
           aggregate_sub,
           readout_sub,
           combine_method='add',
           eps=0.,
           use_bn=True):
    """Graph Isomorphism Network (GIN) [1].

    This implements the GIN network. The input goes through multiple
    GINIntermediateLayers which performs the update equation (eqn 4.1 in paper).

    The final network output is the concatenation of all the intermediate
    layer outputs (eqn 4.2 in paper), with readout aggregation being mean or
    max.

      output = concat([readout(f0), readout(f1),..., readout(fN)]),
      where f1 = intermediate_gin(g0), etc.

    This is akin to a DenseNet structure, where all the intermediate layers
    are short-circuited to the end.

    Using sum corresponds to the proposed method by [1], which enables learning
    graphs with repeated nodes. 'mean' and 'max' may confuse graphs with
    repeated nodes but could better capture distribution statistics or distinct
    elements.

    Note that eps=0 performs well in their results (Section 7).

    [1] How Powerful are Graph Neural Networks?
        Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka. ICLR 2019.
        https://arxiv.org/abs/1810.00826

    Args:
      name: String layer name.
      mlp_dims: List of list of ints representing the intermediate feature MLPs.
        The first dim should match the dim of the input features. Each output
        dim of the MLP should match the next.
      aggregate_sub: Params for a layer that has padding-aware aggregation
        (e.g., PaddedMean, PaddedMax, PaddedSum). This is used by the
        intermediate layers.
      readout_sub: Params for a layer that has padding-aware aggregation
        (e.g., PaddedMean, PaddedMax, PaddedSum). This is used to aggregate
        the feature vectors for representing the entire graph.
      combine_method: Either 'add', 'concat', or 'cond_fc'. This is used by the
        intermediate layers.
      eps: float scale parameter, see _GINIntermediateLayer for details.
      use_bn: Whether to enable batch norm in intermediate FC layers.

    Returns:
      Params for this layer. This layer expects a `NestedMap` with points,
      features, and padding.
    """
    if combine_method not in ['add', 'concat', 'cond_fc']:
      raise ValueError('Unexpected combine method: {}'.format(combine_method))

    # Validate mlp_dims; every adjacent MLP should have a matching number of
    # channels.
    for idx, (mlp_i, mlp_o) in enumerate(zip(mlp_dims[:-1], mlp_dims[1:])):
      prev_layer_dims = mlp_i[-1]
      next_layer_dims = mlp_o[0]
      # When concatenating, we expect the next layer to have double the input
      # channels.
      if combine_method == 'concat':
        prev_layer_dims *= 2
      if prev_layer_dims != next_layer_dims:
        raise ValueError(
            'mlp_dims do not match ({} != {}) at layer {} '
            'with dims: {} and {}'.format(
                idx, prev_layer_dims, next_layer_dims, mlp_i, mlp_o))

    # Build the network recursively in a way that concats all the
    # intermediate outputs.
    def _Build(depth):
      if depth == len(mlp_dims):
        return readout_sub
      else:
        return self._Concat(
            'gin_concat',
            readout_sub,
            self._Seq(
                'seq',
                self._GINIntermediateLayer(
                    'gin_intermediate',
                    mlp_dims[depth],
                    aggregate_sub,
                    combine_method,
                    eps,
                    use_bn),
                _Build(depth + 1)))

    p = _Build(depth=0)

    return self._Seq(name, p)

  @_Decorators.ExpectsNestedMapPointsTensor
  def _GINIntermediateLayer(
      self,
      name,
      dims,
      aggregate_sub,
      combine_method='add',
      eps=0.,
      use_bn=True):
    """Implements an intermediate GIN layer.

    This layer takes in a `NestedMap` and produces a `NestedMap` that contains
    points, features, and padding.

    Given features f_i for each point i, compute new features f'_i:

      f'_i = MLP((1 + eps) * f_i + aggregate(f_j))  where j are neighbors of i

    Note: This layer aggregates over the entire point cloud, not just the
    neighbors. In practice, aggregation over the neighbors can be done by having
    pre-grouping the points before invoking this layer.

    Args:
      name: String layer name.
      dims: List of int dims for the MLP layer.
      aggregate_sub: Params for a layer that has padding-aware aggregation
        (e.g., PaddedMean, PaddedMax, PaddedSum).
      combine_method: Either 'add', 'concat', or 'cond_fc'. Add follows the
        paper suggestion of adding the aggregated feature to each point. Concat
        broadcasts the aggregated features and concats it to each point. Cond_FC
        computes an example-dependent linear transformation; this is similar to
        T-Net.
      eps: float scale parameter, see equation above for details.
      use_bn: Whether to enable batch norm in FC layers.

    Returns:
      Params for this layer.
    """
    combine_method_to_fn = {
        'add': self._Add,
        'concat': self._BroadcastConcat,
        'cond_fc': functools.partial(
            self._GINCondFC,
            idims=dims[0], odims=dims[0], adims=dims[0],
            use_bn=use_bn)
    }
    if combine_method not in combine_method_to_fn:
      raise ValueError('Unexpected combine method: {}'.format(combine_method))
    combine_fn = combine_method_to_fn[combine_method]
    return self._Seq(
        name,
        self._SeqToKey(
            'map_features',
            FEATURES_KEY,
            combine_fn(
                'combine',
                self._Seq(
                    'left',
                    self._GetValue('get_features', FEATURES_KEY),
                    self._ApplyFn('eps_scale', fn=lambda t: (1. + eps) * t)),
                self._Seq(
                    'right',
                    aggregate_sub,
                    self._ApplyFn('expand_dims',
                                  fn=lambda t: tf.expand_dims(t, axis=-2))))),
        self._FeaturesMLP('mlp', dims, use_bn=use_bn))

  @_Decorators.ExpectsNestedMapPointsTensor
  def _ConcatPointsToFeatures(self, name):
    return self._SeqToKey(
        name,
        FEATURES_KEY,
        self._Concat(
            'concat',
            self._GetValue('get_points', 'points'),
            self._GetValue('get_features', FEATURES_KEY)))

  @_Decorators.ExpectsNestedMapPointsTensor
  def _SetAbstraction(self,
                      name,
                      feature_extraction_sub,
                      num_samples,
                      group_size,
                      ball_radius,
                      sample_neighbors_uniformly=True):
    """Set abstraction layer that samples, groups points, and featurizes groups.

    This is based on PointNet++ concept of set abstractions.

    This layer samples `num_samples` points using Farthest Point Sampling
    algorithm. For each sampled point, it forms a group of size `group_size`
    consisting the points within the distance of `ball_radius` from the sampled
    point.

    It then applies the feature_extraction_sub to the extracted groups, note
    that the tensors passed to the feature_extraction_sub will have leading
    shapes corresponding to [batch_size, num_query_points, num_points_per_group,
    ...].

    A simple feature_extraction_sub that does a MLP followed by max pooling
    on the grouped points could be composed using FeaturesMLP and PaddedMax.

    This layer expects a NestedMap of points, features, and padding; it produces
    a corresponding NestedMap with updated points, features, and padding.

    Args:
      name: name for the layer.
      feature_extraction_sub: Params for a layer that takes in a NestedMap of
        points, features, and padding. These tensors correspond to grouped
        points. Note that the leading shape of these tensors are [batch_size,
        num_query_points, num_points_per_group, ...]. This layer should produce
        a single tensor representing the features computed by summarizing the
        grouped points.
      num_samples: Number of points to be sampled.
      group_size: Number neighbours for each sampled point.
      ball_radius: The distance around each sampled point to obtain neighbours.
      sample_neighbors_uniformly: Whether to sample neighbors uniformly within
        the ball radius.

    Returns:
      Params for Set Abstraction layer based on PointNet++.
    """
    return self._Seq(
        name,
        car_layers.SamplingAndGroupingLayer.Params().Set(
            name='sample_group',
            num_samples=num_samples,
            ball_radius=ball_radius,
            group_size=group_size,
            sample_neighbors_uniformly=sample_neighbors_uniformly),
        # Output of sampling and grouping is a NestedMap of grouped_points and
        # query_points. `query_points` is a nestedmap consists of the sampled
        # points and padding. `grouped_points` consists of the groups around the
        # sampled points, corresponding features and padding.
        self._ParMap(
            'pmap',
            dict(
                points=self._Seq('seq_points',
                                 self._GetValue('q', 'query_points'),
                                 self._GetValue('p', 'points')),
                features=self._Seq(
                    'seq_features',
                    self._GetValue('q', 'grouped_points'),
                    feature_extraction_sub),
                padding=self._Seq('seq_padding',
                                  self._GetValue('q', 'query_points'),
                                  self._GetValue('p', 'padding')),
            )))  # pyformat: disable

  @_Decorators.ExpectsNestedMapPointsTensor
  def _PointConvParametricConv(self,
                               name,
                               mlp_dims,
                               num_in_channels,
                               num_out_channels):
    """Parametric convolution based on PointConv.

    Note that this layer assumes that the features have already been weighted
    by density. This layer follows Figure 5. in [1] but without the initial
    inverse density scale weighting part. When working with range images,
    one can use RIScaleFeaturesByDensity to scale the features beforehand.

    [1] PointConv: Deep Convolutional Networks on 3D Point Clouds. CVPR 2019.
        Wu, Wenxuan and Qi, Zhongang and Fuxin, Li.

    Args:
      name: string name for this layer.
      mlp_dims: dims for mlp applied to the points. the first dimension of
        mlp_dims must be 3.
      num_in_channels: integer number of input channels (features)
      num_out_channels: integer number of output channels (features).

    Returns:
      Params for a parametric conv layer.
    """
    if mlp_dims[0] != 3:
      raise ValueError(
          'First dimension of mlp_dims must be 3. mlp_dims={}'.format(mlp_dims))

    def _CombineLastTwoDims(x):
      shape = py_utils.GetShape(x)
      return tf.reshape(x, shape[:-2] + [np.prod(shape[-2:])])

    return self._Seq(
        name,
        self._Par(
            'transpose_matmul',
            lambda x, y: tf.matmul(x, y, transpose_a=True),
            # Features [..., points, num_in_channels].
            self._GetValue('get_features', 'features'),
            self._Seq(
                'transform_points',
                # Map points into features to use FeaturesMLP which is padding
                # batchnorm aware.
                self._SeqToKey(
                    'points_as_features', 'features',
                    self._GetValue('get_points', 'points')),
                self._FeaturesMLP('points_mlp', mlp_dims),
                # Output of this should be [..., points, mlp_dims[-1]].
                self._GetValue('get_transformed_points', 'features'))),
        # Post transform_matmul, should be [..., num_in_channels, mlp_dims[-1]].
        # Note: The paper's use of conv is equivalent to reshaping so that the
        # last two dims are combined together.
        self._ApplyFn('reshape', fn=_CombineLastTwoDims),
        # TODO(jngiam): Consider handling batch norm carefully here, not all
        # center points have valid values.
        self._FC('fc', num_in_channels * mlp_dims[-1], num_out_channels))
# pyformat: enable
