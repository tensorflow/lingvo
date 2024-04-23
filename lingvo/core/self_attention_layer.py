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
"""Self-attention layers.

[1] Attention is all you need.
    https://arxiv.org/pdf/1706.03762.pdf Section 3.
[2] Pay Less Attention with Lightweight and Dynamic Convolutions.
    https://arxiv.org/abs/1901.10430
"""
from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import batch_major_attention
from lingvo.core import py_utils

MultiHeadedSelfAttention = batch_major_attention.MultiHeadedAttention


class Builder(batch_major_attention.Builder):
  """Builder for self-attention layers."""

  def SelfAttention(self, name):
    p = self.params
    input_to_add = (
        'i.vec' if p.selfatten_add_unnormalized_input else 'after_ln')

    attention_inputs = 'after_ln,after_ln,after_ln,i.paddings'
    if p.packed_input:
      attention_inputs += ',i.segment_mask'

    sub_list = [
        ('i.vec->after_ln', self._DefaultLN('LN')),
        (
            '{}->after_att,unused_prob'.format(attention_inputs),
            self._MultiHeadedAtten(
                'atten',
                enable_qkv_proj_in_onestep=p.default_enable_qkv_proj_in_onestep,
                enable_qk_proj_in_onestep=p.atten_tpl.enable_qk_proj_in_onestep,
            ),
        ),
        (
            'after_att->after_dropout',
            self._Dropout('dropout', p.residual_dropout_prob),
        ),
        (
            '{},after_dropout->o.vec'.format(input_to_add),
            self._Add('add', apply_residual=p.atten_apply_residual),
        ),
        ('i.paddings->o.paddings', self._Id('id')),
    ]

    if p.packed_input:
      sub_list.append(('i.segment_mask->o.segment_mask', self._Id('mask')))

    return self._Graph(
        name,
        ['i'],  # input NestedMap with {vec, paddings, segment_mask}
        ['o'],  # output NestedMap with {vec, paddings, segment_mask}
        *sub_list
    )

  def _TransformerLayerBlock(self, name, feed_forward_qdomain=None):
    """(inputs, paddings) -> (encoded, paddings)."""
    return self._Seq(
        name,
        self.SelfAttention('self_atten'),
        self.Feedforward('ff', qdomain=feed_forward_qdomain),
    )

  def TransformerStack(self, name, num_layers=1, feed_forward_qdomain=None):
    """Returns a stack of num_layers self-attention layers."""
    blocks = [
        self._TransformerLayerBlock(
            'block_{}'.format(d), feed_forward_qdomain=feed_forward_qdomain
        )
        for d in range(num_layers)
    ]
    return self._MaybeSplit(name, blocks) or (
        self._Rep(name, num_layers, self._TransformerLayerBlock('block'))
    )

  def _StridedTransformerLayerBlock(
      self, name, *, stride=1, first_n=None, feed_forward_qdomain=None
  ):
    """(inputs, paddings) -> (encoded, paddings)."""
    return self._Seq(
        name,
        self._StridedAttention('self_atten', stride=stride, first_n=first_n),
        self.Feedforward('ff', qdomain=feed_forward_qdomain),
    )

  def TransformerStackV2(
      self,
      name,
      num_layers=1,
      *,
      final_layer_first_n=None,
      final_layer_stride=1,
      feed_forward_qdomain=None
  ):
    """Returns a stack of num_layers self-attention layers."""
    blocks = []
    for i in range(num_layers):
      if i < num_layers - 1:
        stride, first_n = (1, None)
      else:
        stride, first_n = (final_layer_stride, final_layer_first_n)
      blocks.append(
          self._Seq(
              'iter_%03d' % i,
              self._StridedTransformerLayerBlock(
                  'block',
                  stride=stride,
                  first_n=first_n,
                  feed_forward_qdomain=feed_forward_qdomain,
              ),
          )
      )
    return self._MaybeSplit(name, blocks) or self._Seq(name, *blocks)


class SimplifiedTransformerBuilder(Builder):
  """Builder for simplified transformer blocks based on https://arxiv.org/pdf/2311.01906.pdf."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'parallel_attention_mlp',
        False,
        'If Set, Attention and MLP are computed in parallel as in Fig. 10 of'
        'https://arxiv.org/pdf/2311.01906.pdf.',
    )
    return p

  def TransformerLayerBlock(
      self,
      name,
      stride=1,
      first_n=None,
      num_heads=None,
      feed_forward_qdomain=None,
  ):
    """(inputs, paddings) -> (encoded, paddings)."""
    p = self.params
    ff_hidden_dim = p.ff_hidden_dim
    if p.selfatten_add_unnormalized_input:
      tf.logging.warning(
          'This flag is a no-op since there is no residual add for self attn.'
      )

    attention_inputs = 'strided_query,after_ln,after_ln,i.paddings'
    sub_list = []
    if p.packed_input:
      if stride > 1:
        # TODO(huangyp): Make sure striding won't cross segment boundaries.
        tf.logging.warning(
            'Each segment in the packed input should has length '
            'divisible by stride.'
        )
      sub_list += [
          (
              'i.segment_mask->strided_segment_mask',
              self._Stride(
                  'segment_mask_query_stride', stride, first_n, axis=2
              ),
          ),
      ]
      attention_inputs += ',strided_segment_mask'

    if num_heads is None:
      num_heads = p.num_heads

    # compute qkv in one step only if default_enable_qkv_proj_in_onestep
    # and no striding (stride==1) and not first_n
    enable_qkv_proj_in_onestep = (
        p.default_enable_qkv_proj_in_onestep and stride == 1 and not first_n
    )
    # Overriding default param based on stride.
    enable_qk_proj_in_onestep = (
        p.atten_tpl.enable_qk_proj_in_onestep
        and stride == 1
        and not first_n
        and not p.atten_tpl.use_mqa
    )

    sub_list += [
        ('i.vec->strided_input', self._Stride('before_add', stride, first_n)),
        # TODO(b/323011070): Add rezero support.
        ('i.vec->after_ln', self._DefaultLN('LN')),
        (
            'after_ln->strided_query',
            self._Stride('query_after_stride', stride, first_n),
        ),
    ]

    sub_list += [
        (
            '{}->after_att,prob'.format(attention_inputs),
            self._MultiHeadedAtten(
                'atten',
                num_heads,
                enable_qkv_proj_in_onestep=enable_qkv_proj_in_onestep,
                enable_qk_proj_in_onestep=enable_qk_proj_in_onestep,
                query_stride=stride,
                query_first_n=first_n,
            ),
        ),
        (
            'after_att->attn_output.vec',
            self._Dropout('dropout', p.residual_dropout_prob),
        ),
        (
            'i.paddings->attn_output.paddings',
            self._Stride('padding_after_Stride', stride, first_n),
        ),
    ]
    if p.packed_input:
      sub_list += [
          (
              'strided_segment_mask->attn_output.segment_mask',
              self._Stride(
                  'segment_mask_context_stride', stride, first_n, axis=3
              ),
          ),
      ]

    if p.parallel_attention_mlp:
      if p.device_mesh is not None:
        assert p.device_mesh.ndim >= 2
        assert p.weight_split_dims_mapping.df is not None
        assert p.weight_split_dims_mapping.fd is not None
      bias_f_split = (
          [p.weight_split_dims_mapping.df[1]]
          if p.weight_split_dims_mapping.df is not None
          else None
      )
      bias_d_split = (
          [p.weight_split_dims_mapping.fd[1]]
          if p.weight_split_dims_mapping.fd is not None
          else None
      )
      sub_list += [
          (
              'strided_query->after_feedforward',
              self._Seq(
                  'feedforward',
                  self._Linear(
                      'linear01',
                      p.model_dim,
                      ff_hidden_dim,
                      device_mesh=p.device_mesh,
                      weight_split_dims_mapping=(
                          p.weight_split_dims_mapping.df
                      ),
                      qdomain=feed_forward_qdomain,
                  ),
                  self.MeshSplit(
                      'split01', p.activation_split_dims_mapping.blf
                  ),
                  self._Bias(
                      'bias01',
                      ff_hidden_dim,
                      device_mesh=p.device_mesh,
                      weight_split_dims_mapping=bias_f_split,
                  ),
                  self._Activation('act', p.ff_activation_fn),
                  self._Dropout('relu_dropout', p.relu_dropout_prob),
                  self._Linear(
                      'linear02',
                      ff_hidden_dim,
                      p.model_dim,
                      device_mesh=p.device_mesh,
                      weight_split_dims_mapping=(
                          p.weight_split_dims_mapping.fd
                      ),
                      qdomain=feed_forward_qdomain,
                  ),
                  self.MeshSplit(
                      'split02', p.activation_split_dims_mapping.bld
                  ),
                  self._Bias(
                      'bias02',
                      p.model_dim,
                      device_mesh=p.device_mesh,
                      weight_split_dims_mapping=bias_d_split,
                  ),
                  self._Dropout('dropout', p.residual_dropout_prob),
              ),
          ),
      ]
      sub_list += [
          (
              'attn_output.vec,after_feedforward->added_attn_ff',
              self._Add('attn_add_ff', p.ff_residual_weight),
          ),
          ('added_attn_ff,attn_output.paddings->o.vec', self._Pad('pad')),
          ('attn_output.paddings->o.paddings', self._Id('paddings_output')),
      ]
      if p.packed_input:
        sub_list += [
            (
                'attn_output.segment_mask->o.segment_mask',
                self._Id('segment_mask_output'),
            ),
        ]
    else:
      sub_list += [(
          'attn_output->o',
          self.Feedforward(
              'ff',
              ff_hidden_dim=ff_hidden_dim,
              qdomain=feed_forward_qdomain,
          ),
      )]

    return self._Graph(
        name,
        ['i'],  # input NestedMap with {vec, paddings, segment_mask}
        ['o'],  # output NestedMap with {vec, paddings, segment_mask}
        *sub_list
    )

  def TransformerStack(self, name, num_layers=1, feed_forward_qdomain=None):
    """Returns a stack of num_layers self-attention layers."""
    blocks = [
        self.TransformerLayerBlock(
            'block_{}'.format(d), feed_forward_qdomain=feed_forward_qdomain
        )
        for d in range(num_layers)
    ]
    return self._MaybeSplit(name, blocks) or (
        self._Rep(name, num_layers, self.TransformerLayerBlock('block'))
    )

  def TransformerStackV2(
      self,
      name,
      num_layers=1,
      *,
      final_layer_first_n=None,
      final_layer_stride=1,
      feed_forward_qdomain=None
  ):
    """Returns a stack of num_layers self-attention layers."""
    blocks = []
    for i in range(num_layers):
      if i < num_layers - 1:
        stride, first_n = (1, None)
      else:
        stride, first_n = (final_layer_stride, final_layer_first_n)
      blocks.append(
          self._Seq(
              'iter_%03d' % i,
              self.TransformerLayerBlock(
                  'block',
                  stride=stride,
                  first_n=first_n,
                  feed_forward_qdomain=feed_forward_qdomain,
              ),
          )
      )
    return self._MaybeSplit(name, blocks) or self._Seq(name, *blocks)


# TODO(huangyp): remove this layer after transition to nested maps is complete.
class StackedTransformerEncoderLayers(base_layer.BaseLayer):
  """Wrapper class for layers returned by Builder.TransformerStack."""

  @classmethod
  def Cast(cls, params):
    # Cast params returned from the builder to params in this class.
    params.Define('base_cls', params.cls, 'Store the base class in params.')
    params.cls = cls
    return params

  def __init__(self, params):
    # Make this class a sub-class of params.base_cls
    self.__class__ = type(self.__class__.__name__, (params.base_cls, object),
                          dict(self.__class__.__dict__))
    super(self.__class__, self).__init__(params)  # pylint: disable=bad-super-call

  def FProp(self, theta, vec, paddings, segment_mask=None):
    outputs = super(self.__class__, self).FProp(  # pylint: disable=bad-super-call
        theta,
        py_utils.NestedMap(
            vec=vec, paddings=paddings, segment_mask=segment_mask))
    return outputs.vec, outputs.paddings
