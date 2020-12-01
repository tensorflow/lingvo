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
"""Self-attention layers.

[1] Attention is all you need.
    https://arxiv.org/pdf/1706.03762.pdf Section 3.
[2] Pay Less Attention with Lightweight and Dynamic Convolutions.
    https://arxiv.org/abs/1901.10430
"""

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
        ('{}->after_att,unused_prob'.format(attention_inputs),
         self._MultiHeadedAtten('atten')),
        ('after_att->after_dropout',
         self._Dropout('dropout', p.residual_dropout_prob)),
        ('{},after_dropout->o.vec'.format(input_to_add),
         self._Add('add', apply_residual=p.atten_apply_residual)),
        ('i.paddings->o.paddings', self._Id('id')),
    ]

    if p.packed_input:
      sub_list.append(
          ('i.segment_mask->o.segment_mask', self._Id('mask')))

    return self._Graph(
        name,
        ['i'],  # input NestedMap with {vec, paddings, segment_mask}
        ['o'],  # output NestedMap with {vec, paddings, segment_mask}
        *sub_list)

  def _TransformerLayerBlock(self, name):
    """(inputs, paddings) -> (encoded, paddings)."""
    return self._Seq(
        name,
        self.SelfAttention('self_atten'),
        self.Feedforward('ff'))

  def TransformerStack(self, name, num_layers=1):
    """Returns a stack of num_layers self-attention layers."""
    blocks = [self._TransformerLayerBlock(
        'block_{}'.format(d)) for d in range(num_layers)]
    return self._MaybeSplit(name, blocks) or (
        self._Rep(name, num_layers, self._TransformerLayerBlock('block')))

  def _StridedTransformerLayerBlock(self, name, *, stride=1, first_n=None):
    """(inputs, paddings) -> (encoded, paddings)."""
    return self._Seq(
        name,
        self._StridedAttention('self_atten', stride=stride, first_n=first_n),
        self.Feedforward('ff'))

  def TransformerStackV2(self,
                         name,
                         num_layers=1,
                         *,
                         final_layer_first_n=None,
                         final_layer_stride=1):
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
                  'block', stride=stride, first_n=first_n)))
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
