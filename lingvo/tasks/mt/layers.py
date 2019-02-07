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
"""Lingvo MT layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
import tensorflow as tf

from lingvo.core import base_layer
from lingvo.core import layers
from lingvo.core import layers_with_attention


class TransformerStack(base_layer.BaseLayer):
  """Stacked self- multi-head attention and fully connected layers.

  With optional layer normalization applied to the final output.

  See 'Attention Is All You Need' https://arxiv.org/abs/1706.03762
  for details.
  """

  @classmethod
  def Params(cls):
    """Configs for TransformerStack."""
    p = super(TransformerStack, cls).Params()

    # Transformer related
    p.Define('model_dim', 1024, 'Characteristic depth (dimension).')
    p.Define('num_transformer_layers', 6, 'Number of transformer layers.')
    p.Define('transformer_tpl', layers_with_attention.TransformerLayer.Params(),
             'TransformerLayer params tpl.')

    p.Define('ln_tpl', layers.LayerNorm.Params(), 'Layer norm default params')
    p.Define('ln_output', False,
             'If set, layer normalization is applied to the final output'
             ' of the encoder transformer stack.')

    p.Define('is_transparent', False,
             'If set, outputs a merger of embeddings and layer outputs.')
    p.Define('num_transparent_outputs', 6, 'Number of transparent outputs.')
    p.Define(
        'transparent_merger_tpl',
        layers.WeightedSumLayer.Params().Set(add_weight_summaries=True),
        'Merger op for layer outputs.')
    p.Define('packed_input', False,
             'If True, assumes multiple training samples per input.')
    p.Define('has_aux_attention', False,
             'Allows encoder layers to attend auxiliary inputs.')
    p.transformer_tpl.tr_atten_tpl.num_attention_heads = 8
    p.transformer_tpl.tr_fflayer_tpl.hidden_dim = 8192
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(TransformerStack, self).__init__(params)
    p = self.params

    with tf.variable_scope(p.name):
      # Add transformer layers.
      transformer_layer_params = []
      for i in range(p.num_transformer_layers):
        params = p.transformer_tpl.Copy()
        params.name = 'trans_%d' % (i)
        params.source_dim = p.model_dim
        params.packed_input = p.packed_input
        params.has_aux_atten = p.has_aux_attention
        transformer_layer_params.append(params)

      self.CreateChildren('trans', transformer_layer_params)

      # Initialize TransformerStack output layer norm
      if p.ln_output:
        params = p.ln_tpl.Copy()
        # Keeping historic 'enc_out_ln' name for checkpoint compatibility.
        params.name = 'enc_out_ln'
        params.input_dim = p.model_dim
        self.CreateChild('layer_norm_out', params)

      if p.is_transparent:
        transparent_params = []
        if not p.num_transparent_outputs:
          raise ValueError('num_transparent_outputs should be greater than 0.')
        for i in range(p.num_transparent_outputs):
          transparent_param = p.transparent_merger_tpl.Copy()
          transparent_param.name = 'transparent_%d' % i
          transparent_param.num_sources = 1 + p.num_transformer_layers
          transparent_params.append(transparent_param)
        self.CreateChildren('transparent_merger', transparent_params)

  def FProp(self,
            theta,
            transformer_input,
            paddings,
            src_segment_id=None,
            aux_vecs=None,
            aux_paddings=None,
            aux_segment_id=None):
    """Transforms source sequence of Tensors with Transformers layers.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      transformer_input: A sequence of input Tensors of [time, batch, dim]
        shape.
      paddings: A sequence of 0s and 1s indicating input paddings of
         [time, batch] shape.
      src_segment_id: A sequence of ints indicating segment ids of
         [time, batch] shape.
      aux_vecs: A sequence of input Tensors of [aux_time, batch, dim] shape, as
          context for the cross-attention layer.
      aux_paddings: A sequence of 0s and 1s indicating input paddings of
         [aux_time, batch] shape.
      aux_segment_id: A sequence of ints indicating segment ids of
         [aux_time, batch] shape.

    Returns:
      (outputs, out_paddings, segment_ids) tuple. `outputs` is of the shape
      [time, batch, depth], and `out_paddings` has shape [time, batch]. If
      is_transparent is True, can return a list of num_transformer_layers
      tensors of shape [time, batch, depth] if `p.is_eval` is False, and a
      [time, batch, depth, num_transparent_outputs] tensor if `p.is_eval` is
      True. If packed_input is True, also returns segment_id, otherwise returns
      None.
    """
    p = self.params
    if p.packed_input:
      assert src_segment_id is not None, ('Need to specify src_segment_id if '
                                          'packed input is supported.')
    outputs_list = [transformer_input]
    with tf.name_scope(p.name):
      for i, transformer_l in enumerate(self.trans):

        # For encoder, keys, values and queries are the same
        transformer_output, _ = transformer_l.FProp(
            theta.trans[i],
            transformer_input,
            paddings,
            aux_vecs=aux_vecs,
            aux_paddings=aux_paddings,
            source_segment_id=src_segment_id,
            aux_segment_id=aux_segment_id)
        transformer_input = transformer_output
        outputs_list.append(transformer_output)

      if p.ln_output:
        transformer_output = self.layer_norm_out.FProp(theta.layer_norm_out,
                                                       transformer_output)

      # When is_transparent is set, it outputs a list of tensors during
      # training and the stacked tensors otherwise. This dual behavior is meant
      # to avoid excessive memory usage during training (which was prohibiting
      # training on TPUs), and simplify the beam search interface.
      if p.is_transparent:
        if p.num_transparent_outputs == 1:
          transformer_output = self.transparent_merger[0].FProp(
              theta.transparent_merger[0], outputs_list)
        else:
          transformer_output = []
          for i in range(p.num_transparent_outputs):
            merged_outputs = self.transparent_merger[i].FProp(
                theta.transparent_merger[i], outputs_list)
            transformer_output.append(merged_outputs)
          if p.is_eval:
            transformer_output = tf.stack(transformer_output, 3)

      return transformer_output, paddings, src_segment_id
