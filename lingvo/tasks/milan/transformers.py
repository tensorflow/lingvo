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
"""Layers for preprocessing and modeling text inputs."""

from lingvo import compat as tf
from lingvo.core import builder_layers
from lingvo.core import layers
from lingvo.core import py_utils
from lingvo.tasks.milan import utils
from lingvo.tasks.mt import layers as mt_layers


def GetTransformerStackWithEmbeddingInput(*,
                                          input_dim: int,
                                          num_layers: int,
                                          hidden_dim: int,
                                          num_attention_heads: int,
                                          output_dim: int,
                                          name: str = ''):
  """Configures a transformer stack that takes embedding sequences as input.

  The returned layer has the overall signature
    (features, lengths) -> fixed_dim_encodings
  where
    `features` are [batch_size, max_length, input_dim],
    `lengths` are [batch_size], and
    `fixed_dim_encodings` have shape [batch_size, output_dim].

  Args:
    input_dim: int, dimension of the input embeddings.
    num_layers: int, number of transformer layers to use in the encoder.
    hidden_dim: int, size of transformer feed-forward hidden layer (see
      `TransformerFeedForwardLayer`)
    num_attention_heads: int, number of attention heads to use in each
      transformer layer,
    output_dim: int, the dimension of the output (and input) of each transformer
      layer; also the dimension of the final output.
    name: Name to use for the resulting layer.

  Returns:
    Params defining the encoder.
  """

  # Projects features to transformer 'model_dim' (a.k.a. 'output_dim').
  input_projection = layers.ProjectionLayer.Params().Set(
      name='input_projection',
      has_bias=True,
      batch_norm=False,
      input_dim=input_dim,
      output_dim=output_dim)

  @utils.MakeFnLayer
  def ConvertInputs(features, lengths):
    """Constructs the padding mask and transposes inputs to time-major."""
    features.shape.assert_has_rank(3)
    lengths.shape.assert_has_rank(1)
    paddings = 1 - tf.sequence_mask(
        lengths, maxlen=py_utils.GetShape(features)[1], dtype=features.dtype)
    return (utils.BatchMajorToTimeMajor(features),
            utils.BatchMajorToTimeMajor(paddings))

  transformer_stack = mt_layers.TransformerStack.Params().Set(
      name='transformer_stack',
      model_dim=output_dim,
      num_transformer_layers=num_layers)
  layer_template = transformer_stack.transformer_tpl
  layer_template.tr_fflayer_tpl.hidden_dim = hidden_dim
  layer_template.tr_atten_tpl.num_attention_heads = num_attention_heads

  return builder_layers.GraphLayer.Params().Set(
      name=name,
      input_endpoints=['features', 'lengths'],
      output_endpoints=['fixed_dim_encodings'],
      sub=[
          ('features->proj_features', input_projection),
          # Transpose inputs to time-major and construct the paddings.
          ('proj_features,lengths->time_major_features,time_major_paddings',
           ConvertInputs),
          ('time_major_features,time_major_paddings->time_major_encodings,_,_1',
           transformer_stack),
          ('time_major_encodings->fixed_dim_encodings',
           utils.MakeFnLayer(lambda t: t[0, ...], name='select_index_0')),
      ])
