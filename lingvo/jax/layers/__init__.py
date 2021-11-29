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
"""Exposes the public layer functionalities."""

from lingvo.jax.layers.activations import Activation

from lingvo.jax.layers.attentions import AttentionProjection
from lingvo.jax.layers.attentions import causal_mask
from lingvo.jax.layers.attentions import causal_segment_mask
from lingvo.jax.layers.attentions import convert_paddings_to_mask
from lingvo.jax.layers.attentions import DotProductAttention
from lingvo.jax.layers.attentions import PerDimScale
from lingvo.jax.layers.attentions import segment_mask

from lingvo.jax.layers.augmentations import MaskedLmDataAugmenter

from lingvo.jax.layers.convolutions import Conv2D
from lingvo.jax.layers.convolutions import ConvBNAct
from lingvo.jax.layers.convolutions import DepthwiseConv1D
from lingvo.jax.layers.convolutions import LightConv1D

from lingvo.jax.layers.embedding_softmax import PositionalEmbedding
from lingvo.jax.layers.embedding_softmax import SingleShardEmbedding
from lingvo.jax.layers.embedding_softmax import SingleShardFullSoftmax
from lingvo.jax.layers.embedding_softmax import SingleShardSharedEmbeddingSoftmax

from lingvo.jax.layers.flax_wrapper import FlaxModule

from lingvo.jax.layers.linears import Bias
from lingvo.jax.layers.linears import FeedForward
from lingvo.jax.layers.linears import Linear
from lingvo.jax.layers.linears import project_last_dim

from lingvo.jax.layers.ngrammer import get_bigram_ids
from lingvo.jax.layers.ngrammer import Ngrammer
from lingvo.jax.layers.ngrammer import VectorQuantization
from lingvo.jax.layers.ngrammer import VQNgrammer

from lingvo.jax.layers.normalizations import BatchNorm
from lingvo.jax.layers.normalizations import compute_moments
from lingvo.jax.layers.normalizations import LayerNorm

from lingvo.jax.layers.pipeline import LayerwiseShardablePipelined

from lingvo.jax.layers.poolings import GlobalPooling
from lingvo.jax.layers.poolings import Pooling

from lingvo.jax.layers.recurrent import AutodiffCheckpointType
from lingvo.jax.layers.recurrent import recurrent_func
from lingvo.jax.layers.recurrent import recurrent_static
from lingvo.jax.layers.recurrent import scan

from lingvo.jax.layers.repeats import Repeat

from lingvo.jax.layers.resnets import ResNet
from lingvo.jax.layers.resnets import ResNetBlock

from lingvo.jax.layers.stochastics import Dropout
from lingvo.jax.layers.stochastics import StochasticResidual

from lingvo.jax.layers.transformers import compute_attention_masks_for_extend_step
from lingvo.jax.layers.transformers import compute_attention_masks_for_fprop
from lingvo.jax.layers.transformers import StackedTransformer
from lingvo.jax.layers.transformers import StackedTransformerRepeated
from lingvo.jax.layers.transformers import Transformer
from lingvo.jax.layers.transformers import TransformerFeedForward
from lingvo.jax.layers.transformers import TransformerFeedForwardMoe
from lingvo.jax.layers.transformers import TransformerLm
