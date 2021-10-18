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

from lingvo.jax.layers.activations import ActivationLayer

from lingvo.jax.layers.attentions import CausalMask
from lingvo.jax.layers.attentions import CausalSegmentMask
from lingvo.jax.layers.attentions import ConvertPaddingsToMask
from lingvo.jax.layers.attentions import MultiHeadedAttention
from lingvo.jax.layers.attentions import MultiHeadedProjectionLayer
from lingvo.jax.layers.attentions import PerDimScaleLayer
from lingvo.jax.layers.attentions import SegmentMask

from lingvo.jax.layers.augmentations import MaskedLmDataAugmenter

from lingvo.jax.layers.convolutions import Conv2D
from lingvo.jax.layers.convolutions import ConvBNAct

from lingvo.jax.layers.embedding_softmax import PositionalEmbeddingLayer
from lingvo.jax.layers.embedding_softmax import SingleShardEmbeddingLayer
from lingvo.jax.layers.embedding_softmax import SingleShardFullSoftmax
from lingvo.jax.layers.embedding_softmax import SingleShardSharedEmbeddingSoftmax

from lingvo.jax.layers.flax_wrapper import FlaxModuleLayer

from lingvo.jax.layers.linears import BiasLayer
from lingvo.jax.layers.linears import FeedForwardLayer
from lingvo.jax.layers.linears import LinearLayer
from lingvo.jax.layers.linears import ProjectLastDim

from lingvo.jax.layers.normalizations import BatchNormLayer
from lingvo.jax.layers.normalizations import ComputeMoments
from lingvo.jax.layers.normalizations import LayerNorm

from lingvo.jax.layers.poolings import GlobalPoolingLayer
from lingvo.jax.layers.poolings import PoolingLayer

from lingvo.jax.layers.recurrent import AutodiffCheckpointType
from lingvo.jax.layers.recurrent import recurrent_func
from lingvo.jax.layers.recurrent import recurrent_static
from lingvo.jax.layers.recurrent import scan

from lingvo.jax.layers.repeats import RepeatLayer

from lingvo.jax.layers.resnets import ResNet
from lingvo.jax.layers.resnets import ResNetBlock

from lingvo.jax.layers.stochastics import DropoutLayer
from lingvo.jax.layers.stochastics import StochasticResidualLayer

from lingvo.jax.layers.transformers import ComputeAttentionMasksForExtendStep
from lingvo.jax.layers.transformers import ComputeAttentionMasksForFProp
from lingvo.jax.layers.transformers import StackedTransformerLayers
from lingvo.jax.layers.transformers import StackedTransformerLayersRepeated
from lingvo.jax.layers.transformers import TransformerFeedForwardLayer
from lingvo.jax.layers.transformers import TransformerLayer
from lingvo.jax.layers.transformers import TransformerLm
from lingvo.jax.layers.transformers import TransformerShardedMoeLayer
