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
# ==============================================================================
"""Some utilities for configuring models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from lingvo.core import rnn_layers

FLAGS = tf.flags.FLAGS


def CreateUnidirectionalRNNParams(layer_params, cell_params):
  """Creates parameters for uni-directional RNN layer.

  Based on `layer_params.unidi_rnn_type`.

  Args:
    layer_params: parameters for the layers, has to contain unidi_rnn_type.
    cell_params: parameters for the RNN cell.
  Returns:
    Parameters for uni-directional RNN layer.
  Raises:
    ValueError: if `unidi_rnn_type` is invalid.
  """
  p = layer_params
  assert hasattr(p,
                 'unidi_rnn_type'), 'layer params must contain unidi_rnn_type'
  unidi_rnn_type = p.unidi_rnn_type
  if unidi_rnn_type == 'func':
    unidi_cls = rnn_layers.FRNN
  elif unidi_rnn_type == 'cudnn' or unidi_rnn_type == 'native_cudnn':
    unidi_cls = rnn_layers.CuDNNLSTM
  elif unidi_rnn_type == 'quasi_ifo' or unidi_rnn_type == 'sru':
    unidi_cls = rnn_layers.FRNN
  else:
    raise ValueError('Invalid unidi_rnn_type: %s', unidi_rnn_type)
  params = unidi_cls.Params()
  params.cell = cell_params
  return params


def CreateBidirectionalRNNParams(layer_params, forward_cell_params,
                                 backward_cell_params):
  """Creates parameters for bi-directional RNN layer.

  Based on `layer_params.bidi_rnn_type`.

  Args:
    layer_params: parameters for the layers, has to contain bidi_rnn_type.
    forward_cell_params: parameters for the forward RNN cell.
    backward_cell_params: parameters for the backward RNN cell.
  Returns:
    Parameters for bi-directional RNN layer.
  Raises:
    ValueError: if `bidi_rnn_type` is invalid.
  """
  p = layer_params
  assert hasattr(p, 'bidi_rnn_type'), 'layer params must contain bidi_rnn_type'
  bidi_rnn_type = p.bidi_rnn_type
  if bidi_rnn_type == 'func':
    bidi_cls = rnn_layers.BidirectionalFRNN
  elif bidi_rnn_type == 'native_cudnn':
    bidi_cls = rnn_layers.BidirectionalNativeCuDNNLSTM
  elif bidi_rnn_type == 'quasi_ifo' or bidi_rnn_type == 'sru':
    bidi_cls = rnn_layers.BidirectionalFRNNQuasi
  else:
    raise ValueError('Invalid bidi_rnn_type: %s', bidi_rnn_type)
  params = bidi_cls.Params()
  params.fwd = forward_cell_params
  params.bak = backward_cell_params
  return params
