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
"""Common utilities for ASR decoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def SetRnnCellNodes(decoder_params, rnn_cell_params):
  rnn_cell_params.num_output_nodes = decoder_params.rnn_cell_dim
  if decoder_params.rnn_cell_hidden_dim > 0:
    if not hasattr(rnn_cell_params, 'num_hidden_nodes'):
      raise ValueError(
          'num_hidden_nodes not supported by the RNNCell: %s' % rnn_cell_params)
    rnn_cell_params.num_hidden_nodes = decoder_params.rnn_cell_hidden_dim
