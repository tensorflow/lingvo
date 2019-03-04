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
"""Common encoder interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lingvo.core import base_layer


class BaseEncoder(base_layer.BaseLayer):
  """Base class for all encoders."""

  @classmethod
  def Params(cls):
    p = super(BaseEncoder, cls).Params()
    p.Define(
        'packed_input', False, 'If True, encoder and all layers support '
        'multiple examples in a single sequence.')
    return p

  def FProp(self, theta, inputs, paddings, segment_id):
    """Encodes source as represented by `inputs` and `paddings`.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: The inputs tensor. It is expected to be of shape [batch, time,
        ...].
      paddings: The paddings tensor. It is expected to be of shape [batch,
        time].
      segment_id: source segment id, of shape [batch, time]. This input is meant
        to support multiple samples in a single training sequence. The id
        identifiess the sample that the input at the corresponding time-step
        belongs to. For example, if the two examples packed together are
        ['good', 'day'] -> ['guten-tag'] and ['thanks'] -> ['danke'] to produce
        ['good', 'day', 'thanks'] -> ['guten-tag', 'danke'], the source segment
        ids would be [0, 0, 1] and target segment ids would be [0, 1]. These ids
        are meant to enable masking computations for different examples from
        each other. Models or layers that don't support packed inputs should
        pass None.

    Returns:
      A NestedMap, usually containing:
        * `encoded` - feature tensor of the shape [time, batch, depth]
        * `padding` - 0/1 tensor of the shape [time, batch, 1].
        * `segment_id` - (optional) present and have the shape [time, batch]
          if packed inputs are supported by the model (and all layers).
    """
    raise NotImplementedError('Abstract method')
