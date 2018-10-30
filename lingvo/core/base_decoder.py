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
"""Common decoder interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lingvo.core import base_layer
from lingvo.core import beam_search_helper
from lingvo.core import target_sequence_sampler


class BaseDecoder(base_layer.BaseLayer):
  """Base class for all decoders."""

  @classmethod
  def Params(cls):
    p = super(BaseDecoder, cls).Params()
    p.Define(
        'packed_input', False, 'If True, decoder and all layers support '
        'multiple examples in a single sequence.')
    return p

  def FProp(self, theta, src_encs, src_enc_paddings, targets, src_segment_id):
    """Decodes `targets` given encoded source.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      src_encs: source encoding, of shape [time, batch, depth].
      src_enc_paddings: source encoding padding, of shape [time, batch].
      targets: A dict of string to tensors representing the targets one try to
        predict.
      src_segment_id: source segment id, of shape [time, batch]. This input is
        meant to support multiple samples in a single training sequence. The id
        identifies the sample that the input at the corresponding time-step
        belongs to. For example, if the two examples packed together are
        ['good', 'day'] -> ['guten-tag'] and ['thanks'] -> ['danke'] to produce
        ['good', 'day', 'thanks'] -> ['guten-tag', 'danke'], the source segment
        ids would be [0, 0, 1] and target segment ids would be [0, 1]. These ids
        are meant to enable masking computations for different examples from
        each other. Models or layers than don't support packed inputs should
        pass None.

    Returns:
      A map from metric name (a python string) to a tuple (value, weight).
      Both value and weight are scalar Tensors.
    """
    predictions = self.ComputePredictions(theta, src_encs, src_enc_paddings,
                                          targets, src_segment_id)
    return self.ComputeLoss(theta, predictions, targets)

  def ComputePredictions(self, theta, src_encs, src_enc_paddings, targets,
                         src_segment_id):
    raise NotImplementedError('Abstract method: %s' % type(self))

  def ComputeLoss(self, theta, predictions, targets):
    raise NotImplementedError('Abstract method: %s' % type(self))


class BaseBeamSearchDecoder(BaseDecoder):
  """Decoder that does beam search."""

  @classmethod
  def Params(cls):
    p = super(BaseBeamSearchDecoder, cls).Params()
    p.Define('target_sos_id', 1, 'Id of the target sequence sos symbol.')
    p.Define('target_eos_id', 2, 'Id of the target sequence eos symbol.')
    # TODO(rpang): remove target_seq_len and use beam_search.target_seq_len
    # instead.
    p.Define('target_seq_len', 0, 'Target seq length.')
    p.Define('beam_search', beam_search_helper.BeamSearchHelper.Params(),
             'BeamSearchHelper params.')
    p.Define('target_sequence_sampler',
             target_sequence_sampler.TargetSequenceSampler.Params(),
             'TargetSequenceSampler params.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BaseBeamSearchDecoder, self).__init__(params)
    p = self.params
    p.beam_search.target_seq_len = p.target_seq_len
    p.beam_search.target_sos_id = p.target_sos_id
    p.beam_search.target_eos_id = p.target_eos_id
    self.CreateChild('beam_search', p.beam_search)
    p.target_sequence_sampler.target_seq_len = p.target_seq_len
    p.target_sequence_sampler.target_sos_id = p.target_sos_id
    p.target_sequence_sampler.target_eos_id = p.target_eos_id
    self.CreateChild('target_sequence_sampler', p.target_sequence_sampler)

  def BeamSearchDecode(self, src_encs, src_enc_paddings):
    # pylint: disable=line-too-long
    """Performs beam search based decoding.

    Args:
      src_encs: source encoding, of shape [time, batch, depth].
      src_enc_paddings: source encoding padding, of shape [time, batch].
    returns:
      `.BeamSearchDecodeOutput`, A namedtuple whose elements are tensors.
    """
    # pylint: enable=line-too-long
    raise NotImplementedError('Abstract method')
