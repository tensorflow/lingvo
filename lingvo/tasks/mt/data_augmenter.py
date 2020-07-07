# Lint as: python3
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
"""Classes which modifies inputs for pretraining.

Currently support: MASS
"""

from lingvo.core import base_layer
from lingvo.core import ops
from lingvo.core import py_utils
import numpy as np


class MASS(base_layer.BaseLayer):
  """Prepare data for MASS pretraining.

  This class is a wrapper for the MASS op.
  """

  @classmethod
  def Params(cls):
    # Params for the MASS op api.
    p = super().Params()
    p.Define('mask_id', 3, 'Id for the mask token.')
    p.Define('mask_ratio', 0.5, 'Mask fraction.')
    p.Define('mask_minlen', 0, 'Minimum number of tokens to mask.')
    p.Define(
        'span_len', 100000,
        'Split total mask_len into segments of this size and randomly'
        ' distribute them across the src sentence for segmented masking.')
    p.Define(
        'random_start_prob', 0.6,
        'Probability that placement of masked segments will be entirely '
        'random. Remaining cases are split evenly between masking at the '
        'beginning and at the end.')
    # keep_prob/rand_prob/mask_prob must sum to 1.
    p.Define(
        'keep_prob', 0.1,
        'Probability that a token designated for masking will be '
        'unchanged.')
    p.Define(
        'rand_prob', 0.1,
        'Probability that a token designated for masking will be replaced '
        'with a random token.')
    p.Define(
        'mask_prob', 0.8,
        'Probability that a token designated for masking will be replaced '
        'with mask_id.')
    p.Define(
        'mask_target', True,
        'If true, the target is masked with the inverse of the source '
        'mask.')
    p.Define('vocab_size', 0,
             'Used when selecting a random token to replaced a masked token.')
    p.Define(
        'first_unreserved_id', 4,
        'Tokens greater than or equal to this may be selected at random '
        'to replace a masked token.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if not np.isclose([p.keep_prob + p.rand_prob + p.mask_prob], [1]):
      raise ValueError('keep_prob, rand_prob, mask_prob must sum to 1')
    if p.vocab_size == 0:
      raise ValueError('vocab_size parameter must be set explicitly')

  def Mask(self, seq_ids, weights, actual_seq_len):
    p = self.params
    (src_ids, tgt_ids, tgt_labels, tgt_weights) = ops.mass(
        seq_ids,
        weights,
        actual_seq_len,
        mask_id=p.mask_id,
        mask_ratio=p.mask_ratio,
        mask_minlen=p.mask_minlen,
        span_len=p.span_len,
        random_start_prob=p.random_start_prob,
        keep_prob=p.keep_prob,
        rand_prob=p.rand_prob,
        mask_prob=p.mask_prob,
        mask_target=p.mask_target,
        vocab_size=p.vocab_size,
        first_unreserved_id=p.first_unreserved_id)

    mass_out = py_utils.NestedMap()
    mass_out.src = py_utils.NestedMap()
    mass_out.src.ids = src_ids
    mass_out.tgt = py_utils.NestedMap()
    mass_out.tgt.ids = tgt_ids
    mass_out.tgt.labels = tgt_labels
    mass_out.tgt.weights = tgt_weights
    return mass_out
