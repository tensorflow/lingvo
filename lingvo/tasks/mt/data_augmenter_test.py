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
"""Tests for data_augmenter."""

from lingvo import compat as tf
from lingvo.core import test_utils
from lingvo.tasks.mt import data_augmenter
import numpy as np

FLAGS = tf.flags.FLAGS


class MassLayerTest(test_utils.TestCase):

  def _MassParams(self):
    p = data_augmenter.MASS.Params()
    p.mask_id = 3
    p.mask_ratio = 0.5
    p.mask_minlen = 0
    p.span_len = 3
    p.random_start_prob = 0.8
    p.keep_prob = 0
    p.rand_prob = 0
    p.mask_prob = 1
    p.mask_target = True
    p.vocab_size = 64
    p.first_unreserved_id = 4
    p.name = "mass_layer"
    return p

  def testMassLayer(self):
    with self.session(use_gpu=False) as sess:
      batch_size = 3
      seq_len = 10
      p = self._MassParams()
      mass_layer = data_augmenter.MASS(p)
      seq_ids = tf.fill([batch_size, seq_len], 4)
      weights = tf.ones([batch_size, seq_len])
      actual_seq_len = tf.fill([batch_size], 10)
      mass_out = mass_layer.Mask(seq_ids, weights, actual_seq_len)
      (src_ids, tgt_ids, tgt_labels, tgt_weights) = sess.run([
          mass_out.src.ids, mass_out.tgt.ids, mass_out.tgt.labels,
          mass_out.tgt.weights
      ])
      self.assertAllEqual(np.sum(src_ids == 3, axis=1), [5, 5, 5])
      self.assertAllEqual(np.sum(tgt_ids == 3, axis=1), [5, 5, 5])
      self.assertAllEqual(tgt_labels,
                          4 * np.ones([batch_size, seq_len], dtype=np.int32))
      self.assertAllEqual(np.sum(tgt_weights, axis=1), [5., 5., 5.])


if __name__ == "__main__":
  tf.test.main()
