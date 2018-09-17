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
"""Tests for Punctuator model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from lingvo.core import test_helper
from lingvo.tasks.mt import model_test
from lingvo.tasks.punctuator import input_generator
from lingvo.tasks.punctuator import model


class PunctuatorModelTest(model_test.TransformerModelTest):
  """Tests for the Punctuator model.

  Overriding parameters and inheriting
  tests from TransformerModelTest.
  """

  def _InputParams(self):
    p = input_generator.PunctuatorInput.Params()
    input_file = test_helper.test_src_dir_path('tasks/lm/testdata/lm1b_100.txt')
    p.tokenizer.token_vocab_filepath = test_helper.test_src_dir_path(
        'tasks/punctuator/testdata/test_vocab.txt')
    p.file_pattern = 'text:' + input_file
    p.file_random_seed = 31415
    p.file_parallelism = 1
    p.bucket_upper_bound = [200, 400]
    p.bucket_batch_limit = [4, 8]
    p.source_max_length = 300
    p.target_max_length = 300
    return p

  def _testParams(self):
    p = model.PunctuatorBase.Params()
    p.name = 'test_mdl'
    p.input = self._InputParams()
    p.encoder = self._EncoderParams()
    p.decoder = self._DecoderParams()
    return p

  def testFProp(self, dtype=tf.float32):
    with self.session() as sess:
      tf.set_random_seed(model_test._TF_RANDOM_SEED)
      p = self._testParams()
      p.dtype = dtype
      mdl = p.cls(p)
      mdl.FProp(mdl.theta)
      loss = mdl.loss
      logp = mdl.eval_metrics['log_pplx'][0]
      tf.global_variables_initializer().run()
      vals = []
      for _ in range(5):
        vals += [sess.run((loss, logp))]

      print('actual vals = %s' % np.array_repr(np.array(vals)))
      expected_vals = [(796.953002, 10.383752), (1082.566772, 10.384333),
                       (1077.408813, 10.384663), (965.820556, 10.385167),
                       (1155.147460, 10.383347)]
      self.assertAllClose(vals, expected_vals)

  def testBProp(self):
    with self.session() as sess:
      tf.set_random_seed(model_test._TF_RANDOM_SEED)
      p = self._testParams()
      mdl = p.cls(p)
      mdl.FPropDefaultTheta()
      mdl.BProp()
      loss = mdl.loss
      logp = mdl.eval_metrics['log_pplx'][0]

      tf.global_variables_initializer().run()
      vals = []
      for _ in range(5):
        vals += [sess.run((loss, logp, mdl.train_op))[:2]]
      print('BProp actual vals = ', vals)
      expected_vals = [(796.953002, 10.383752), (1082.566772, 10.384333),
                       (1077.408813, 10.384663), (965.820556, 10.385167),
                       (1155.147460, 10.383347)]
      self.assertAllClose(vals, expected_vals)

  def testFPropEvalMode(self):
    with self.session() as sess:
      tf.set_random_seed(model_test._TF_RANDOM_SEED)
      p = self._testParams()
      p.is_eval = True
      mdl = p.cls(p)
      mdl.FPropDefaultTheta()
      loss = mdl.loss
      logp = mdl.eval_metrics['log_pplx'][0]
      tf.global_variables_initializer().run()
      vals = []
      for _ in range(5):
        vals += [sess.run((loss, logp))]
      print('actual vals = ', vals)
      expected_vals = [(796.953002, 10.383752), (1082.566772, 10.384333),
                       (1077.408813, 10.384663), (965.820556, 10.385167),
                       (1155.147460, 10.383347)]
      self.assertAllClose(vals, expected_vals)


if __name__ == '__main__':
  tf.test.main()
