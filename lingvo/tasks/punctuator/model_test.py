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
from lingvo.tasks.punctuator import model
from lingvo.tasks.punctuator import input_generator

_TF_RANDOM_SEED = 93820986


class PunctuatorModelTest(tf.test.TestCase):
  """Tests for the Punctuator model.

  Overriding parameters and inheriting tests from RNMTModelTest.
  """

  def _InputParams(self):
    p = input_generator.PunctuatorInput.Params()
    input_file = test_helper.test_src_dir_path('tasks/lm/testdata/lm1b_100.txt')
    p.tokenizer.vocab_filepath = test_helper.test_src_dir_path(
        'tasks/punctuator/params/brown_corpus_wpm.16000.vocab')
    p.tokenizer.vocab_size = 16000
    p.file_pattern = 'text:' + input_file
    p.file_random_seed = 31415
    p.file_parallelism = 1
    p.bucket_upper_bound = [40]
    p.bucket_batch_limit = [4]
    p.source_max_length = 40
    p.target_max_length = 40
    return p

  def _testParams(self):
    p = model.RNMTModel.Params()
    p.name = 'test_mdl'
    p.input = self._InputParams()
    p.train.learning_rate = 2e-4
    return p

  def testConstruction(self):
    with self.session():
      p = self._testParams()
      mdl = p.cls(p)
      print('vars = ', mdl.vars)
      flatten_vars = mdl.vars.Flatten()
      print('vars flattened = ', flatten_vars)
      self.assertEqual(len(flatten_vars), 115)

      # Should match tf.trainable_variables().
      self.assertEqual(len(tf.trainable_variables()), len(flatten_vars))

  def testFProp(self, dtype=tf.float32):
    with self.session() as sess:
      tf.set_random_seed(_TF_RANDOM_SEED)
      p = self._testParams()
      p.dtype = dtype
      mdl = p.cls(p)
      mdl.FPropDefaultTheta()
      loss = mdl.loss
      logp = mdl.eval_metrics['log_pplx'][0]
      tf.global_variables_initializer().run()
      vals = []
      for _ in range(3):
        vals += [sess.run((loss, logp))]

      print('actual vals = %s' % np.array_repr(np.array(vals)))
      expected_vals = [[326.767578, 10.373574], [306.010498, 10.373238],
                       [280.089264, 10.373676]]
      self.assertAllClose(vals, expected_vals)

  def testBProp(self):
    with self.session() as sess:
      tf.set_random_seed(_TF_RANDOM_SEED)
      p = self._testParams()
      mdl = p.cls(p)
      mdl.FPropDefaultTheta()
      mdl.BProp()
      loss = mdl.loss
      logp = mdl.eval_metrics['log_pplx'][0]

      tf.global_variables_initializer().run()
      vals = []
      for _ in range(3):
        vals += [sess.run((loss, logp, mdl.train_op))[:2]]
      print('BProp actual vals = ', vals)
      expected_vals = [[326.767578, 10.373574], [305.863251, 10.368246],
                       [279.644745, 10.357213]]
      self.assertAllClose(vals, expected_vals)

  def testFPropEvalMode(self):
    with self.session() as sess:
      tf.set_random_seed(_TF_RANDOM_SEED)
      p = self._testParams()
      p.is_eval = True
      mdl = p.cls(p)
      mdl.FPropDefaultTheta()
      loss = mdl.loss
      logp = mdl.eval_metrics['log_pplx'][0]
      tf.global_variables_initializer().run()
      vals = []
      for _ in range(3):
        vals += [sess.run((loss, logp))]
      print('actual vals = ', vals)
      expected_vals = [[326.767578, 10.373574], [306.010498, 10.373238],
                       [280.089264, 10.373676]]
      self.assertAllClose(vals, expected_vals)

  def testInference(self):
    with self.session(use_gpu=False) as sess:
      tf.set_random_seed(93820985)
      p = self._testParams()
      p.is_eval = True
      mdl = p.cls(p)
      fetches, feeds = mdl.Inference()['default']

      tf.global_variables_initializer().run()
      src_strings = ['the cat sat on the mat', 'the dog sat on the mat']
      dec_out = sess.run(fetches, {feeds['src_strings']: src_strings})
      print('dec_out', dec_out)


if __name__ == '__main__':
  tf.test.main()
