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
"""Tests for lm.model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import range

import tensorflow as tf
from lingvo.core import test_helper
from lingvo.core import test_utils
from lingvo.tasks.lm import input_generator
from lingvo.tasks.lm import model


class ModelTest(tf.test.TestCase):

  def _InputParams(self, for_training):
    p = input_generator.LmInput.Params()
    p.file_pattern = 'text:' + test_helper.test_src_dir_path(
        'tasks/lm/testdata/lm1b_100.txt')
    if for_training:
      p.file_buffer_size = 10000
      p.bucket_upper_bound = [50]
      p.bucket_batch_limit = [64]
      p.target_max_length = 1000
    else:
      p.file_random_seed = 31415
      p.file_parallelism = 1
      p.file_buffer_size = 1
      p.bucket_upper_bound = [20]
      p.bucket_batch_limit = [2]
      p.target_max_length = 20
    p.tokenizer.vocab_size = 76
    return p

  def _Params(self):
    p = model.LanguageModel.Params()
    p.name = 'lm_test'
    vocab, dims = 76, 64
    p.lm.vocab_size = vocab
    p.lm.emb.vocab_size = vocab
    p.lm.emb.embedding_dim = dims
    p.lm.rnns.num_layers = 4
    p.lm.rnns.cell_tpl.num_output_nodes = dims
    p.lm.rnns.cell_tpl.num_input_nodes = dims
    p.lm.softmax.input_dim = dims
    p.lm.softmax.num_classes = vocab
    return p

  def testLmFprop(self):
    tf.set_random_seed(93820986)
    p = self._Params()
    p.input = self._InputParams(for_training=False)

    with self.test_session(use_gpu=False) as sess:
      mdl = p.cls(p)
      mdl.FPropDefaultTheta()
      loss = mdl.eval_metrics['loss'][0]
      logp = mdl.eval_metrics['log_pplx'][0]
      logp_per_word = mdl.eval_metrics['log_pplx_per_word'][0]
      accuracy = mdl.eval_metrics['fraction_of_correct_next_step_preds'][0]
      tf.global_variables_initializer().run()

      loss, logp, logp_per_word, accuracy = sess.run(
          [loss, logp, logp_per_word, accuracy])
      test_utils.CompareToGoldenSingleFloat(self, 4.329850, loss)
      test_utils.CompareToGoldenSingleFloat(self, 4.329850, logp)
      test_utils.CompareToGoldenSingleFloat(self, 6.185500, logp_per_word)
      test_utils.CompareToGoldenSingleFloat(self, 0.025000, accuracy)

  def testLmTrain(self):
    p = self._Params()
    p.input = self._InputParams(for_training=True)
    tp = p.train
    tp.learning_rate = 3e-3

    with self.test_session() as sess:
      mdl = p.cls(p)
      mdl.FPropDefaultTheta()
      mdl.BProp()
      loss = mdl.eval_metrics['loss'][0]
      tf.global_variables_initializer().run()

      # Run some steps and we expect the loss goes down.
      loss_val, _ = sess.run([loss, mdl.train_op])
      self.assertGreater(loss_val, 4.0)
      for i in range(10):
        loss_val, _ = sess.run([loss, mdl.train_op])
        tf.logging.info('%d loss = %f', i, loss_val)
      self.assertLess(loss_val, 3.8)


if __name__ == '__main__':
  tf.test.main()
