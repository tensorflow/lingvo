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
from lingvo.core import tokenizers
from lingvo.tasks.lm import input_generator
from lingvo.tasks.lm import model


class ModelTest(test_utils.TestCase):

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
    p.tokenizer.vocab_size = 64
    return p

  def _Params(self):
    p = model.LanguageModel.Params()
    p.name = 'lm_test'
    vocab, dims = 64, 64
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

    with self.session(use_gpu=False) as sess:
      mdl = p.cls(p)
      mdl.FPropDefaultTheta()
      loss = mdl.eval_metrics['loss'][0]
      logp = mdl.eval_metrics['log_pplx'][0]
      logp_per_word = mdl.eval_metrics['log_pplx_per_word'][0]
      accuracy = mdl.eval_metrics['fraction_of_correct_next_step_preds'][0]
      tf.global_variables_initializer().run()

      loss, logp, logp_per_word, accuracy = sess.run(
          [loss, logp, logp_per_word, accuracy])
      test_utils.CompareToGoldenSingleFloat(self, 4.160992, loss)
      test_utils.CompareToGoldenSingleFloat(self, 4.160992, logp)
      test_utils.CompareToGoldenSingleFloat(self, 5.944274, logp_per_word)
      test_utils.CompareToGoldenSingleFloat(self, 0.000000, accuracy)

  def testLmTrain(self):
    p = self._Params()
    p.input = self._InputParams(for_training=True)
    tp = p.train
    tp.learning_rate = 3e-3

    with self.session() as sess:
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

  def testLmInference(self):
    tf.set_random_seed(93820986)
    p = self._Params()
    p.input = self._InputParams(for_training=False)
    tf.logging.info('Params: %s', p.ToText())

    with self.session(use_gpu=False) as sess:
      mdl = p.cls(p)
      subgraphs = mdl.Inference()
      self.assertTrue('default' in subgraphs)
      fetches, feeds = subgraphs['default']
      tf.global_variables_initializer().run()
      vals = sess.run(
          fetches=fetches,
          feed_dict={feeds['text']: ['pray for world peace', 'happy birthday']})
      print('actual vals = ', vals)
      self.assertEqual(vals['log_pplx_per_sample'].shape, (2,))
      self.assertEqual(vals['log_pplx_per_token'].shape, (2, 20))
      self.assertEqual(vals['paddings'].shape, (2, 20))

  def testLmInferenceWordLevel(self):
    tf.set_random_seed(93820986)
    p = self._Params()
    p.input = self._InputParams(for_training=False)
    p.input.tokenizer = tokenizers.VocabFileTokenizer.Params()
    p.input.tokenizer.vocab_size = 64
    p.input.target_max_length = 5
    p.input.pad_to_max_seq_length = True
    # target_{sos,eos,unk}_id must be consistent with token_vocab_filepath.
    p.input.tokenizer.target_sos_id = 1
    p.input.tokenizer.target_eos_id = 2
    p.input.tokenizer.target_unk_id = 3
    p.input.tokenizer.token_vocab_filepath = test_helper.test_src_dir_path(
        'tasks/lm/testdata/small_word_vocab.txt')
    tf.logging.info('Params: %s', p.ToText())

    with self.session(use_gpu=False) as sess:
      mdl = p.cls(p)
      subgraphs = mdl.Inference()
      self.assertTrue('default' in subgraphs)
      fetches, feeds = subgraphs['default']
      tf.global_variables_initializer().run()
      vals = sess.run(
          fetches=fetches,
          feed_dict={
              feeds['text']: [
                  'pray for more peace', 'happy about', 'one flambergastic will'
              ]
          })
      print('actual vals = ', vals)
      self.assertEqual(vals['log_pplx_per_sample'].shape, (3,))
      self.assertEqual(vals['log_pplx_per_token'].shape, (3, 5))
      self.assertEqual(vals['paddings'].shape, (3, 5))
      expected_tokens_from_labels = [
          '<UNK> for more <UNK> </S>', '<UNK> about </S>', 'one <UNK> will </S>'
      ]
      self.assertListEqual(vals['tokens_from_labels'].tolist(),
                           expected_tokens_from_labels)
      expected_num_oovs_per_sample = [2, 1, 1]
      self.assertListEqual(vals['num_oovs_per_sample'].tolist(),
                           expected_num_oovs_per_sample)
      expected_ids = [[1, 1, 1], [3, 3, 41], [21, 8, 3], [35, 2, 61], [3, 2, 2]]
      self.assertListEqual(vals['ids'].tolist(), expected_ids)


if __name__ == '__main__':
  tf.test.main()
