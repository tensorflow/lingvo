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
# ==============================================================================
"""Tests for input generator."""

from absl.testing import parameterized
import lingvo.compat as tf
from lingvo.core import cluster_factory
from lingvo.core import py_utils
from lingvo.core import test_helper
from lingvo.core import test_utils
from lingvo.core import tokenizers
from lingvo.tasks.mt import input_generator
import mock
import numpy as np


class InputTest(test_utils.TestCase, parameterized.TestCase):

  def _CreateMlPerfInputParams(self):
    p = input_generator.MlPerfInput.Params()
    input_file = test_helper.test_src_dir_path(
        'tasks/mt/testdata/translate_ende_wmt32k-train-00511-of-00512')
    p.file_pattern = 'tfrecord:' + input_file
    p.file_random_seed = 31415
    p.file_parallelism = 1
    p.bucket_upper_bound = [20, 40]
    p.bucket_batch_limit = [4, 8]
    return p

  def _CreateMlPerfPackedInputParams(self):
    p = input_generator.MlPerfInput.Params()
    input_file = test_helper.test_src_dir_path(
        'tasks/mt/testdata/translate_ende_mlperf.packed.tfrecord')
    p.file_pattern = 'tfrecord:' + input_file
    p.packed_input = True
    p.file_random_seed = 31415
    p.file_parallelism = 1
    p.bucket_upper_bound = [20, 240]
    p.bucket_batch_limit = [4, 4]
    return p

  def _CreateNmtInputParams(self):
    p = input_generator.NmtInput.Params()
    input_file = test_helper.test_src_dir_path(
        'tasks/mt/testdata/wmt14_ende_wpm_32k_test.tfrecord')
    p.file_pattern = 'tfrecord:' + input_file
    p.file_random_seed = 31415
    p.file_parallelism = 1
    p.bucket_upper_bound = [20, 40]
    p.bucket_batch_limit = [4, 8]
    return p

  def testBasic(self):
    p = self._CreateNmtInputParams()
    with self.session(use_gpu=False):
      inp = input_generator.NmtInput(p)
      # Runs a few steps.
      for _ in range(10):
        self.evaluate(inp.GetPreprocessedInputBatch())

  def testMlPerfPackedInput(self):
    p = self._CreateMlPerfPackedInputParams()
    with self.session(use_gpu=False):
      inp = input_generator.MlPerfInput(p)
      for _ in range(1):
        fetched = py_utils.NestedMap(
            self.evaluate(inp.GetPreprocessedInputBatch()))
        tf.logging.info(fetched.src.ids.shape)
        tf.logging.info(fetched.src.segment_ids.shape)
        tf.logging.info(fetched.src.segment_pos.shape)
        tf.logging.info(fetched.tgt.segment_ids.shape)
        tf.logging.info(fetched.tgt.segment_pos.shape)

  def checkPadShape(self, x, pad, batch_size, actual_max, pad_length):
    # Check the shape: (batch, maxlen)
    self.assertEqual(x.shape, (batch_size, pad_length))
    # Check the padding.
    self.assertAllEqual(x[:, actual_max:],
                        np.full((batch_size, (pad_length - actual_max)), pad))

  def testMlPerfPackedInputPadToMax(self):
    p = self._CreateMlPerfPackedInputParams()
    p.source_max_length = 300
    p.target_max_length = 300
    p.pad_to_max_seq_length = True
    with self.session(use_gpu=False):
      inp = input_generator.MlPerfInput(p)
      for _ in range(1):
        fetched = py_utils.NestedMap(
            self.evaluate(inp.GetPreprocessedInputBatch()))

    self.checkPadShape(
        fetched.src.ids, pad=0, batch_size=4, actual_max=240, pad_length=300)

    self.checkPadShape(
        fetched.tgt.ids, pad=0, batch_size=4, actual_max=240, pad_length=300)

    self.checkPadShape(
        fetched.tgt.segment_ids,
        pad=0,
        batch_size=4,
        actual_max=240,
        pad_length=300)

    self.checkPadShape(
        fetched.tgt.segment_pos,
        pad=0,
        batch_size=4,
        actual_max=240,
        pad_length=300)

  def testMlPerf(self):
    p = self._CreateMlPerfInputParams()
    with self.session(use_gpu=False):
      inp = input_generator.MlPerfInput(p)
      # Runs a few steps.
      for _ in range(10):
        fetched = py_utils.NestedMap(
            self.evaluate(inp.GetPreprocessedInputBatch()))
        tf.logging.info(fetched)

  def testMlPerfPadToMax(self):
    p = self._CreateMlPerfInputParams()
    p.bucket_upper_bound = [20]
    p.bucket_batch_limit = [4]
    p.source_max_length = 30
    p.target_max_length = 30
    p.pad_to_max_seq_length = True

    with self.session(use_gpu=False):
      inp = input_generator.MlPerfInput(p)
      # Runs a few steps.
      for _ in range(10):
        fetched = py_utils.NestedMap(
            self.evaluate(inp.GetPreprocessedInputBatch()))

    def Check(x, pad):
      # Check the shape: (batch, maxlen)
      self.assertEqual(x.shape, (4, 30))
      # Check the padding.
      self.assertAllEqual(x[:, 20:], np.full((4, 10), pad))
    Check(fetched.src.ids, 0)
    Check(fetched.src.paddings, 1)
    Check(fetched.tgt.ids, 0)
    Check(fetched.tgt.labels, 0)
    Check(fetched.tgt.weights, 0)
    Check(fetched.tgt.paddings, 1)

  def testPadToMax(self):
    p = self._CreateNmtInputParams()
    p.bucket_upper_bound = [20]
    p.bucket_batch_limit = [4]
    p.source_max_length = 30
    p.target_max_length = 30
    p.pad_to_max_seq_length = True
    with self.session(use_gpu=False):
      inp = input_generator.NmtInput(p)
      fetched = py_utils.NestedMap(
          self.evaluate(inp.GetPreprocessedInputBatch()))

    def Check(x, pad):
      # Check the shape: (batch, maxlen)
      self.assertEqual(x.shape, (4, 30))
      # Check the padding.
      self.assertAllEqual(x[:, 20:], np.full((4, 10), pad))

    Check(fetched.src.ids, 0)
    Check(fetched.src.paddings, 1)
    Check(fetched.tgt.ids, 0)
    Check(fetched.tgt.labels, 0)
    Check(fetched.tgt.weights, 0)
    Check(fetched.tgt.paddings, 1)

  def testSplitSources(self):
    p = self._CreateNmtInputParams()
    num_splits = 2
    expected_ids_split_1 = [
        [
            93, 15027, 643, 8, 2985, 3, 27025, 6, 4569, 2, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0
        ],
        [
            15027, 1668, 4125, 54, 139, 24, 3, 101, 8, 2031, 5545, 2962, 5, 2,
            0, 0, 0, 0, 0, 0
        ],
    ]

    expected_ids_split_2 = [
        [
            626, 854, 11, 392, 45, 77, 67, 1346, 30, 25, 10, 2283, 933, 14,
            22255, 425, 872, 4677, 5, 2
        ],
        [
            52, 21, 1034, 4, 3, 274, 30, 7203, 6275, 3, 967, 795, 142, 5, 2, 0,
            0, 0, 0, 0
        ],
    ]

    with self.session(use_gpu=False):
      inp = input_generator.NmtInput(p)
      splits = inp.SplitInputBatch(num_splits)
      split_ids = self.evaluate([splits[0].src.ids, splits[1].src.ids])
      tf.logging.info('split_ids[0] = %r', split_ids[0])
      tf.logging.info('split_ids[1] = %r', split_ids[1])
      self.assertAllEqual(expected_ids_split_1, split_ids[0])
      self.assertAllEqual(expected_ids_split_2, split_ids[1])

  def testSplitTargets(self):
    p = self._CreateNmtInputParams()
    num_splits = 2

    with self.session(use_gpu=False):
      inp = input_generator.NmtInput(p)
      fetched = self.evaluate(inp.SplitInputBatch(num_splits))

    expected_ids_split_1 = [
        [
            1, 272, 7514, 10944, 2220, 815, 3, 39, 6, 3021, 4893, 10, 6693,
            23788, 3410, 0, 0, 0, 0
        ],
        [
            1, 28, 18764, 6, 1413, 2338, 8068, 107, 431, 14, 6, 1083, 3, 11,
            782, 19664, 9, 3622, 4
        ],
    ]

    expected_ids_split_2 = [
        [
            1, 15149, 12, 583, 43, 61, 179, 1265, 22, 27, 7193, 16, 5, 782,
            14077, 6734, 4, 0, 0
        ],
        [
            1, 81, 90, 1397, 9207, 61, 241, 2102, 15, 3003, 424, 6, 483, 4, 0,
            0, 0, 0, 0
        ],
    ]

    tf.logging.info('fetched[0].tgt.ids = %r', fetched[0].tgt.ids)
    tf.logging.info('fetched[1].tgt.ids = %r', fetched[1].tgt.ids)
    self.assertAllEqual(expected_ids_split_1, fetched[0].tgt.ids)
    self.assertAllEqual(expected_ids_split_2, fetched[1].tgt.ids)

  def testTextPackedInputProto(self):
    p = input_generator.TextPackedInput.Params()
    p.flush_every_n = 0
    p.repeat_count = 1
    p.file_pattern = 'tfrecord:' + test_helper.test_src_dir_path(
        'tasks/mt/testdata/en_fr.tfrecord')
    p.pad_to_max_seq_length = True
    p.tokenizer = tokenizers.AsciiTokenizer.Params()
    p.input_file_type = 'sentence_proto'
    p.source_max_length = 22
    p.target_max_length = 24
    p.bucket_batch_limit = [2]
    with self.session() as sess:
      inp = p.Instantiate()
      batch_tensor = inp.GetPreprocessedInputBatch()
      for k, x in batch_tensor.FlattenItems():
        self.assertTrue(x.shape.is_fully_defined(), k)
      batch = sess.run(batch_tensor)
    self.assertLen(batch.src, 8)
    self.assertAllEqual(batch.src.strs,
                        [b'I love paragliding!', b'vol biv paragliding'])
    self.assertAllEqual(batch.tgt.strs,
                        [b"J'adore le parapente!", b'vol biv parapente'])
    self.assertAllEqual(
        batch.src.ids,
        np.array([
            [
                13, 3, 16, 19, 26, 9, 3, 20, 5, 22, 5, 11, 16, 13, 8, 13, 18,
                11, 35, 2, 0, 0
            ],
            [
                26, 19, 16, 3, 6, 13, 26, 3, 20, 5, 22, 5, 11, 16, 13, 8, 13,
                18, 11, 2, 0, 0
            ],
        ]))
    self.assertAllEqual(
        batch.tgt.ids,
        np.array([
            [
                1, 14, 32, 5, 8, 19, 22, 9, 3, 16, 9, 3, 20, 5, 22, 5, 20, 9,
                18, 24, 9, 35, 0, 0
            ],
            [
                1, 26, 19, 16, 3, 6, 13, 26, 3, 20, 5, 22, 5, 20, 9, 18, 24, 9,
                0, 0, 0, 0, 0, 0
            ],
        ]))
    self.assertAllEqual(
        batch.tgt.labels,
        np.array([
            [
                14, 32, 5, 8, 19, 22, 9, 3, 16, 9, 3, 20, 5, 22, 5, 20, 9, 18,
                24, 9, 35, 2, 0, 0
            ],
            [
                26, 19, 16, 3, 6, 13, 26, 3, 20, 5, 22, 5, 20, 9, 18, 24, 9, 2,
                0, 0, 0, 0, 0, 0
            ],
        ]))

  @parameterized.named_parameters(
      ('no_per_host_infeed_no_packing', False, None),
      ('per_host_infeed_no_packing', True, None),
      ('no_per_host_infeed_with_packing', False, 3.5),
      ('per_host_infeed_with_packing', True, 3.5))
  def testTextPackedInputBatchSize(self, use_per_host_infeed, packing_factor):
    p = cluster_factory.Current().params.Copy()
    p.job = 'trainer'
    p.worker.tpus_per_replica = 8
    p.worker.num_tpu_hosts = 16
    p.worker.devices_per_split = 2
    cluster = p.Instantiate()

    with cluster, mock.patch('lingvo.core.py_utils.use_tpu', return_value=True):
      p = input_generator.TextPackedInput.Params()
      p.use_per_host_infeed = use_per_host_infeed
      p.file_random_seed = 0
      p.file_pattern = 'tfrecord:' + test_helper.test_src_dir_path(
          'tasks/mt/testdata/en_fr.tfrecord')
      p.pad_to_max_seq_length = True
      p.tokenizer = tokenizers.AsciiTokenizer.Params()
      p.input_file_type = 'sentence_proto'
      p.source_max_length = 32
      p.target_max_length = 32
      p.bucket_batch_limit = [128]
      p.packing_factor = packing_factor
      with self.session() as sess:
        inp = p.Instantiate()
        # GlobalBatchSize is batch_size (128) * num_splits_per_client (4).
        # num_splits_per_client is 4, because num_splits_per_replica is 4.
        # num_splits_per_replica is 4 because that's tpus_per_replica
        # divided by devices_per_split.
        expected_global_batch_size = (
            p.bucket_batch_limit[0] // cluster.params.worker.devices_per_split *
            cluster.params.worker.tpus_per_replica)
        if p.packing_factor is not None:
          expected_global_batch_size = np.math.floor(
              expected_global_batch_size * p.packing_factor)

        expected_infeed_batch_size = expected_global_batch_size
        if use_per_host_infeed:
          expected_infeed_batch_size = (
              expected_global_batch_size // cluster.params.worker.num_tpu_hosts)

        expected_packed_infeed_batch_size = expected_infeed_batch_size
        if p.packing_factor is not None:
          expected_packed_infeed_batch_size = np.math.floor(
              expected_infeed_batch_size / p.packing_factor)

        self.assertEqual(expected_global_batch_size, inp.GlobalBatchSize())
        self.assertEqual(expected_infeed_batch_size, inp.InfeedBatchSize())

        batch_tensor = inp.GetPreprocessedInputBatch()
        for k, x in batch_tensor.FlattenItems():
          self.assertTrue(x.shape.is_fully_defined(), k)
        batch = sess.run(batch_tensor)
        self.assertEqual(batch.src.ids.shape,
                         (expected_packed_infeed_batch_size, 32))

  def testTextPackedInputTextWpm(self):
    p = input_generator.TextPackedInput.Params()
    p.flush_every_n = 0
    p.repeat_count = 1
    p.file_pattern = 'text:' + test_helper.test_src_dir_path(
        'tasks/mt/testdata/en_de.text')
    p.tokenizer = tokenizers.WpmTokenizer.Params().Set(
        vocab_filepath=test_helper.test_src_dir_path(
            'tasks/mt/wpm-ende-2k.voc'),
        vocab_size=2000)
    p.source_max_length = 12
    p.target_max_length = 15
    p.bucket_batch_limit = [2]
    with self.session() as sess:
      inp = p.Instantiate()
      batch_tensor = inp.GetPreprocessedInputBatch()
      batch = sess.run(batch_tensor)
      print(batch)
    self.assertAllEqual(
        batch.src.ids,
        np.array([[109, 251, 98, 595, 1009, 245, 326, 129, 4, 2, 0, 0],
                  [115, 276, 18, 66, 2, 0, 0, 0, 0, 0, 0, 0]]))
    self.assertAllEqual(
        batch.tgt.ids,
        np.array([[
            1, 197, 446, 458, 419, 284, 323, 1411, 571, 456, 409, 13, 4, 0, 0
        ], [1, 115, 281, 18, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
    self.assertAllEqual(
        batch.tgt.labels,
        np.array([[
            197, 446, 458, 419, 284, 323, 1411, 571, 456, 409, 13, 4, 2, 0, 0
        ], [115, 281, 18, 66, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

  def testTextPackedInputTextPacking(self):
    p = input_generator.TextPackedInput.Params()
    p.flush_every_n = 0
    p.file_pattern = 'text:' + test_helper.test_src_dir_path(
        'tasks/mt/testdata/en_de.text')
    p.tokenizer = tokenizers.WpmTokenizer.Params().Set(
        vocab_filepath=test_helper.test_src_dir_path(
            'tasks/mt/wpm-ende-2k.voc'),
        vocab_size=2000)
    # We repeat the 2-line file twice for a batch of 2, each packing both lines.
    p.repeat_count = 2
    p.source_max_length = 16
    p.target_max_length = 20
    p.bucket_batch_limit = [2]
    p.packing_factor = 2
    with self.session() as sess:
      inp = p.Instantiate()
      batch_tensor = inp.GetPreprocessedInputBatch()
      batch = sess.run(batch_tensor)
    self.assertAllEqual(
        batch.src.ids,
        np.array([
            [
                109, 251, 98, 595, 1009, 245, 326, 129, 4, 2, 115, 276, 18, 66,
                2, 0
            ],
            [
                115, 276, 18, 66, 2, 109, 251, 98, 595, 1009, 245, 326, 129, 4,
                2, 0
            ],
        ]))
    self.assertAllEqual(
        batch.src.segment_ids,
        np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0],
                  [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0]],
                 dtype=np.float32))
    self.assertAllEqual(
        batch.src.segment_pos,
        np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 0],
                  [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]))
    self.assertAllEqual(
        batch.src.strs,
        np.array([
            b'Too much has changed.\tHello!', b'Hello!\tToo much has changed.'
        ]))

    self.assertAllEqual(
        batch.tgt.ids,
        np.array([
            [
                1, 197, 446, 458, 419, 284, 323, 1411, 571, 456, 409, 13, 4, 1,
                115, 281, 18, 66, 0, 0
            ],
            [
                1, 115, 281, 18, 66, 1, 197, 446, 458, 419, 284, 323, 1411, 571,
                456, 409, 13, 4, 0, 0
            ],
        ]))
    self.assertAllEqual(
        batch.tgt.labels,
        np.array([
            [
                197, 446, 458, 419, 284, 323, 1411, 571, 456, 409, 13, 4, 2,
                115, 281, 18, 66, 2, 0, 0
            ],
            [
                115, 281, 18, 66, 2, 197, 446, 458, 419, 284, 323, 1411, 571,
                456, 409, 13, 4, 2, 0, 0
            ],
        ]))
    self.assertAllEqual(
        batch.tgt.segment_ids,
        np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0],
                  [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0]],
                 dtype=np.float32))
    self.assertAllEqual(
        batch.tgt.segment_pos,
        np.array(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 0, 0],
             [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0]]))
    self.assertAllEqual(
        batch.tgt.strs,
        np.array([
            b'Daf\xc3\xbcr hat sich zu viel ver\xc3\xa4ndert.\tHallo!',
            b'Hallo!\tDaf\xc3\xbcr hat sich zu viel ver\xc3\xa4ndert.'
        ]))

  @parameterized.named_parameters(('1', 1., 1., 8), ('2', 2., 1., 16),
                                  ('3', 3., .6667, 16))
  def testMetrics(self, packing_factor, expected_ratio, expected_count):
    p = input_generator.TextPackedInput.Params()
    p.flush_every_n = 0
    p.repeat_count = -1
    p.file_pattern = 'text:' + test_helper.test_src_dir_path(
        'tasks/mt/testdata/en_de.text')
    p.tokenizer = tokenizers.WpmTokenizer.Params().Set(
        vocab_filepath=test_helper.test_src_dir_path(
            'tasks/mt/wpm-ende-2k.voc'),
        vocab_size=2000)
    p.source_max_length = 20
    p.target_max_length = 20
    p.bucket_batch_limit = [8]
    p.packing_factor = packing_factor
    with cluster_factory.ForTestingWorker(add_summary=True):
      with self.session() as sess:
        inp = p.Instantiate()
        inp.GetPreprocessedInputBatch()
        summary_str = sess.run(tf.summary.merge_all(scope='examples'))
        summary = tf.summary.Summary.FromString(summary_str)

        self.assertLen(summary.value, 3)
        self.assertEqual(summary.value[0].tag,
                         'examples/src_packed_token_ratio')
        self.assertEqual(summary.value[1].tag,
                         'examples/tgt_packed_token_ratio')
        self.assertEqual(summary.value[2].tag, 'examples/num_packed_examples')
        self.assertAllClose(
            summary.value[0].simple_value, expected_ratio, atol=0.0001)
        self.assertAllClose(
            summary.value[1].simple_value, expected_ratio, atol=0.0001)
        self.assertEqual(summary.value[2].simple_value, expected_count)


class DoubleInputTest(test_utils.TestCase, parameterized.TestCase):

  def _CreateNmtInputParams(self):
    p = input_generator.NmtDoubleInput.Params()
    input_file = test_helper.test_src_dir_path(
        'tasks/mt/testdata/wmt14_ende_wpm_32k_doublebatch_test-000-001')
    p.file_pattern = 'tfrecord:' + input_file
    p.tokenizer.token_vocab_filepath = test_helper.test_src_dir_path(
        'tasks/mt/testdata/wmt14_ende_wpm_32k_test.vocab')
    p.file_random_seed = 31415
    p.file_parallelism = 1
    p.bucket_upper_bound = [10, 20]
    p.bucket_batch_limit = [4, 2]
    p.source_mask_ratio = -1
    p.source_mask_ratio_beta = '2,6'
    p.mask_word_id = 31999
    p.pad_id = 31998
    p.mask_words_ratio = 0.25
    p.permutation_distance = 3
    p.vocab_file = p.tokenizer.token_vocab_filepath
    p.packed_input = False
    return p

  def _CreatePackedNmtInputParams(self):
    p = self._CreateNmtInputParams()
    p.packed_input = True
    p.bucket_upper_bound = [10, 20, 40]
    p.bucket_batch_limit = [4, 2, 1]
    return p

  def testBasicInput(self):
    self.skipTest('TODO(b/196852574): Reenable.')
    p = self._CreateNmtInputParams()
    with self.session(use_gpu=False) as sess:
      inp = p.Instantiate()
      batch_tensor = inp.GetPreprocessedInputBatch()
      batch = sess.run(batch_tensor)
    self.assertAllEqual([2, 20], batch.src.ids.shape)
    self.assertAllEqual(
        batch.src.ids,
        [[
            30, 23020, 1497, 4593, 3870, 23880, 833, 3, 10311, 7, 1632, 3, 11,
            2267, 76, 7, 249, 4, 2, 31998
        ],
         [
             125, 2475, 883, 103, 5004, 7, 784, 10182, 1990, 4, 2, 31998, 31998,
             31998, 31998, 31998, 31998, 31998, 31998, 31998
         ]])
    self.assertAllEqual(
        batch.other_src.ids,
        [[
            7499, 31999, 3, 741, 31999, 173, 41, 1354, 3316, 4, 2, 31998, 31998,
            31998, 31998, 31998, 31998, 31998, 31998, 31998
        ],
         [
             27822, 510, 223, 31999, 46, 9, 13220, 12, 21965, 2241, 35, 31999,
             4104, 31999, 1179, 235, 31999, 4, 2, 31998
         ]])
    self.assertAllEqual(batch.src.source_mask, [
        [
            0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.,
            1., 0., 0.
        ],
        [
            1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0.
        ],
    ])

  def testPackedInput(self):
    self.skipTest('TODO(b/196852574): Reenable.')
    p = self._CreatePackedNmtInputParams()
    with self.session(use_gpu=False) as sess:
      inp = p.Instantiate()
      batch_tensor = inp.GetPreprocessedInputBatch()
      for _ in range(7):
        batch = sess.run(batch_tensor)
    self.assertAllEqual([1, 40], batch.src.ids.shape)
    self.assertAllEqual(batch.src.segment_ids, [[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
        2., 2., 2., 2.
    ]])
    self.assertAllEqual(batch.other_src.segment_pos, [[
        0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        16., 17., 18., 19., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.,
        12., 13., 14., 15., 16., 17., 18., 19.
    ]])
    self.assertAllEqual(batch.src.source_mask, [[
        0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.
    ]])


if __name__ == '__main__':
  tf.test.main()
