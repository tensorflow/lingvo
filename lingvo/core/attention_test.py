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
"""Tests for attention."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow as tf

from lingvo.core import attention
from lingvo.core import py_utils
from lingvo.core import quant_utils
from lingvo.core import test_utils


class AttentionTest(test_utils.TestCase):
  """Test attention models."""

  def _CheckStaticShapes(self, atten_vec, atten_prob, target_batch_size,
                         source_length, context_dim):
    """Static shape must be set correctly for RNN beam search compatibility."""
    self.assertIsNotNone(atten_prob.shape.ndims)
    self.assertEqual((target_batch_size, source_length), atten_prob.shape)
    self.assertIsNotNone(atten_vec.shape.ndims)
    self.assertEqual((target_batch_size, context_dim), atten_vec.shape)

  def _AdditiveAttentionInputs(self, packed_inputs=False, tgt_bs=6):
    np.random.seed(12345)
    source_vecs = tf.constant(np.random.rand(6, 3, 4), dtype=tf.float32)
    source_contexts = tf.constant(np.random.rand(6, 3, 5), dtype=tf.float32)
    source_padding = tf.transpose(
        tf.constant(
            [[0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0]],
            dtype=tf.float32))
    source_segment_id = tf.transpose(
        tf.constant(
            [[0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 1, 2]],
            dtype=tf.float32))
    query_vec = tf.constant(np.random.rand(tgt_bs, 7), dtype=tf.float32)
    qsi = [0, 1, 1, 1, 2, 2]
    query_segment_id = tf.constant(qsi[:tgt_bs], dtype=tf.float32)

    params = attention.AdditiveAttention.Params()
    params.name = 'atten'
    params.params_init = py_utils.WeightInit.Gaussian(0.1, 12345)
    params.source_dim = 4
    params.query_dim = 7
    params.hidden_dim = 7
    params.vn.global_vn = False
    params.vn.per_step_vn = False
    params.packed_input = packed_inputs
    tensors = (source_vecs, source_contexts, source_padding, source_segment_id,
               query_vec, query_segment_id)
    return params, tensors

  def testAdditiveAttention(self):
    with self.session(use_gpu=True) as sess:
      params, tensors = self._AdditiveAttentionInputs()
      source_vecs, source_contexts, source_padding, _, query_vec, _ = tensors
      atten = attention.AdditiveAttention(params)
      self.assertEqual(len(atten.vars.Flatten()), 3)
      atten.InitForSourcePacked(atten.theta, source_vecs, source_contexts,
                                source_padding)
      atten_vec, atten_prob, _ = atten.ComputeContextVector(
          atten.theta, query_vec)

      self._CheckStaticShapes(
          atten_vec,
          atten_prob,
          target_batch_size=query_vec.shape[0],
          source_length=source_contexts.shape[0],
          context_dim=source_contexts.shape[2])

      # TODO(yonghui): add atten.vars for the variables attention model
      # declares.
      atten_vars = tf.get_collection('AdditiveAttention_vars')
      self.assertEqual(3, len(atten_vars))

      tf.global_variables_initializer().run()

      all_vars = tf.trainable_variables()
      for v in all_vars:
        print(v.eval())

      atten_vec_out, prob_out = sess.run([atten_vec, atten_prob])

      print(['additive attention prob_out', np.array_repr(prob_out)])
      print(['additive attention atten_vec_out', np.array_repr(atten_vec_out)])

      # pyformat: disable
      # pylint: disable=bad-whitespace
      expected_prob_out = [
          [0.2555742 ,  0.24073002,  0.        ,  0.        ,  0.25412574,
           0.24957004],
          [0.        ,  0.25394136,  0.24764746,  0.25480017,  0.        ,
           0.24361098],
          [0.25094295,  0.2499937 ,  0.        ,  0.24308342,  0.        ,
           0.25597993],
          [0.25559244,  0.24070661,  0.        ,  0.        ,  0.25412717,
           0.24957375],
          [0.        ,  0.25393167,  0.24765188,  0.25481117,  0.        ,
           0.24360526],
          [0.25113183,  0.24990553,  0.        ,  0.24246082,  0.        ,
           0.25650182]]

      expected_atten_vec_out = [
          [0.49745506,  0.63471669,  0.49220526,  0.5683012 ,  0.42753702],
          [0.51502365,  0.56183743,  0.37644109,  0.87425125,  0.46182787],
          [0.57862502,  0.44246522,  0.36931852,  0.41002905,  0.14327194],
          [0.49745634,  0.63471717,  0.49220967,  0.56829125,  0.4275257 ],
          [0.51501834,  0.56183696,  0.37644821,  0.87425053,  0.46182543],
          [0.57893348,  0.44248882,  0.36938411,  0.41006744,  0.14328158]]
      # pylint: enable=bad-whitespace
      # pyformat: enable

      self.assertAllClose(expected_prob_out, prob_out)
      self.assertAllClose(expected_atten_vec_out, atten_vec_out)

  def testAdditiveAttentionWithPackedInputs(self):
    with self.session(use_gpu=True) as sess:
      params, tensors = self._AdditiveAttentionInputs(packed_inputs=True)
      (source_vecs, source_contexts, source_padding, source_segment_id,
       query_vec, query_segment_id) = tensors
      atten = attention.AdditiveAttention(params)
      self.assertEqual(len(atten.vars.Flatten()), 3)
      atten.InitForSourcePacked(atten.theta, source_vecs, source_contexts,
                                source_padding, source_segment_id)
      atten_vec, atten_prob, _ = atten.ComputeContextVector(
          atten.theta, query_vec, query_segment_id=query_segment_id)

      # TODO(yonghui): add atten.vars for the variables attention model
      # declares.
      atten_vars = tf.get_collection('AdditiveAttention_vars')
      self.assertEqual(3, len(atten_vars))

      tf.global_variables_initializer().run()

      all_vars = tf.trainable_variables()
      for v in all_vars:
        print(v.eval())

      atten_vec_out, prob_out = sess.run([atten_vec, atten_prob])

      self._CheckStaticShapes(
          atten_vec,
          atten_prob,
          target_batch_size=query_vec.shape[0],
          source_length=source_contexts.shape[0],
          context_dim=source_contexts.shape[2])

      print(['packed additive attention prob_out', np.array_repr(prob_out)])
      print([
          'packed additive attention atten_vec_out',
          np.array_repr(atten_vec_out)
      ])

      # pyformat: disable
      # pylint: disable=bad-whitespace
      expected_prob_out = [
          [0.51495469, 0.48504525, 0         , 0         , 0         ,
           0        ],
          [0         , 0         , 0.49288213, 0.50711787, 0         ,
           0        ],
          [0.        , 0.5070073 , 0.        , 0.4929927 , 0.        ,
           0        ],
          [0.        , 0         , 0.        , 0.        , 0.50451994,
           0.49548006],
          [0.        , 0.        , 0.        , 0.        , 0.        ,
           1        ],
          [0.        , 0.        , 0.        , 0.        , 0.        ,
           1        ]]

      expected_atten_vec_out = [
          [0.35256192,  0.68348885,  0.41128731,  0.48906463,  0.50537711],
          [0.45880911,  0.6068666 ,  0.59867024,  0.82797134,  0.33504993],
          [0.54934788,  0.50335771,  0.26117462,  0.32834488,  0.16398546],
          [0.64022166,  0.58665955,  0.571935  ,  0.64637613,  0.35084069],
          [0.27927336,  0.06444023,  0.19862361,  0.93168277,  0.85441357],
          [0.95473474,  0.05225335,  0.57947171,  0.48049626,  0.02170898]]
      # pylint: enable=bad-whitespace
      # pyformat: enable

      self.assertAllClose(expected_prob_out, prob_out)
      self.assertAllClose(expected_atten_vec_out, atten_vec_out)

  def testAdditiveAttentionDeterministicDropout(self):
    with self.session(use_gpu=True) as sess:
      params, tensors = self._AdditiveAttentionInputs()
      source_vecs, source_contexts, source_padding, _, query_vec, _ = tensors
      params.atten_dropout_prob = 0.5
      params.atten_dropout_deterministic = True
      params.random_seed = 78924

      atten = attention.AdditiveAttention(params)
      self.assertEqual(len(atten.vars.Flatten()), 3)
      atten.InitForSourcePacked(atten.theta, source_vecs, source_contexts,
                                source_padding)
      atten_vec, atten_prob, _ = atten.ComputeContextVector(
          atten.theta, query_vec)

      self._CheckStaticShapes(
          atten_vec,
          atten_prob,
          target_batch_size=query_vec.shape[0],
          source_length=source_contexts.shape[0],
          context_dim=source_contexts.shape[2])

      atten_vars = tf.get_collection('AdditiveAttention_vars')
      self.assertEqual(3, len(atten_vars))

      tf.global_variables_initializer().run()

      all_vars = tf.trainable_variables()
      for v in all_vars:
        print(v.eval())

      atten_vec_out, prob_out = sess.run([atten_vec, atten_prob])

      print('additive attention prob_out %r' % prob_out)
      print('additive attention atten_vec_out %r' % atten_vec_out)

      expected_prob_out = [
          [0.51114839, 0.48146003, 0., 0., 0., 0.],
          [0., 0.50788271, 0., 0.50960034, 0., 0.48722193],
          [0., 0.49998739, 0., 0., 0., 0.51195991],
          [0., 0.48141322, 0., 0., 0.50825435, 0.4991475],
          [0., 0.50786334, 0.49530372, 0., 0., 0.48721054],
          [0., 0.49981108, 0., 0., 0., 0.51300365],
      ]

      expected_atten_vec_out = [
          [0.34995595, 0.67843682, 0.40824726, 0.4854497, 0.50164163],
          [0.60576487, 0.80303985, 0.46480939, 1.3962903, 0.79863495],
          [0.90196574, 0.47579059, 0.31802341, 0.34388986, 0.15836108],
          [0.81517166, 0.90433061, 0.72681838, 1.02123988, 0.72982419],
          [0.99326241, 0.83445895, 0.43935478, 1.26866817, 0.71197236],
          [0.90281653, 0.47568679, 0.31862068, 0.34435683, 0.15833181],
      ]

      self.assertAllClose(expected_prob_out, prob_out)
      self.assertAllClose(expected_atten_vec_out, atten_vec_out)

  def _testSameBatchSize(self, same_batch_size, packed_inputs=False):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(398847392)
      params, tensors = self._AdditiveAttentionInputs(packed_inputs, tgt_bs=3)
      source_vecs, source_contexts, source_padding, _, query_vec, _ = tensors
      params.same_batch_size = same_batch_size

      atten = attention.AdditiveAttention(params)
      atten.InitForSourcePacked(atten.theta, source_vecs, source_contexts,
                                source_padding)
      atten_vec, atten_prob, _ = atten.ComputeContextVector(
          atten.theta, query_vec)

      self._CheckStaticShapes(
          atten_vec,
          atten_prob,
          target_batch_size=query_vec.shape[0],
          source_length=source_contexts.shape[0],
          context_dim=source_contexts.shape[2])

      self.assertEqual(3, len(atten.vars.Flatten()))

    with self.session(use_gpu=True, graph=g) as sess:
      tf.global_variables_initializer().run()
      atten_vec_out, prob_out = sess.run([atten_vec, atten_prob])
    return atten_vec_out, prob_out

  def testAdditiveAttentionSameBatchSize(self):
    vec0, prob0 = self._testSameBatchSize(False)
    vec1, prob1 = self._testSameBatchSize(True)
    self.assertAllClose(vec0, vec1)
    self.assertAllClose(prob0, prob1)

  def testAdditiveAttentionSameBatchSizePackedInputs(self):
    vec0, prob0 = self._testSameBatchSize(False, True)
    vec1, prob1 = self._testSameBatchSize(True, True)
    self.assertAllClose(vec0, vec1)
    self.assertAllClose(prob0, prob1)

  def testAdditiveAttentionSmallerHiddenLayer(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(12345)
      source_vecs = tf.constant(np.random.rand(6, 3, 4), dtype=tf.float32)
      source_contexts = tf.constant(np.random.rand(6, 3, 5), dtype=tf.float32)
      source_padding = tf.transpose(
          tf.constant(
              [[0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0]],
              dtype=tf.float32))
      query_vec = tf.constant(np.random.rand(6, 7), dtype=tf.float32)

      params = attention.AdditiveAttention.Params()
      params.name = 'atten'
      params.params_init = py_utils.WeightInit.Gaussian(0.1, 12345)
      params.source_dim = 4
      params.query_dim = 7
      params.hidden_dim = 5
      params.vn.global_vn = False
      params.vn.per_step_vn = False

      atten = attention.AdditiveAttention(params)
      atten.InitForSourcePacked(atten.theta, source_vecs, source_contexts,
                                source_padding)
      atten_vec, atten_prob, _ = atten.ComputeContextVector(
          atten.theta, query_vec)

      self._CheckStaticShapes(
          atten_vec,
          atten_prob,
          target_batch_size=query_vec.shape[0],
          source_length=source_contexts.shape[0],
          context_dim=source_contexts.shape[2])

      atten_vars = tf.get_collection('AdditiveAttention_vars')
      self.assertEqual(3, len(atten_vars))

      tf.global_variables_initializer().run()

      all_vars = tf.trainable_variables()
      for v in all_vars:
        print(v.eval())

      atten_vec_out, prob_out = sess.run([atten_vec, atten_prob])

      print(['prob_out smaller hidden layer', np.array_repr(prob_out)])
      print(
          ['atten_vec_out smaller hidden layer',
           np.array_repr(atten_vec_out)])

      # pyformat: disable
      # pylint: disable=bad-whitespace
      expected_prob_out = [
          [0.25242305,  0.24356601,  0.        ,  0.        ,  0.25346902,
           0.25054196],
          [0.        ,  0.25230604,  0.24693871,  0.25406054,  0.        ,
           0.24669473],
          [0.2501823 ,  0.24922216,  0.        ,  0.24693316,  0.        ,
           0.25366238],
          [0.25267059,  0.24300526,  0.        ,  0.        ,  0.25369659,
           0.25062758],
          [0.        ,  0.25272119,  0.24642748,  0.25435579,  0.        ,
           0.24649554],
          [0.25044653,  0.24924593,  0.        ,  0.24560687,  0.        ,
           0.25470066]]

      expected_atten_vec_out = [
          [0.49746257,  0.63428223,  0.4914251 ,  0.57035601,  0.42964566],
          [0.51383036,  0.55960417,  0.37601081,  0.87443453,  0.46342701],
          [0.57660079,  0.44147781,  0.36953348,  0.41017395,  0.14293665],
          [0.49755943,  0.63429612,  0.49157569,  0.57015073,  0.42933062],
          [0.51371205,  0.55982226,  0.37590009,  0.87454152,  0.4633899 ],
          [0.57732767,  0.44161472,  0.36958888,  0.41019297,  0.14298658]]
      # pylint: enable=bad-whitespace
      # pyformat: enable

      self.assertAllClose(expected_prob_out, prob_out)
      self.assertAllClose(expected_atten_vec_out, atten_vec_out)

  def testAdditiveAttentionFp16NoNaN(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(12345)
      source_vecs = tf.constant(np.random.rand(6, 3, 4), dtype=tf.float16)
      source_contexts = tf.constant(np.random.rand(6, 3, 5), dtype=tf.float16)
      source_padding = tf.transpose(
          tf.constant(
              [[0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0]],
              dtype=tf.float16))
      query_vec = tf.constant(np.random.rand(6, 7), dtype=tf.float16)

      params = attention.AdditiveAttention.Params()
      params.dtype = tf.float16
      params.name = 'atten'
      params.params_init = py_utils.WeightInit.Gaussian(0.1, 12345)
      params.source_dim = 4
      params.query_dim = 7
      params.hidden_dim = 7
      params.vn.global_vn = False
      params.vn.per_step_vn = False

      atten = attention.AdditiveAttention(params)
      self.assertEqual(len(atten.vars.Flatten()), 3)
      atten.InitForSourcePacked(atten.theta, source_vecs, source_contexts,
                                source_padding)
      atten_vec, atten_prob, _ = atten.ComputeContextVector(
          atten.theta, query_vec)

      self._CheckStaticShapes(
          atten_vec,
          atten_prob,
          target_batch_size=query_vec.shape[0],
          source_length=source_contexts.shape[0],
          context_dim=source_contexts.shape[2])

      # TODO(yonghui): add atten.vars for the variables attention model
      # declares.
      atten_vars = tf.get_collection('AdditiveAttention_vars')
      self.assertEqual(3, len(atten_vars))

      tf.global_variables_initializer().run()

      all_vars = tf.trainable_variables()
      for v in all_vars:
        print(v.eval())

      atten_vec_out, prob_out = sess.run([atten_vec, atten_prob])
      print(atten_vec_out.dtype)
      print(prob_out.dtype)
      self.assertTrue(np.all(np.isfinite(atten_vec_out)))
      self.assertTrue(np.all(np.isfinite(prob_out)))

  def testAdditiveAttentionVN64bits(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(12345)
      source_vecs = tf.constant(np.random.rand(5, 3, 4), dtype=tf.float64)
      source_contexts = tf.constant(np.random.rand(5, 3, 5), dtype=tf.float64)
      source_padding = tf.transpose(
          tf.constant(
              [[0, 0, 1, 1, 0], [1, 0, 0, 0, 1], [0, 0, 1, 0, 1]],
              dtype=tf.float64))
      query_vec = tf.constant(np.random.rand(6, 7), dtype=tf.float64)

      params = attention.AdditiveAttention.Params()
      params.name = 'atten'
      params.dtype = tf.float64
      params.params_init = py_utils.WeightInit.Gaussian(0.1, 12345)
      params.source_dim = 4
      params.query_dim = 7
      params.hidden_dim = 5
      params.vn.global_vn = True
      params.vn.per_step_vn = True
      params.vn.scale = 1.0
      params.vn.seed = 54321

      atten = attention.AdditiveAttention(params)
      atten.InitForSourcePacked(atten.theta, source_vecs, source_contexts,
                                source_padding)
      atten_vec, atten_prob, _ = atten.ComputeContextVector(
          atten.theta, query_vec)

      self._CheckStaticShapes(
          atten_vec,
          atten_prob,
          target_batch_size=query_vec.shape[0],
          source_length=source_contexts.shape[0],
          context_dim=source_contexts.shape[2])

      atten_vars = tf.get_collection('AdditiveAttention_vars')
      self.assertEqual(3, len(atten_vars))

      tf.global_variables_initializer().run()

      all_vars = tf.trainable_variables()
      for v in all_vars:
        print(v.eval())

      atten_vec_out, prob_out = sess.run([atten_vec, atten_prob])

      print(['prob_out with vn:', np.array_repr(prob_out)])
      print(['atten_vec_out with vn:', np.array_repr(atten_vec_out)])

      # pyformat: disable
      # pylint: disable=bad-whitespace
      expected_prob_out =[
          [ 0.43249266,  0.18638571,  0.        ,  0.        ,  0.38112162],
          [ 0.        ,  0.32589137,  0.3505654 ,  0.32354323,  0.        ],
          [ 0.26777833,  0.43991441,  0.        ,  0.29230726,  0.        ],
          [ 0.34583678,  0.32633085,  0.        ,  0.        ,  0.32783237],
          [ 0.        ,  0.32734872,  0.34749836,  0.32515292,  0.        ],
          [ 0.33614176,  0.33607175,  0.        ,  0.32778649,  0.        ]
      ]
      expected_atten_vec_out = [
          [ 0.56117282,  0.37872234,  0.42109472,  0.38981267,  0.45946841],
          [ 0.85743407,  0.37325286,  0.66322611,  0.69286686,  0.141359  ],
          [ 0.7377786 ,  0.42298519,  0.39970782,  0.67703222,  0.4157012 ],
          [ 0.51011499,  0.35817489,  0.47894328,  0.41259201,  0.54384056],
          [ 0.85716326,  0.37340558,  0.66250852,  0.69187486,  0.14179651],
          [ 0.78078121,  0.45646575,  0.4052385 ,  0.68248276,  0.43502425]
      ]
      # pylint: enable=bad-whitespace
      # pyformat: enable

      self.assertAllClose(expected_prob_out, prob_out)
      self.assertAllClose(expected_atten_vec_out, atten_vec_out)

  def _DotProductAttention(self, packed_inputs=False):
    # TODO(colincherry): Dead code?
    with self.session(use_gpu=True) as sess:
      np.random.seed(12345)
      # source_vecs_p, source_contexts_p, source_padding_p, query_vec_p are used
      # for both TensorFlow and numpy computation.
      source_vecs_p = [np.random.rand(3, 4) for _ in range(6)]
      source_vecs = tf.stack(
          [tf.constant(x, dtype=tf.float32) for x in source_vecs_p])
      source_contexts_p = [np.random.rand(3, 5) for _ in range(6)]
      source_contexts = tf.stack(
          [tf.constant(x, dtype=tf.float32) for x in source_contexts_p])
      source_padding_p = [[0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 1, 0],
                          [0, 0, 1, 0, 1, 0]]
      source_padding = tf.transpose(
          tf.constant(source_padding_p, dtype=tf.float32))
      query_vec_p = np.random.rand(6, 4)
      query_vec = tf.constant(query_vec_p, dtype=tf.float32)
      query_segment_id_p = [0, 1, 1, 1, 2, 2]
      source_segment_id_p = [[0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 2, 2],
                             [0, 1, 1, 1, 1, 2]]
      source_segment_id = None
      query_segment_id = None
      if packed_inputs:
        source_segment_id = tf.transpose(
            tf.constant(source_segment_id_p, dtype=tf.float32))
        query_segment_id = tf.constant(query_segment_id_p, dtype=tf.float32)
      params = attention.DotProductAttention.Params()
      params.name = 'dotproduct_atten'
      params.source_dim = 4
      params.query_dim = 4
      params.hidden_dim = 4
      params.packed_input = packed_inputs
      atten = attention.DotProductAttention(params)
      atten.InitForSourcePacked(atten.theta, source_vecs, source_contexts,
                                source_padding, source_segment_id)
      tf.global_variables_initializer().run()
      atten_vec, atten_prob, _ = atten.ComputeContextVector(
          atten.theta, query_vec, query_segment_id=query_segment_id)
      atten_vec_out, prob_out = sess.run([atten_vec, atten_prob])
      # Use numpy to perform the same computation to generate expected results.
      source_vecs_p = np.array(source_vecs_p)
      # Dot-product part.
      expected_logit = np.array([
          np.dot(source_vecs_p[:, i % 3, :], query_vec_p[i, :])
          for i in range(6)
      ]) / math.sqrt(4)
      elexp = np.exp(expected_logit)
      source_padding_p = np.array(source_padding_p)
      elexp *= (1 - np.tile(source_padding_p, (2, 1)))
      if packed_inputs:
        # Manually constructed packed input mask.
        mask = np.asarray([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0],
                           [0, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 1]])
        elexp *= mask
      expected_prob_out = elexp / np.expand_dims(np.sum(elexp, axis=1), axis=1)
      expanded_epo = np.expand_dims(expected_prob_out, axis=2)
      source_contexts_p = np.array(source_contexts_p)
      expected_atten_vec_out = np.array([
          np.sum(
              source_contexts_p[:, i % 3, :] * expanded_epo[i, :, :], axis=0)
          for i in range(6)
      ])

      print(['additive attention prob_out', np.array_repr(prob_out)])
      print(['additive attention atten_vec_out', np.array_repr(atten_vec_out)])

      self.assertAllClose(expected_prob_out, prob_out)
      self.assertAllClose(expected_atten_vec_out, atten_vec_out)

  def _MultiHeadedAttentionInputs(self, source_dim=4, dtype=tf.float32):
    np.random.seed(6348575)
    # source_vecs_p, source_contexts_p, source_padding_p, query_vec_p are used
    # for both TensorFlow and numpy computation.
    source_vecs_p = [np.random.rand(3, source_dim) for _ in range(6)]
    source_vecs = tf.stack([tf.constant(x, dtype=dtype) for x in source_vecs_p])
    source_contexts_p = [np.random.rand(3, 6) for _ in range(6)]
    source_contexts = tf.stack(
        [tf.constant(x, dtype=dtype) for x in source_contexts_p])
    source_padding_p = [[0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 1, 0]]
    source_padding = tf.transpose(tf.constant(source_padding_p, dtype=dtype))
    query_vec_p = np.random.rand(6, 4)
    query_vec = tf.constant(query_vec_p, dtype=dtype)
    query_segment_id_p = [0, 1, 1, 1, 2, 2]
    source_segment_id_p = [[0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 2, 2],
                           [0, 1, 1, 1, 1, 2]]
    source_segment_id = tf.transpose(
        tf.constant(source_segment_id_p, dtype=dtype))
    query_segment_id = tf.constant(query_segment_id_p, dtype=dtype)
    return (source_vecs, source_contexts, source_padding, source_padding_p,
            query_vec, source_segment_id, query_segment_id)

  def testMultiHeadedAttentionDotProductWithFeedinProbs(self):
    with self.session(use_gpu=True) as sess:
      (source_vecs, source_contexts, source_padding, _, _, _,
       _) = self._MultiHeadedAttentionInputs()
      iap = attention.DotProductAttention.Params()
      iap.name = 'dot_atten'
      params = attention.MultiHeadedAttention.Params().Set(
          name='multihead_atten',
          source_dim=4,
          query_dim=4,
          hidden_dim=4,
          inner_atten_params=iap,
          num_attention_heads=2,
          use_source_vec_as_attention_value=False)
      atten = params.cls(params)
      packed_src = atten.InitForSourcePacked(atten.theta, source_vecs,
                                             source_contexts, source_padding)
      tf.global_variables_initializer().run()
      atten_probs = tf.constant([[1.0] + [0.0] * 5] * 3 * 2, dtype=tf.float32)
      atten_vec_proj, atten_vec = atten.ComputeContextVectorWithAttenProbs(
          atten.theta, packed_src.source_contexts, atten_probs)
      atten_vec_proj, atten_vec, packed_context = sess.run(
          [atten_vec_proj, atten_vec, packed_src.source_contexts])
      self.assertAllClose(
          atten_vec,
          np.reshape(np.transpose(packed_context, (0, 2, 1)),
                     [3, 6, 6])[:, :, 0])
      self.assertAllClose([2.5694468, 4.36386967, 3.24537992],
                          np.sum(atten_vec_proj, axis=1))

  def _testMultiHeadedAttentionExtendCachedSourceVecsHelper(
      self, additive_atten, dtype, fprop_dtype):
    # source_batch:3, target_batch:6. Test n = 2 case.
    use_gpu = (dtype == tf.float32 and fprop_dtype == tf.float32)
    with self.session(use_gpu=use_gpu) as sess:
      (source_vecs, source_contexts, source_padding, _, query_vec,
       source_seg_id,
       query_seg_id) = self._MultiHeadedAttentionInputs(dtype=fprop_dtype)
      if additive_atten:
        iap = attention.AdditiveAttention.Params()
        iap.name = 'add_atten'
      else:
        iap = attention.DotProductAttention.Params()
        iap.name = 'dot_atten'
      params = attention.MultiHeadedAttention.Params().Set(
          name='multihead_atten',
          dtype=dtype,
          fprop_dtype=fprop_dtype,
          source_dim=4,
          query_dim=4,
          hidden_dim=4,
          inner_atten_params=iap,
          num_attention_heads=2,
          use_source_vec_as_attention_value=False,
          packed_input=True)
      atten = params.cls(params)
      theta = atten.theta
      packed_src1 = atten.InitForSourcePacked(
          theta, source_vecs, source_contexts, source_padding, source_seg_id)
      cached_src = py_utils.NestedMap(
          source_vecs=tf.zeros([0, 3, 4], dtype=packed_src1.source_vecs.dtype),
          source_contexts=tf.zeros([0, 3, 6],
                                   dtype=packed_src1.source_contexts.dtype),
          source_padding=tf.zeros([0, 3, 2],
                                  dtype=packed_src1.source_padding.dtype),
          source_segment_id=tf.zeros([0, 3, 2],
                                     dtype=packed_src1.source_segment_id.dtype))
      for i in range(6):
        cached_src = atten.ExtendSourcePacked(
            theta, source_vecs[i, :, :], source_contexts[i, :, :],
            source_padding[i, :], source_seg_id[i, :], cached_src)
      packed_src2 = atten.PackCachedSource(cached_src)
      tf.global_variables_initializer().run()

      atten_vec_1, prob_1, _ = atten.ComputeContextVectorWithSource(
          theta, packed_src1, query_vec, query_segment_id=query_seg_id)
      atten_vec_2, prob_2, _ = atten.ComputeContextVectorWithCachedSource(
          theta, cached_src, query_vec, query_segment_id=query_seg_id)

      packed_src1_v, packed_src2_v, cached_src_v = sess.run(
          [packed_src1, packed_src2, cached_src])
      tf.logging.info('packed_src1=%s', packed_src1_v)
      tf.logging.info('packed_src2=%s', packed_src2_v)
      tf.logging.info('cached_src=%s', cached_src_v)
      self.assertAllClose(packed_src1_v.source_vecs, packed_src2_v.source_vecs)
      self.assertAllClose(packed_src1_v.source_contexts,
                          packed_src2_v.source_contexts)
      self.assertAllClose(packed_src1_v.source_padding,
                          packed_src2_v.source_padding)
      self.assertAllClose(packed_src1_v.source_segment_id,
                          packed_src2_v.source_segment_id)
      atten_vec1_v, prob1_v, atten_vec2_v, prob2_v = sess.run(
          [atten_vec_1, prob_1, atten_vec_2, prob_2])
      self.assertAllClose(prob1_v, prob2_v)
      self.assertAllClose(atten_vec1_v, atten_vec2_v)

  def testMultiHeadedAttentionExtendCachedSourceVecsAdditiveFloat32(self):
    self._testMultiHeadedAttentionExtendCachedSourceVecsHelper(
        additive_atten=True, dtype=tf.float32, fprop_dtype=tf.float32)

  def testMultiHeadedAttentionExtendCachedSourceVecsAdditiveFloat32Float16(
      self):
    self._testMultiHeadedAttentionExtendCachedSourceVecsHelper(
        additive_atten=True, dtype=tf.float32, fprop_dtype=tf.float16)

  def testMultiHeadedAttentionExtendCachedSourceVecsDotFloat32(self):
    self._testMultiHeadedAttentionExtendCachedSourceVecsHelper(
        additive_atten=False, dtype=tf.float32, fprop_dtype=tf.float32)

  def testMultiHeadedAttentionExtendCachedSourceVecsDotFloat32Float16(self):
    self._testMultiHeadedAttentionExtendCachedSourceVecsHelper(
        additive_atten=False, dtype=tf.float32, fprop_dtype=tf.float16)

  def _testMultiHeadedAttentionExtendCachedSourceVecsNoPaddingsHelper(
      self, additive_attention=False):
    # source_batch:3, target_batch:6. Test n = 2 case.
    with self.session(use_gpu=True) as sess:
      (source_vecs, source_contexts, _, _, query_vec, _,
       _) = self._MultiHeadedAttentionInputs()
      source_padding = tf.zeros([6, 3])
      if additive_attention:
        iap = attention.AdditiveAttention.Params()
        iap.name = 'add_atten'
      else:
        iap = attention.DotProductAttention.Params()
        iap.name = 'dot_atten'
      params = attention.MultiHeadedAttention.Params().Set(
          name='multihead_atten',
          source_dim=4,
          query_dim=4,
          hidden_dim=4,
          inner_atten_params=iap,
          num_attention_heads=2,
          use_source_vec_as_attention_value=False)
      atten = params.cls(params)
      packed_src1 = atten.InitForSourcePacked(atten.theta, source_vecs,
                                              source_contexts, source_padding)
      cached_src = py_utils.NestedMap(
          source_vecs=tf.zeros([0, 3, 4], dtype=packed_src1.source_vecs.dtype),
          source_contexts=tf.zeros([0, 3, 6],
                                   dtype=packed_src1.source_contexts.dtype),
          source_padding=None,
          source_seg_id=None)
      for i in range(6):
        cached_src = atten.ExtendSourcePacked(atten.theta, source_vecs[i, :, :],
                                              source_contexts[i, :, :], None,
                                              None, cached_src)
      packed_src2 = atten.PackCachedSource(cached_src)
      tf.global_variables_initializer().run()

      atten_vec_1, prob_1, _ = atten.ComputeContextVectorWithSource(
          atten.theta, packed_src1, query_vec)
      atten_vec_2, prob_2, _ = atten.ComputeContextVectorWithCachedSource(
          atten.theta, cached_src, query_vec)

      (source_vec1_v, source_context1_v, source_vec2_v, source_context2_v,
       atten_vec1_v, prob1_v, atten_vec2_v, prob2_v) = sess.run([
           packed_src1.source_vecs, packed_src1.source_contexts,
           packed_src2.source_vecs, packed_src2.source_contexts, atten_vec_1,
           prob_1, atten_vec_2, prob_2
       ])
      self.assertAllClose(source_vec1_v, source_vec2_v)
      self.assertAllClose(source_context1_v, source_context2_v)
      self.assertAllClose(atten_vec1_v, atten_vec2_v)
      self.assertAllClose(prob1_v, prob2_v)

  def testMultiHeadedDotAttentionExtendCachedSourceVecsNoPaddings(self):
    self._testMultiHeadedAttentionExtendCachedSourceVecsNoPaddingsHelper(False)

  def testMultiHeadedAddAttentionExtendCachedSourceVecsNoPaddings(self):
    self._testMultiHeadedAttentionExtendCachedSourceVecsNoPaddingsHelper(True)

  def testMultiHeadedAttentionDotProduct(self):
    # source_batch:3, target_batch:6. Test n = 2 case.
    with self.session(use_gpu=True) as sess:
      (source_vecs, source_contexts, source_padding, source_padding_p,
       query_vec, _, _) = self._MultiHeadedAttentionInputs()
      iap = attention.DotProductAttention.Params()
      iap.name = 'dot_atten'
      params = attention.MultiHeadedAttention.Params().Set(
          name='multihead_atten',
          source_dim=4,
          query_dim=4,
          hidden_dim=4,
          inner_atten_params=iap,
          num_attention_heads=2,
          use_source_vec_as_attention_value=False)
      atten = params.cls(params)
      atten.InitForSourcePacked(atten.theta, source_vecs, source_contexts,
                                source_padding)
      tf.global_variables_initializer().run()
      atten_vec, atten_prob, _ = atten.ComputeContextVector(
          atten.theta, query_vec)

      self._CheckStaticShapes(
          atten_vec,
          atten_prob,
          target_batch_size=query_vec.shape[0],
          source_length=source_contexts.shape[0],
          context_dim=source_contexts.shape[2])

      atten_vec_out, prob_out = sess.run([atten_vec, atten_prob])
      print('atten_vec_out', np.sum(atten_vec_out, axis=1))
      self.assertAllClose([
          2.84679317, 2.36924601, 3.54831171, 2.86487937, 2.3537426, 3.54308939
      ], np.sum(atten_vec_out, axis=1))
      print('atten_vec_out', atten_vec_out)
      print('prob_out', prob_out)
      t_batch_size = 6
      s_batch_size = 3
      for i in range(t_batch_size):
        # Test to make sure we didn't mess up indexing.
        s_index = i % s_batch_size
        atten.InitForSourcePacked(atten.theta,
                                  source_vecs[:, s_index:s_index + 1, :],
                                  source_contexts[:, s_index:s_index + 1, :],
                                  source_padding[:, s_index:s_index + 1])
        atten_vec_i, prob_i, _ = atten.ComputeContextVector(
            atten.theta, query_vec[i:i + 1])
        atten_vec_i_out, prob_i_out = sess.run([atten_vec_i, prob_i])
        self.assertAllClose(prob_i_out, prob_out[i:i + 1])
        self.assertAllClose(atten_vec_i_out, atten_vec_out[i:i + 1])
        padding_i = source_padding_p[s_index]
        # Check to make sure prob exists only on valid timesteps.
        self.assertEqual(0.0, np.sum(padding_i * prob_i_out))

  def testMultiHeadedAttentionDotProductPackedInput(self):
    # source_batch:3, target_batch:6. Test n = 2 case.
    with self.session(use_gpu=True) as sess:
      (source_vecs, source_contexts, source_padding, source_padding_p,
       query_vec, source_seg_id,
       query_seg_id) = self._MultiHeadedAttentionInputs()
      iap = attention.DotProductAttention.Params()
      iap.name = 'dot_atten'
      params = attention.MultiHeadedAttention.Params().Set(
          name='multihead_atten',
          source_dim=4,
          query_dim=4,
          hidden_dim=4,
          inner_atten_params=iap,
          num_attention_heads=2,
          use_source_vec_as_attention_value=False,
          packed_input=True)
      atten = params.cls(params)
      atten.InitForSourcePacked(atten.theta, source_vecs, source_contexts,
                                source_padding, source_seg_id)
      tf.global_variables_initializer().run()
      atten_vec, atten_prob, _ = atten.ComputeContextVector(
          atten.theta, query_vec, query_segment_id=query_seg_id)
      atten_vec_out, prob_out = sess.run([atten_vec, atten_prob])
      print('atten_vec_out', np.sum(atten_vec_out, axis=1))
      self.assertAllClose(
          [2.565648, 2.268182, 3.739031, 3.093884, 2.770367, 3.580353],
          np.sum(atten_vec_out, axis=1))
      print('atten_vec_out', atten_vec_out)
      print('prob_out', prob_out)
      t_batch_size = 6
      s_batch_size = 3
      for i in range(t_batch_size):
        # Test to make sure we didn't mess up indexing.
        s_index = i % s_batch_size
        src_seg_id = source_seg_id[:, s_index:s_index + 1]
        atten.InitForSourcePacked(
            atten.theta, source_vecs[:, s_index:s_index + 1, :],
            source_contexts[:, s_index:s_index + 1, :],
            source_padding[:, s_index:s_index + 1], src_seg_id)
        qry_seg_id = query_seg_id[i:i + 1]
        atten_vec_i, prob_i, _ = atten.ComputeContextVector(
            atten.theta, query_vec[i:i + 1], query_segment_id=qry_seg_id)
        atten_vec_i_out, prob_i_out = sess.run([atten_vec_i, prob_i])
        self.assertAllClose(prob_i_out, prob_out[i:i + 1])
        self.assertAllClose(atten_vec_i_out, atten_vec_out[i:i + 1])
        padding_i = source_padding_p[s_index]
        # Check to make sure prob exists only on valid timesteps.
        self.assertEqual(0.0, np.sum(padding_i * prob_i_out))

  def testMultiHeadedAttentionDotProductDeterministicDropout(self):
    # source_batch:3, target_batch:6. Test n = 2 case.
    with self.session(use_gpu=True) as sess:
      source_vecs, source_contexts, source_padding, _, query_vec, _, _ = (
          self._MultiHeadedAttentionInputs())
      iap = attention.DotProductAttention.Params()
      iap.name = 'dot_atten'
      params = attention.MultiHeadedAttention.Params().Set(
          name='multihead_atten',
          source_dim=4,
          query_dim=4,
          hidden_dim=4,
          inner_atten_params=iap,
          num_attention_heads=2,
          atten_dropout_prob=0.5,
          atten_dropout_deterministic=True,
          random_seed=7249528,
          use_source_vec_as_attention_value=False)
      atten = params.cls(params)
      atten.InitForSourcePacked(atten.theta, source_vecs, source_contexts,
                                source_padding)
      atten_state = atten.ZeroAttentionState(2, 6)
      print('atten_state:', atten_state)

      atten_vec, atten_prob, atten_state = atten.ComputeContextVector(
          atten.theta, query_vec, atten_state)

      tf.global_variables_initializer().run()
      atten_vec_out, prob_out = sess.run([atten_vec, atten_prob])

      print('atten_vec_out', np.sum(atten_vec_out, axis=1))
      self.assertAllClose(
          [4.400547, 3.79999, 4.441541, 4.13225, 1.798602, 3.380396],
          np.sum(atten_vec_out, axis=1))

      print('atten_vec_out %r' % atten_vec_out)
      print('prob_out %r' % prob_out)

      expected_prob_out = [
          [0.20665196, 0.27156532, 0., 0., 0.26471972, 0.257063],
          [0., 0.23842147, 0.22581396, 0.25578159, 0., 0.27998304],
          [0.320409, 0.19248575, 0., 0.20368136, 0., 0.2834239],
          [0.20825484, 0.27220166, 0., 0., 0.2568481, 0.26269543],
          [0., 0.23891166, 0.24187937, 0.24953011, 0., 0.26967883],
          [0.32084343, 0.19087103, 0., 0.2172718, 0., 0.27101377],
      ]
      expected_atten_vec_out = [
          [
              0.96826828, 0.76794815, 0.96536565, 0.67366356, 0.66884744,
              0.35645404
          ],
          [
              0.44098836, 0.76838356, 1.28262615, 0.44513249, 0.2947804,
              0.56807894
          ],
          [
              0.67532849, 1.19130635, 0.98330915, 0.5132336, 0.75915694,
              0.31920666
          ],
          [
              0.84831893, 0.82190067, 0.84433675, 0.70621985, 0.54718214,
              0.36429209
          ],
          [
              0.15644459, 0.02022849, 0.06057758, 0.5914318, 0.25838101,
              0.71153873
          ],
          [
              0.43432298, 1.0173521, 0.94046545, 0.43881211, 0.28064111,
              0.26880282
          ],
      ]
      self.assertAllClose(expected_prob_out, prob_out)
      self.assertAllClose(expected_atten_vec_out, atten_vec_out)

  def testMultiHeadedAttentionMonotonic(self):
    # source_batch:3, target_batch:6. Test n = 2 case.
    with self.session(use_gpu=True) as sess:
      (source_vecs, source_contexts, source_padding, source_padding_p,
       query_vec, _, _) = self._MultiHeadedAttentionInputs()
      iap = attention.MonotonicAttention.Params()
      iap.name = 'mono_atten'
      params = attention.MultiHeadedAttention.Params().Set(
          name='multihead_atten',
          source_dim=4,
          query_dim=4,
          hidden_dim=4,
          inner_atten_params=iap,
          num_attention_heads=2,
          use_source_vec_as_attention_value=False)
      atten = params.cls(params)
      atten.InitForSourcePacked(atten.theta, source_vecs, source_contexts,
                                source_padding)
      # [batch * 2 heads, time]
      atten_init_state = self._attentionStateWithRandomEmitProbabilities(
          atten, 12, 6)
      tf.global_variables_initializer().run()
      atten_vec, atten_prob, _ = atten.ComputeContextVector(
          atten.theta, query_vec, atten_init_state)
      atten_vec_out, prob_out = sess.run([atten_vec, atten_prob])
      print('atten_vec_out', np.sum(atten_vec_out, axis=1))
      self.assertAllClose(
          [1.494033, 1.120422, 1.699309, 1.508609, 1.1329, 1.670303],
          np.sum(atten_vec_out, axis=1))
      print('atten_vec_out', atten_vec_out)
      print('prob_out', prob_out)
      t_batch_size = 6
      s_batch_size = 3
      for i in range(t_batch_size):
        # Test to make sure we didn't mess up indexing.
        s_index = i % s_batch_size
        atten.InitForSourcePacked(atten.theta,
                                  source_vecs[:, s_index:s_index + 1, :],
                                  source_contexts[:, s_index:s_index + 1, :],
                                  source_padding[:, s_index:s_index + 1])
        j = i * 2
        sliced_atten_state = py_utils.NestedMap(
            emit_probs=atten_init_state.emit_probs[j:j + 2])
        atten_vec_i, prob_i, _ = atten.ComputeContextVector(
            atten.theta, query_vec[i:i + 1], sliced_atten_state)
        atten_vec_i_out, prob_i_out = sess.run([atten_vec_i, prob_i])
        self.assertAllClose(prob_i_out, prob_out[i:i + 1])
        self.assertAllClose(atten_vec_i_out, atten_vec_out[i:i + 1])
        padding_i = source_padding_p[s_index]
        # Check to make sure prob exists only on valid timesteps.
        self.assertEqual(0.0, np.sum(padding_i * prob_i_out))

  def testMultiHeadedAttentionDotProductWithAllProj(self):
    # source_batch:3, target_batch:6. Test n = 2 case.
    with self.session(use_gpu=True) as sess:
      (source_vecs, source_contexts, source_padding, source_padding_p,
       query_vec, _, _) = self._MultiHeadedAttentionInputs()
      iap = attention.DotProductAttention.Params()
      iap.name = 'dot_atten'
      params = attention.MultiHeadedAttention.Params().Set(
          name='multihead_atten',
          source_dim=4,
          query_dim=4,
          hidden_dim=4,
          inner_atten_params=iap,
          num_attention_heads=2,
          use_source_vec_as_attention_value=False,
          enable_ctx_pre_proj=True,
          enable_ctx_post_proj=True,
          ctx_post_proj_dim=5,
          context_dim=6)
      atten = params.cls(params)
      atten.InitForSourcePacked(atten.theta, source_vecs, source_contexts,
                                source_padding)
      tf.global_variables_initializer().run()
      atten_vec, atten_prob, _ = atten.ComputeContextVector(
          atten.theta, query_vec)

      self._CheckStaticShapes(
          atten_vec,
          atten_prob,
          target_batch_size=query_vec.shape[0],
          source_length=source_contexts.shape[0],
          context_dim=params.ctx_post_proj_dim)

      atten_vec_out, prob_out = sess.run([atten_vec, atten_prob])
      print('atten_vec_out', np.sum(atten_vec_out, axis=1))
      self.assertAllClose([
          1.356745, 0.65274805, 1.39460433, 1.34961343, 0.63025361, 1.41543126
      ], np.sum(atten_vec_out, axis=1))
      print('atten_vec_out', atten_vec_out)
      print('prob_out', prob_out)
      t_batch_size = 6
      s_batch_size = 3
      for i in range(t_batch_size):
        # Test to make sure we didn't mess up indexing.
        s_index = i % s_batch_size
        atten.InitForSourcePacked(atten.theta,
                                  source_vecs[:, s_index:s_index + 1, :],
                                  source_contexts[:, s_index:s_index + 1, :],
                                  source_padding[:, s_index:s_index + 1])
        atten_vec_i, prob_i, _ = atten.ComputeContextVector(
            atten.theta, query_vec[i:i + 1])
        atten_vec_i_out, prob_i_out = sess.run([atten_vec_i, prob_i])
        self.assertAllClose(prob_i_out, prob_out[i:i + 1])
        self.assertAllClose(atten_vec_i_out, atten_vec_out[i:i + 1])
        padding_i = source_padding_p[s_index]
        # Check to make sure prob exists only on valid timesteps.
        self.assertEqual(0.0, np.sum(padding_i * prob_i_out))

  def _testMultiHeadedAttentionAdditiveHelper(self,
                                              source_dim,
                                              expected_vec,
                                              packed_input=False):
    # source_batch:3, target_batch:6. Test n = 2 case.
    with self.session(use_gpu=True) as sess:
      (source_vecs, source_contexts, source_padding, source_padding_p,
       query_vec, source_seg_id,
       query_seg_id) = self._MultiHeadedAttentionInputs(source_dim)
      if not packed_input:
        source_seg_id = None
        query_seg_id = None
      iap = attention.AdditiveAttention.Params()
      iap.name = 'add_atten'
      params = attention.MultiHeadedAttention.Params().Set(
          name='multihead_atten',
          source_dim=source_dim,
          query_dim=4,
          hidden_dim=4,
          num_attention_heads=2,
          inner_atten_params=iap,
          use_source_vec_as_attention_value=False,
          vn=py_utils.VariationalNoiseParams(0.0, False, False),
          packed_input=packed_input)
      atten = params.cls(params)
      atten.InitForSourcePacked(atten.theta, source_vecs, source_contexts,
                                source_padding, source_seg_id)
      atten_vec, atten_prob, _ = atten.ComputeContextVector(
          atten.theta, query_vec, query_segment_id=query_seg_id)

      self._CheckStaticShapes(
          atten_vec,
          atten_prob,
          target_batch_size=query_vec.shape[0],
          source_length=source_contexts.shape[0],
          context_dim=source_contexts.shape[2])

      tf.global_variables_initializer().run()
      atten_vec_out, prob_out = sess.run([atten_vec, atten_prob])
      print('atten_vec_out', np.sum(atten_vec_out, axis=1))
      self.assertAllClose(expected_vec, np.sum(atten_vec_out, axis=1))
      print('atten_vec_out', atten_vec_out)
      print('prob_out', prob_out)
      t_batch_size = 6
      s_batch_size = 3
      for i in range(t_batch_size):
        # Test to make sure we didn't mess up indexing.
        s_index = i % s_batch_size
        src_seg_id = None
        if packed_input:
          src_seg_id = source_seg_id[:, s_index:s_index + 1]
        atten.InitForSourcePacked(
            atten.theta, source_vecs[:, s_index:s_index + 1, :],
            source_contexts[:, s_index:s_index + 1, :],
            source_padding[:, s_index:s_index + 1], src_seg_id)
        qry_seg_id = None
        if packed_input:
          qry_seg_id = query_seg_id[i:i + 1]
        atten_vec_i, prob_i, _ = atten.ComputeContextVector(
            atten.theta, query_vec[i:i + 1], query_segment_id=qry_seg_id)
        atten_vec_i_out, prob_i_out = sess.run([atten_vec_i, prob_i])
        self.assertAllClose(prob_i_out, prob_out[i:i + 1])
        self.assertAllClose(atten_vec_i_out, atten_vec_out[i:i + 1])
        padding_i = source_padding_p[s_index]
        # Check to make sure prob exists only on valid timesteps.
        self.assertEqual(0.0, np.sum(padding_i * prob_i_out))

  def testMultiHeadedAttentionAdditive(self):
    self._testMultiHeadedAttentionAdditiveHelper(
        4, [2.858081, 2.33295, 3.529434, 2.856466, 2.342262, 3.526487])

  def testMultiHeadedAttentionAdditivePackedInput(self):
    self._testMultiHeadedAttentionAdditiveHelper(
        4, [2.585192, 2.267683, 3.708972, 3.107646, 2.770367, 3.580353],
        packed_input=True)

  def testMultiHeadedAttentionAdditiveUnequalDim(self):
    self._testMultiHeadedAttentionAdditiveHelper(
        14, [3.189594, 2.462574, 2.912001, 3.19924, 2.462459, 2.909231])

  def testLocationSensitiveAttention1(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(12345)
      source_vecs = tf.stack([
          tf.constant(np.random.rand(3, 4), dtype=tf.float32) for _ in range(6)
      ])
      source_contexts = tf.stack([
          tf.constant(np.random.rand(3, 5), dtype=tf.float32) for _ in range(6)
      ])
      source_padding = tf.transpose(
          tf.constant(
              [[0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0]],
              dtype=tf.float32))
      query_vec = tf.constant(np.random.rand(6, 7), dtype=tf.float32)

      params = attention.LocationSensitiveAttention.Params()
      params.name = 'loc_atten'
      params.params_init = py_utils.WeightInit.Gaussian(0.1, 12345)
      params.source_dim = 4
      params.query_dim = 7
      params.hidden_dim = 7
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      params.location_filter_size = 3
      params.location_num_filters = 4

      atten = attention.LocationSensitiveAttention(params)
      atten.InitForSourcePacked(atten.theta, source_vecs, source_contexts,
                                source_padding)

      atten_init_state = tf.nn.softmax(
          tf.constant(
              np.random.rand(6, len(params.location_features), 6),
              dtype=tf.float32))

      atten_vec, atten_prob, atten_state = atten.ComputeContextVector(
          atten.theta, query_vec, atten_init_state)

      atten_vars = tf.get_collection('LocationSensitiveAttention_vars')
      self.assertEqual(5, len(atten_vars))

      tf.global_variables_initializer().run()

      atten_vec_out, prob_out, atten_init_state_out, atten_state_out = sess.run(
          [atten_vec, atten_prob, atten_init_state, atten_state])

      self.assertEqual(atten_init_state_out.shape, atten_state_out.shape)

      print(['additive attention prob_out', np.array_repr(prob_out)])
      print(['additive attention atten_vec_out', np.array_repr(atten_vec_out)])

      # pyformat: disable
      # pylint: disable=bad-whitespace
      expected_prob_out = [
          [ 0.25557119,  0.2407331 ,  0.        ,  0.        ,  0.25413439,
            0.24956135],
          [ 0.        ,  0.2539435 ,  0.24765202,  0.25480285,  0.        ,
            0.24360162],
          [ 0.25094694,  0.25000173,  0.        ,  0.24308425,  0.        ,
            0.25596702],
          [ 0.25559491,  0.24071115,  0.        ,  0.        ,  0.2541317 ,
            0.24956223],
          [ 0.        ,  0.25393987,  0.24765508,  0.25481141,  0.        ,
            0.24359357],
          [ 0.25112614,  0.24990462,  0.        ,  0.24246819,  0.        ,
            0.25650105]]
      expected_atten_vec_out = [
          [ 0.49745601,  0.63471878,  0.49220741,  0.56829882,  0.42753279],
          [ 0.51502693,  0.56184328,  0.37644374,  0.87425017,  0.46182287],
          [ 0.57862061,  0.44247472,  0.36931327,  0.41002682,  0.14327496],
          [ 0.49745524,  0.63471991,  0.49221092,  0.56828701,  0.427522  ],
          [ 0.51502484,  0.5618462 ,  0.37644884,  0.87424958,  0.46181911],
          [ 0.57893252,  0.44248456,  0.36938512,  0.4100675 ,  0.14328022]]
      # pyformat: enable
      # pylint: enable=bad-whitespace

      self.assertAllClose(expected_prob_out, prob_out)
      self.assertAllClose(expected_atten_vec_out, atten_vec_out)

  def testLocationSensitiveAttention2(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(12345)
      source_vecs = tf.stack([
          tf.constant(np.random.rand(3, 4), dtype=tf.float32) for _ in range(6)
      ])
      source_contexts = tf.stack([
          tf.constant(np.random.rand(3, 5), dtype=tf.float32) for _ in range(6)
      ])
      source_padding = tf.transpose(
          tf.constant(
              [[0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0]],
              dtype=tf.float32))
      query_vec = tf.constant(np.random.rand(6, 7), dtype=tf.float32)

      params = attention.LocationSensitiveAttention.Params()
      params.name = 'loc_atten'
      params.params_init = py_utils.WeightInit.Gaussian(0.1, 12345)
      params.source_dim = 4
      params.query_dim = 7
      params.hidden_dim = 7
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      params.location_filter_size = 3
      params.location_num_filters = 4
      params.location_features = ['PREV_PROBS', 'CUMULATIVE_PROBS']

      atten = attention.LocationSensitiveAttention(params)
      atten.InitForSourcePacked(atten.theta, source_vecs, source_contexts,
                                source_padding)

      atten_init_state = atten.ZeroAttentionState(tf.shape(source_vecs)[0], 6)

      (unused_atten_vec,
       unused_atten_prob, atten_state) = atten.ComputeContextVector(
           atten.theta, query_vec, atten_init_state)

      atten_vars = tf.get_collection('LocationSensitiveAttention_vars')
      self.assertEqual(5, len(atten_vars))

      tf.global_variables_initializer().run()

      atten_init_state_out, atten_state_out = sess.run(
          [atten_init_state, atten_state])

      self.assertEqual(atten_init_state_out.shape, atten_state_out.shape)

  def _testLocationSensitiveAttentionSameBatchSizeHelper(
      self, same_batch_size, quantized=False):
    with self.session(tf.Graph(), use_gpu=True) as sess:
      np.random.seed(12345)
      dtype = tf.float32 if quantized else tf.float64
      source_vecs = tf.constant(np.random.rand(6, 3, 4), dtype=dtype)
      source_contexts = tf.constant(np.random.rand(6, 3, 5), dtype=dtype)
      source_padding = tf.transpose(
          tf.constant(
              [[0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0]],
              dtype=dtype))

      query_vec = tf.constant(np.random.rand(3, 7), dtype=dtype)

      params = attention.LocationSensitiveAttention.Params()
      params.dtype = dtype
      params.name = 'loc_atten'
      params.params_init = py_utils.WeightInit.Gaussian(0.1, 12345)
      params.source_dim = 4
      params.query_dim = 7
      params.hidden_dim = 7
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      params.location_filter_size = 3
      params.location_num_filters = 4
      params.same_batch_size = same_batch_size

      if quantized:
        cc_schedule = quant_utils.FakeQuantizationSchedule.Params().Set(
            clip_start_step=0,
            clip_end_step=13000,
            quant_start_step=14000,
            start_cap=8.0,
            end_cap=1.0)
        qdomain = quant_utils.SymmetricScheduledClipQDomain.Params().Set(
            cc_schedule=cc_schedule.Copy())
        params.qdomain.default = qdomain.Copy()
        params.qdomain.atten_context = qdomain.Copy()

      atten = attention.LocationSensitiveAttention(params)
      atten.InitForSourcePacked(atten.theta, source_vecs, source_contexts,
                                source_padding)

      atten_init_state = tf.nn.softmax(
          tf.constant(
              np.random.rand(3, len(params.location_features), 6), dtype=dtype))

      atten_vec, atten_prob, atten_state = atten.ComputeContextVector(
          atten.theta, query_vec, atten_init_state)

      atten_vars = tf.get_collection('LocationSensitiveAttention_vars')
      self.assertEqual(5, len(atten_vars))

      tf.global_variables_initializer().run()

      atten_vec_out, prob_out, atten_init_state_out, atten_state_out = sess.run(
          [atten_vec, atten_prob, atten_init_state, atten_state])

      self.assertEqual(atten_init_state_out.shape, atten_state_out.shape)
      return atten_vec_out, prob_out, atten_init_state_out, atten_state_out

  def testLocationSensitiveAttentionSameBatchSize(self):
    (atten_vec_out1, prob_out1, atten_init_state_out1, atten_state_out1) = (
        self._testLocationSensitiveAttentionSameBatchSizeHelper(True))
    (atten_vec_out2, prob_out2, atten_init_state_out2, atten_state_out2) = (
        self._testLocationSensitiveAttentionSameBatchSizeHelper(False))
    self.assertAllClose(atten_vec_out1, atten_vec_out2, rtol=1e-04, atol=1e-04)
    self.assertAllClose(prob_out1, prob_out2, rtol=1e-04, atol=1e-04)
    self.assertAllClose(
        atten_init_state_out1, atten_init_state_out2, rtol=1e-04, atol=1e-04)
    self.assertAllClose(
        atten_state_out1, atten_state_out2, rtol=1e-04, atol=1e-04)

  def testLocationSensitiveAttentionQuantized(self):
    (atten_vec_out1, prob_out1, atten_init_state_out1, atten_state_out1) = (
        self._testLocationSensitiveAttentionSameBatchSizeHelper(False, False))
    (atten_vec_out2, prob_out2, atten_init_state_out2, atten_state_out2) = (
        self._testLocationSensitiveAttentionSameBatchSizeHelper(False, True))
    self.assertAllClose(atten_vec_out1, atten_vec_out2, rtol=1e-02, atol=1e-02)
    self.assertAllClose(prob_out1, prob_out2, rtol=1e-02, atol=1e-02)
    self.assertAllClose(
        atten_init_state_out1, atten_init_state_out2, rtol=1e-02, atol=1e-02)
    self.assertAllClose(
        atten_state_out1, atten_state_out2, rtol=1e-02, atol=1e-02)

  def testLocationSensitiveAttentionQuantizedSameBatch(self):
    (atten_vec_out1, prob_out1, atten_init_state_out1, atten_state_out1) = (
        self._testLocationSensitiveAttentionSameBatchSizeHelper(True, False))
    (atten_vec_out2, prob_out2, atten_init_state_out2, atten_state_out2) = (
        self._testLocationSensitiveAttentionSameBatchSizeHelper(True, True))
    self.assertAllClose(atten_vec_out1, atten_vec_out2, rtol=1e-02, atol=1e-02)
    self.assertAllClose(prob_out1, prob_out2, rtol=1e-02, atol=1e-02)
    self.assertAllClose(
        atten_init_state_out1, atten_init_state_out2, rtol=1e-02, atol=1e-02)
    self.assertAllClose(
        atten_state_out1, atten_state_out2, rtol=1e-02, atol=1e-02)

  def _attentionStateWithRandomEmitProbabilities(self,
                                                 atten,
                                                 batch_size,
                                                 time,
                                                 dtype=tf.float32):
    atten_state = atten.ZeroAttentionState(time, batch_size)
    atten_state.emit_probs = tf.nn.softmax(
        tf.constant(np.random.rand(batch_size, time), dtype=dtype))
    return atten_state

  def testMonotonicAttention(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(12345)
      batch_size = 3
      source_dim = 4
      context_dim = 5
      time = 6
      query_dim = 7
      source_vecs = tf.constant(
          np.random.rand(time, batch_size, source_dim), dtype=tf.float32)
      source_contexts = tf.constant(
          np.random.rand(time, batch_size, context_dim), dtype=tf.float32)
      source_padding = tf.transpose(
          tf.constant(
              [[0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0]],
              dtype=tf.float32))
      query_vec = tf.constant(
          np.random.rand(batch_size, query_dim), dtype=tf.float32)

      params = attention.MonotonicAttention.Params()
      params.name = 'monotonic_attention'
      params.params_init = py_utils.WeightInit.Gaussian(0.1, 12345)
      params.source_dim = source_dim
      params.query_dim = query_dim
      params.hidden_dim = query_dim
      params.vn.global_vn = False
      params.vn.per_step_vn = False

      atten = attention.MonotonicAttention(params)
      atten.InitForSourcePacked(atten.theta, source_vecs, source_contexts,
                                source_padding)

      atten_init_state = self._attentionStateWithRandomEmitProbabilities(
          atten, batch_size, time)
      atten_vec, atten_prob, atten_state = atten.ComputeContextVector(
          atten.theta, query_vec, atten_init_state)

      atten_vars = tf.get_collection('MonotonicAttention_vars')
      self.assertEqual(6, len(atten_vars))

      tf.global_variables_initializer().run()

      atten_vec_out, prob_out, atten_init_state_out, atten_state_out = sess.run(
          [atten_vec, atten_prob, atten_init_state, atten_state])

      self.assertEqual(atten_init_state_out.emit_probs.shape,
                       atten_state_out.emit_probs.shape)

      print(['monotonic attention prob_out', np.array_repr(prob_out)])
      print(['monotonic attention atten_vec_out', np.array_repr(atten_vec_out)])

      expected_prob_out = [[
          0.03654566, 0.05925026, 0., 0., 0.20958641, 0.19560105
      ], [0., 0.09670404, 0.13182665, 0.13221622, 0.,
          0.18074416], [0.04112773, 0.07072841, 0., 0.13837409, 0., 0.23935230]]

      expected_atten_vec_out = [[
          0.2937718, 0.30372939, 0.27034321, 0.31328040, 0.19393572
      ], [0.2553753, 0.26388022, 0.20429659, 0.47469878, 0.27512118], [
          0.33394262, 0.1191523, 0.22405925, 0.21366173, 0.03946214
      ]]

      self.assertAllClose(expected_prob_out, prob_out)
      self.assertAllClose(expected_atten_vec_out, atten_vec_out)

  def testMonotonicAttentionHard(self):
    with self.session(use_gpu=True) as sess:
      batch_size = 3
      source_dim = 4
      context_dim = 5
      time = 6
      query_dim = 10
      source_vecs = tf.constant(
          np.random.randn(time, batch_size, source_dim), dtype=tf.float32)
      source_contexts = tf.constant(
          np.random.randn(time, batch_size, context_dim), dtype=tf.float32)
      source_padding = tf.zeros((time, batch_size), dtype=tf.float32)
      query_vec = tf.constant(
          np.random.randn(batch_size, query_dim), dtype=tf.float32)

      params = attention.MonotonicAttention.Params()
      params.name = 'monotonic_attention'
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.source_dim = source_dim
      params.query_dim = query_dim
      params.hidden_dim = query_dim
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      params.hard_sigmoid = True
      # To encourage some probabilities to be > 0
      params.hidden_bias_init = 0.

      atten = attention.MonotonicAttention(params)
      atten.InitForSourcePacked(atten.theta, source_vecs, source_contexts,
                                source_padding)

      atten_init_state = atten.ZeroAttentionState(time, batch_size)

      _, atten_prob, atten_state = atten.ComputeContextVector(
          atten.theta, query_vec, atten_init_state)

      atten_vars = tf.get_collection('MonotonicAttention_vars')
      self.assertEqual(6, len(atten_vars))

      tf.global_variables_initializer().run()

      prob_out, atten_state_out = sess.run([atten_prob, atten_state])
      print(['hard monotonic prob', np.array_repr(prob_out)])
      # Make sure all probabilities are binary
      self.assertTrue(np.all(np.logical_or(prob_out == 0, prob_out == 1)))
      # Make sure either one index was attended or none were
      prob_sum = np.sum(prob_out, 1)
      self.assertTrue(np.all(np.logical_or(prob_sum == 1, prob_sum == 0)))

      query_vec = tf.constant(
          np.random.randn(batch_size, query_dim), dtype=tf.float32)
      # Feed state back in
      _, atten_prob, atten_state = atten.ComputeContextVector(
          atten.theta, query_vec, atten_state_out)
      prob_out2 = sess.run(atten_prob)
      print(['hard monotonic prob2', np.array_repr(prob_out2)])
      # Get indices of where attention was assigned at each output timestep
      idx1 = np.argmax(prob_out, 1)
      idx2 = np.argmax(prob_out2, 1)
      # Either the index must have increased, or all probs were 0
      self.assertTrue(
          np.all(np.logical_or(idx1 <= idx2,
                               np.sum(prob_out2, 1) == 0)))

  def testMonotonicAttentionBackProp(self):
    with self.session(use_gpu=True) as sess:
      # Use float64 dtype for numeric checks
      dtype = tf.float64
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      batch_size = 3
      source_dim = 4
      context_dim = 5
      time = 6
      query_dim = 7
      source_vecs = tf.constant(
          np.random.rand(time, batch_size, source_dim), dtype=tf.float64)
      source_contexts = tf.constant(
          np.random.rand(time, batch_size, context_dim), dtype=tf.float64)
      source_padding = tf.zeros((time, batch_size), dtype=tf.float64)
      query_vec = tf.constant(
          np.random.rand(batch_size, query_dim), dtype=tf.float64)

      params = attention.MonotonicAttention.Params()
      params.name = 'monotonic_attention'
      params.params_init = py_utils.WeightInit.Gaussian(0.1, 12345)
      params.source_dim = source_dim
      params.query_dim = query_dim
      params.hidden_dim = query_dim
      params.dtype = dtype
      params.vn.global_vn = False
      params.vn.per_step_vn = False

      atten = attention.MonotonicAttention(params)
      atten.InitForSourcePacked(atten.theta, source_vecs, source_contexts,
                                source_padding)

      atten_init_state = self._attentionStateWithRandomEmitProbabilities(
          atten, batch_size, time, dtype=dtype)

      atten_vec, _, _ = atten.ComputeContextVector(atten.theta, query_vec,
                                                   atten_init_state)

      loss = tf.reduce_sum(atten_vec)

      all_vars = tf.trainable_variables()
      self.assertEqual(6, len(all_vars))

      grads = tf.gradients(loss, all_vars)
      tf.global_variables_initializer().run()
      sym_grads = [sg.eval() for sg in grads]
      num_grads = [
          test_utils.ComputeNumericGradient(sess, loss, v) for v in all_vars
      ]

      print(sym_grads)
      print(num_grads)

      for sg, ng in zip(sym_grads, num_grads):
        self.assertAllClose(sg, ng, rtol=1e-06, atol=1e-06)

  def _testPerStepSourcePaddingHelper(self, atten, depth=6, atten_state=None):
    with self.session(use_gpu=True) as sess:
      np.random.seed(505837249)
      source_vecs = tf.stack([
          tf.constant(np.random.rand(2, depth), dtype=tf.float32)
          for _ in range(6)
      ])
      source_contexts = tf.stack([
          tf.constant(np.random.rand(2, depth), dtype=tf.float32)
          for _ in range(6)
      ])
      source_padding = tf.transpose(
          tf.constant(
              [[0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 1, 0]], dtype=tf.float32))
      query_vec = tf.constant(np.random.rand(2, depth), dtype=tf.float32)
      query_vec = tf.concat([query_vec, query_vec], 0)

      atten.InitForSourcePacked(atten.theta, source_vecs, source_contexts,
                                source_padding)
      # No per_step_padding.
      atten_vec1, atten_prob1, _ = atten.ComputeContextVector(
          atten.theta,
          query_vec,
          attention_state=atten_state,
          per_step_source_padding=None)
      per_step_padding = tf.constant(
          [[0.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 1.0, 1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
          dtype=tf.float32)
      atten_vec2, atten_prob2, _ = atten.ComputeContextVector(
          atten.theta,
          query_vec,
          attention_state=atten_state,
          per_step_source_padding=per_step_padding)

      tf.global_variables_initializer().run()
      atten_vec1_out, atten_prob1_out = sess.run([atten_vec1, atten_prob1])
      atten_vec2_out, atten_prob2_out = sess.run([atten_vec2, atten_prob2])
      print('atten_prob1_out', atten_prob1_out)
      print('atten_prob2_out', atten_prob2_out)
      print('atten_vec1_out', atten_vec1_out)
      print('atten_vec2_out', atten_vec2_out)
      self.assertAllClose(atten_prob1_out[:2], atten_prob1_out[2:])
      self.assertAllClose(atten_vec1_out[:2], atten_vec1_out[2:])
      self.assertAllClose(atten_prob1_out[1], atten_prob2_out[1])
      self.assertAllClose(atten_vec1_out[1], atten_vec2_out[1])
      self.assertAllClose(atten_prob1_out[3], atten_prob2_out[3])
      self.assertAllClose(atten_vec1_out[3], atten_vec2_out[3])
      self.assertAllClose(atten_prob2_out[1], atten_prob2_out[3])
      self.assertAllClose(atten_vec2_out[1], atten_vec2_out[3])
      self.assertGreater(
          np.max(np.abs(atten_prob1_out[0] - atten_prob2_out[0])), 0.1)
      self.assertGreater(
          np.max(np.abs(atten_prob1_out[2] - atten_prob2_out[2])), 0.1)
      self.assertGreater(
          np.max(np.abs(atten_prob2_out[0] - atten_prob2_out[2])), 0.1)
      return atten_prob2_out, atten_vec2_out

  def testPerStepSourcePaddingAdditiveAttention(self):
    params = attention.AdditiveAttention.Params()
    params.name = 'atten'
    params.params_init = py_utils.WeightInit.Gaussian(0.1, 877374)
    depth = 6
    params.source_dim = depth
    params.query_dim = depth
    params.hidden_dim = depth
    params.vn.global_vn = False
    params.vn.per_step_vn = False
    atten = params.cls(params)
    prob_out, vec_out = self._testPerStepSourcePaddingHelper(atten, depth)
    print('vec_out', np.array_repr(np.sum(vec_out, 1)))
    self.assertAllClose([2.00084352, 3.2933836, 2.30622029, 3.2933836],
                        np.sum(vec_out, 1))
    self.assertAllClose([1.0, 1.0, 1.0, 1.0], np.sum(prob_out, 1))

  def testPerStepSourcePaddingDotProductAttention(self):
    params = attention.DotProductAttention.Params()
    params.name = 'atten'
    depth = 6
    params.source_dim = depth
    params.query_dim = depth
    params.hidden_dim = depth
    params.vn.global_vn = False
    params.vn.per_step_vn = False
    atten = params.cls(params)
    prob_out, vec_out = self._testPerStepSourcePaddingHelper(atten, depth)
    print('vec_out', np.array_repr(np.sum(vec_out, 1)))
    self.assertAllClose([2.02671742, 3.38590097, 2.34964013, 3.38590097],
                        np.sum(vec_out, 1))
    self.assertAllClose([1.0, 1.0, 1.0, 1.0], np.sum(prob_out, 1))

  def testPerStepSourcePaddingMultiHeadedAttention(self):
    params = attention.MultiHeadedAttention.Params()
    params.name = 'atten'
    params.params_init = py_utils.WeightInit.Gaussian(0.1, 877374)
    depth = 6
    params.source_dim = depth
    params.query_dim = depth
    params.hidden_dim = depth
    params.vn.global_vn = False
    params.vn.per_step_vn = False
    atten = params.cls(params)
    prob_out, vec_out = self._testPerStepSourcePaddingHelper(atten, depth)
    print('vec_out', np.array_repr(np.sum(vec_out, 1)))
    self.assertAllClose([-0.006338, -0.025153, 0.041647, -0.025153],
                        np.sum(vec_out, 1))
    self.assertAllClose([1.0, 1.0, 1.0, 1.0], np.sum(prob_out, 1))

  def testPerStepSourcePaddingLocationSensitiveAttention(self):
    params = attention.LocationSensitiveAttention.Params()
    params.name = 'atten'
    params.params_init = py_utils.WeightInit.Gaussian(0.1, 877374)
    depth = 6
    params.source_dim = depth
    params.query_dim = depth
    params.hidden_dim = depth
    params.location_filter_size = 3
    params.location_num_filters = 4
    params.vn.global_vn = False
    params.vn.per_step_vn = False
    atten_state = tf.concat(
        [tf.ones([4, 1], tf.float32),
         tf.zeros([4, 5], tf.float32)], 1)
    atten_state = tf.expand_dims(atten_state, 1)
    atten = params.cls(params)
    prob_out, vec_out = self._testPerStepSourcePaddingHelper(
        atten, depth, atten_state=atten_state)
    print('vec_out', np.array_repr(np.sum(vec_out, 1)))
    self.assertAllClose([2.001103, 3.293414, 2.306448, 3.293414],
                        np.sum(vec_out, 1))
    self.assertAllClose([1.0, 1.0, 1.0, 1.0], np.sum(prob_out, 1))

  def testPerStepSourcePaddingMonotonicAttention(self):
    params = attention.MonotonicAttention.Params()
    params.name = 'atten'
    params.params_init = py_utils.WeightInit.Gaussian(0.1, 877374)
    depth = 6
    params.source_dim = depth
    params.query_dim = depth
    params.hidden_dim = depth
    params.vn.global_vn = False
    params.vn.per_step_vn = False
    atten = params.cls(params)
    atten_state = atten.ZeroAttentionState(6, 4)
    atten_state.emit_probs = tf.concat(
        [tf.ones([4, 1], tf.float32),
         tf.zeros([4, 5], tf.float32)], 1)
    prob_out, vec_out = self._testPerStepSourcePaddingHelper(
        atten, depth, atten_state=atten_state)
    print('prob_out', np.array_repr(np.sum(prob_out, 1)))
    print('vec_out', np.array_repr(np.sum(vec_out, 1)))

  def testGmmMonotonicAttentionDropout(self):
    p = attention.GmmMonotonicAttention.Params().Set(
        name='gmm_monotonic_attention', atten_dropout_prob=0.5)
    with self.assertRaises(NotImplementedError):
      p.cls(p)

  def testGmmMonotonicAttention(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(12345)
      source_vecs = tf.stack([
          tf.constant(np.random.rand(3, 4), dtype=tf.float32) for _ in range(6)
      ])
      source_contexts = tf.stack([
          tf.constant(np.random.rand(3, 5), dtype=tf.float32) for _ in range(6)
      ])
      source_padding = tf.transpose(
          tf.constant(
              [[0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0]],
              dtype=tf.float32))
      query_vec = tf.constant(np.random.rand(6, 7), dtype=tf.float32)

      params = attention.GmmMonotonicAttention.Params()
      params.name = 'gmm_atten'
      params.params_init = py_utils.WeightInit.Gaussian(0.1, 12345)
      params.source_dim = 4
      params.query_dim = 7
      params.gmm_mlp_hidden_dim = 7
      params.num_mixtures = 2
      params.vn.global_vn = False
      params.vn.per_step_vn = False

      atten = attention.GmmMonotonicAttention(params)
      atten.InitForSourcePacked(atten.theta, source_vecs, source_contexts,
                                source_padding)
      # target_batch=6
      atten_init_state = atten.ZeroAttentionState(tf.shape(source_vecs)[0], 6)

      atten_vec, atten_prob, atten_state = atten.ComputeContextVector(
          atten.theta, query_vec, atten_init_state)

      tf.global_variables_initializer().run()

      atten_vec_out, prob_out, atten_init_state_out, atten_state_out = sess.run(
          [atten_vec, atten_prob, atten_init_state, atten_state])

      self.assertEqual(atten_init_state_out.shape, atten_state_out.shape)
      self.assertEqual(atten_init_state_out.shape, (6, 2, 4))

      print(['gmm attention prob_out', np.array_repr(prob_out)])
      print(['gmm attention atten_vec_out', np.array_repr(atten_vec_out)])

      # pyformat: disable
      # pylint: disable=bad-whitespace
      expected_prob_out = [
          [ 2.45764434e-01, 3.97835493e-01, 0., 0., 4.25808690e-03,
            1.29864624e-04],
          [ 0., 3.98021877e-01, 2.37964690e-01, 5.23146540e-02, 0.,
            1.29256863e-04],
          [ 2.46294901e-01, 3.97767872e-01, 0., 5.21243662e-02, 0.,
            1.29372784e-04],
          [ 2.45875627e-01, 3.97635251e-01, 0., 0., 4.27022483e-03,
            1.30706903e-04],
          [ 0., 3.97709191e-01, 2.37897262e-01, 5.24106659e-02, 0.,
            1.30714150e-04],
          [ 2.46048093e-01, 3.97871077e-01, 0., 5.21884784e-02, 0.,
            1.29211781e-04]]
      expected_atten_vec_out = [
          [ 0.23010808,  0.43757612,  0.25150469,  0.3631629 ,  0.37140277],
          [ 0.54693544,  0.56182981,  0.21333349,  0.58108622,  0.21566363],
          [ 0.4048025 ,  0.53986353,  0.13288836,  0.22497796,  0.17450145],
          [ 0.23008531,  0.4375343 ,  0.25150725,  0.36303982,  0.37127423],
          [ 0.54661846,  0.5615437 ,  0.21332006,  0.58084518,  0.21558265],
          [ 0.40484226,  0.53978455,  0.13283314,  0.22490481,  0.17447782]]
      # pyformat: enable
      # pylint: enable=bad-whitespace

      self.assertAllClose(expected_prob_out, prob_out)
      self.assertAllClose(expected_atten_vec_out, atten_vec_out)


if __name__ == '__main__':
  tf.test.main()
