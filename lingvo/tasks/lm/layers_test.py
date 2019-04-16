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
"""Tests for layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow as tf
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.tasks.lm import layers as lm_layers

FLAGS = tf.flags.FLAGS


class RnnLmNoEmbeddingTest(test_utils.TestCase):

  def _testParams(self, dims, vocab):
    p = lm_layers.RnnLmNoEmbedding.Params()
    p.name = 'rnnlm'
    p.vocab_size = vocab
    p.rnns.cell_tpl.num_output_nodes = dims
    p.rnns.cell_tpl.num_input_nodes = dims
    p.softmax.input_dim = dims
    p.softmax.num_classes = vocab
    return p

  def testBasic(self):
    time, batch, dims, vocab = 5, 3, 6, 8
    p = self._testParams(dims, vocab)

    with self.session(use_gpu=True) as sess:
      lm = p.cls(p)
      np.random.seed(12345)
      inputs = np.random.normal(size=[time, batch, dims])
      inputs = tf.constant(inputs, tf.float32)
      paddings = np.zeros([time, batch])
      paddings[-1] = 1.0
      paddings = tf.constant(paddings, tf.float32)
      targets = tf.constant(
          np.random.randint(vocab, size=(time, batch)), tf.int32)
      sess.run(tf.global_variables_initializer())
      xent_output, state1 = lm.FPropDefaultTheta(
          inputs=inputs,
          paddings=paddings,
          state0=lm.zero_state(batch),
          labels=py_utils.NestedMap(
              class_weights=1 - paddings, class_ids=targets))

      xent_output_val, state1_val = sess.run([xent_output, state1])

      print('xent_output_val', xent_output_val)
      print('state1', state1_val)
      test_utils.CompareToGoldenSingleFloat(self, 2.1042602, xent_output_val.avg_xent)  # pyformat: disable pylint: disable=line-too-long
      self.assertAllEqual(xent_output_val.per_example_argmax,
                          np.argmax(xent_output_val.logits, axis=-1))

  def testProjection(self):
    time, batch, dims, proj, vocab = 5, 3, 6, 4, 8
    p = self._testParams(dims, vocab)
    p.rnns.cell_tpl.num_output_nodes = proj
    p.rnns.cell_tpl.num_hidden_nodes = dims
    p.softmax.input_dim = proj

    with self.session(use_gpu=True) as sess:
      lm = p.cls(p)
      np.random.seed(12345)
      inputs = np.random.normal(size=[time, batch, dims])
      inputs = tf.constant(inputs, tf.float32)
      paddings = np.zeros([time, batch])
      paddings[-1] = 1.0
      paddings = tf.constant(paddings, tf.float32)
      targets = tf.constant(
          np.random.randint(vocab, size=(time, batch)), tf.int32)
      sess.run(tf.global_variables_initializer())
      xent_output, state1 = lm.FPropDefaultTheta(
          inputs=inputs,
          paddings=paddings,
          state0=lm.zero_state(batch),
          labels=py_utils.NestedMap(
              class_weights=1 - paddings, class_ids=targets))

      xent_output_val, state1_val = sess.run([xent_output, state1])

      print('xent_output_val', xent_output_val)
      print('state1', state1_val)
      test_utils.CompareToGoldenSingleFloat(self, 2.1322777, xent_output_val.avg_xent)  # pyformat: disable pylint: disable=line-too-long
      self.assertAllEqual(xent_output_val.per_example_argmax,
                          np.argmax(xent_output_val.logits, axis=-1))

  def testBasicGrad(self):
    time, batch, dims, vocab = 5, 3, 6, 8
    p = self._testParams(dims, vocab)
    p.dtype = tf.float64

    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      lm = p.cls(p)
      np.random.seed(12345)
      inputs = np.random.normal(size=[time, batch, dims])
      inputs = tf.constant(inputs, tf.float64)
      paddings = np.zeros([time, batch])
      paddings[-1] = 1.0
      paddings = tf.constant(paddings, tf.float64)
      targets = tf.constant(
          np.random.randint(vocab, size=(time, batch)), tf.int32)

      xent_output, _ = lm.FPropDefaultTheta(
          inputs=inputs,
          paddings=paddings,
          state0=lm.zero_state(batch),
          labels=py_utils.NestedMap(
              class_weights=1 - paddings, class_ids=targets))

      lm_vars = lm.vars.Flatten()
      # Now add the backward graph.
      grads = tf.gradients(xent_output.avg_xent, lm_vars)

      tf.global_variables_initializer().run()
      self.assertEqual(len(lm_vars), len(grads))
      for x, grad_x in zip(lm_vars, grads):
        grad_symbolic = sess.run(grad_x)
        grad_numeric = test_utils.ComputeNumericGradient(
            sess, xent_output.avg_xent, x, delta=1e-6)
        self.assertAllClose(grad_symbolic, grad_numeric, atol=0.005)

  def testDirectFeatures(self,):
    time, batch, dims, vocab = 5, 3, 6, 8
    p = self._testParams(dims, vocab)
    direct_features_dim = 4
    p.direct_features_dim = direct_features_dim
    p.softmax.input_dim = dims + direct_features_dim

    with self.session(use_gpu=True) as sess:
      lm = p.cls(p)
      np.random.seed(12345)
      inputs = np.random.normal(size=[time, batch, dims])
      inputs = tf.constant(inputs, tf.float32)

      direct_features = np.random.normal(
          size=[time, batch, direct_features_dim])
      direct_features = tf.constant(direct_features, tf.float32)

      paddings = np.zeros([time, batch])
      paddings[-1] = 1.0
      paddings = tf.constant(paddings, tf.float32)
      targets = tf.constant(
          np.random.randint(vocab, size=(time, batch)), tf.int32)
      sess.run(tf.global_variables_initializer())
      xent_output, state1 = lm.FPropDefaultTheta(
          inputs=inputs,
          paddings=paddings,
          state0=lm.zero_state(batch),
          labels=py_utils.NestedMap(
              class_weights=1 - paddings, class_ids=targets),
          direct_features=direct_features)

      xent_output_val, state1_val = sess.run([xent_output, state1])

      print('xent_output_val', xent_output_val)
      print('state1', state1_val)
      test_utils.CompareToGoldenSingleFloat(self, 2.3522419929504395, xent_output_val.avg_xent)  # pyformat: disable pylint: disable=line-too-long
      self.assertAllEqual(xent_output_val.per_example_argmax,
                          np.argmax(xent_output_val.logits, axis=-1))

  def testCombineStates(self):
    time, batch, dims, vocab = 5, 3, 6, 8
    p = self._testParams(dims, vocab)

    with self.session(use_gpu=True) as sess:
      lm = p.cls(p)
      np.random.seed(12345)
      inputs = np.random.normal(size=[time, batch, dims])
      inputs = tf.constant(inputs, tf.float32)
      paddings = np.zeros([time, batch])
      paddings[-1] = 1.0
      paddings = tf.constant(paddings, tf.float32)
      targets = tf.constant(
          np.random.randint(vocab, size=(time, batch)), tf.int32)
      sess.run(tf.global_variables_initializer())

      state0 = lm.zero_state(batch)
      _, state1 = lm.FPropDefaultTheta(
          inputs=inputs,
          paddings=paddings,
          state0=state0,
          labels=py_utils.NestedMap(
              class_weights=1 - paddings, class_ids=targets))
      switch = tf.constant([True, True, False], dtype=tf.bool)
      combined_state = lm.CombineStates(state0, state1, switch)
      state0_val, state1_val, combined_state_val = sess.run(
          [state0, state1, combined_state])

      print('state1_val', state1_val)
      print('combined_state_val', combined_state_val)
      combined_m_state = combined_state_val.rnn[0].m
      combined_c_state = combined_state_val.rnn[0].c
      self.assertAllEqual(combined_m_state[0], state0_val.rnn[0].m[0])
      self.assertAllEqual(combined_c_state[0], state0_val.rnn[0].c[0])
      self.assertAllEqual(combined_m_state[1], state0_val.rnn[0].m[1])
      self.assertAllEqual(combined_c_state[1], state0_val.rnn[0].c[1])
      self.assertAllEqual(combined_m_state[2], state1_val.rnn[0].m[2])
      self.assertAllEqual(combined_c_state[2], state1_val.rnn[0].c[2])


class RnnLmTest(test_utils.TestCase):

  def testBasic(self):
    time, batch, dims, vocab = 5, 3, 6, 8

    p = lm_layers.RnnLm.Params()
    p.name = 'rnnlm'
    p.vocab_size = vocab
    p.emb.vocab_size = vocab
    p.emb.embedding_dim = dims
    p.rnns.cell_tpl.num_output_nodes = dims
    p.rnns.cell_tpl.num_input_nodes = dims
    p.softmax.input_dim = dims
    p.softmax.num_classes = vocab

    with self.session(use_gpu=True) as sess:
      lm = p.cls(p)
      np.random.seed(12345)
      inputs = np.random.randint(vocab, size=[time, batch])
      targets = np.zeros([time, batch])
      targets[:-1] = inputs[1:]
      inputs = tf.constant(inputs, tf.int32)
      paddings = np.zeros([time, batch])
      paddings[-1] = 1.0
      paddings = tf.constant(paddings, tf.float32)
      targets = tf.constant(targets, tf.int32)
      sess.run(tf.global_variables_initializer())
      xent_output, state1 = lm.FPropDefaultTheta(
          inputs=inputs,
          paddings=paddings,
          state0=lm.zero_state(batch),
          labels=py_utils.NestedMap(
              class_weights=1 - paddings, class_ids=targets))

      xent_output_val, state1_val = sess.run([xent_output, state1])

      print('xent_output_val', xent_output_val)
      print('state1', state1_val)
      test_utils.CompareToGoldenSingleFloat(self, 2.0853612, xent_output_val.avg_xent)  # pyformat: disable pylint: disable=line-too-long
      self.assertAllEqual(xent_output_val.per_example_argmax,
                          np.argmax(xent_output_val.logits, axis=-1))

  def testDropout(self):
    seed = 12345
    tf.set_random_seed(seed)
    np.random.seed(seed)

    time, batch, dims, vocab = 5, 3, 6, 8

    p = lm_layers.RnnLm.Params()
    p.name = 'rnnlm'
    p.vocab_size = vocab
    p.emb.vocab_size = vocab
    p.emb.embedding_dim = dims
    p.embedding_dropout_keep_prob = 0.8
    p.embedding_dropout_seed = seed
    p.rnns.cell_tpl.num_output_nodes = dims
    p.rnns.cell_tpl.num_input_nodes = dims
    p.softmax.input_dim = dims
    p.softmax.num_classes = vocab

    with self.session(use_gpu=True) as sess:
      lm = p.cls(p)
      inputs = np.random.randint(vocab, size=[time, batch])
      targets = np.zeros([time, batch])
      targets[:-1] = inputs[1:]
      inputs = tf.constant(inputs, tf.int32)
      paddings = np.zeros([time, batch])
      paddings[-1] = 1.0
      paddings = tf.constant(paddings, tf.float32)
      targets = tf.constant(targets, tf.int32)
      sess.run(tf.global_variables_initializer())
      xent_output, state1 = lm.FPropDefaultTheta(
          inputs=inputs,
          paddings=paddings,
          state0=lm.zero_state(batch),
          labels=py_utils.NestedMap(
              class_weights=1 - paddings, class_ids=targets))

      xent_output_val, state1_val = sess.run([xent_output, state1])

      print('xent_output_val', xent_output_val)
      print('state1', state1_val)
      test_utils.CompareToGoldenSingleFloat(self, 2.084798, xent_output_val.avg_xent)  # pyformat: disable pylint: disable=line-too-long
      self.assertAllEqual(xent_output_val.per_example_argmax,
                          np.argmax(xent_output_val.logits, axis=-1))

  def testBasicGrad(self):
    time, batch, dims, vocab = 5, 3, 6, 8

    p = lm_layers.RnnLm.Params()
    p.dtype = tf.float64
    p.name = 'rnnlm'
    p.vocab_size = vocab
    p.emb.vocab_size = vocab
    p.emb.embedding_dim = dims
    p.rnns.cell_tpl.num_output_nodes = dims
    p.rnns.cell_tpl.num_input_nodes = dims
    p.softmax.input_dim = dims
    p.softmax.num_classes = vocab

    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      lm = p.cls(p)
      np.random.seed(12345)
      inputs = np.random.randint(vocab, size=[time, batch])
      targets = np.zeros([time, batch])
      targets[:-1] = inputs[1:]
      inputs = tf.constant(inputs, tf.int32)
      paddings = np.zeros([time, batch])
      paddings[-1] = 1.0
      paddings = tf.constant(paddings, tf.float64)
      targets = tf.constant(targets, tf.int32)
      sess.run(tf.global_variables_initializer())
      xent_output, _ = lm.FPropDefaultTheta(
          inputs=inputs,
          paddings=paddings,
          state0=lm.zero_state(batch),
          labels=py_utils.NestedMap(
              class_weights=1 - paddings, class_ids=targets))

      lm_vars = lm.vars.Flatten()
      # Now add the backward graph.
      grads = tf.gradients(xent_output.avg_xent, lm_vars)

      for i, x in enumerate(grads):
        if isinstance(x, tf.IndexedSlices):
          grads[i] = tf.unsorted_segment_sum(x.values, x.indices,
                                             x.dense_shape[0])

      tf.global_variables_initializer().run()
      self.assertEqual(len(lm_vars), len(grads))
      for x, grad_x in zip(lm_vars, grads):
        grad_symbolic = sess.run(grad_x)
        grad_numeric = test_utils.ComputeNumericGradient(
            sess, xent_output.avg_xent, x, delta=1e-6)
        self.assertAllClose(grad_symbolic, grad_numeric, atol=0.005)


class ConditionalRnnLmTest(test_utils.TestCase):

  def testBasic(self):
    time, batch, dims, vocab, condition_dim = 5, 3, 6, 8, 7

    p = lm_layers.ConditionalRnnLm.Params()
    p.name = 'conditionalrnnlm'
    p.vocab_size = vocab
    p.emb.vocab_size = vocab
    p.emb.embedding_dim = dims
    model_dim = dims + condition_dim
    p.rnns.cell_tpl.num_output_nodes = model_dim
    p.rnns.cell_tpl.num_input_nodes = model_dim
    p.softmax.input_dim = model_dim
    p.softmax.num_classes = vocab
    p.condition_dim = condition_dim

    with self.session(use_gpu=True) as sess:
      lm = p.cls(p)
      np.random.seed(12345)
      inputs = np.random.randint(vocab, size=[time, batch])
      targets = np.zeros([time, batch])
      targets[:-1] = inputs[1:]
      inputs = tf.constant(inputs, tf.int32)
      paddings = np.zeros([time, batch])
      paddings[-1] = 1.0
      paddings = tf.constant(paddings, tf.float32)
      targets = tf.constant(targets, tf.int32)
      condition = tf.constant(np.ones([batch, condition_dim]), tf.float64)
      sess.run(tf.global_variables_initializer())
      xent_output, state1 = lm.FPropDefaultTheta(
          inputs=inputs,
          paddings=paddings,
          state0=lm.zero_state(batch),
          condition=condition,
          labels=py_utils.NestedMap(
              class_weights=1 - paddings, class_ids=targets))

      xent_output_val, state1_val = sess.run([xent_output, state1])

      print('xent_output_val', xent_output_val)
      print('state1', state1_val)
      test_utils.CompareToGoldenSingleFloat(self, 2.10713076, xent_output_val.avg_xent)  # pyformat: disable pylint: disable=line-too-long
      self.assertAllEqual(xent_output_val.per_example_argmax,
                          np.argmax(xent_output_val.logits, axis=-1))

  def testDropout(self):
    seed = 12345
    tf.set_random_seed(seed)
    np.random.seed(seed)

    time, batch, dims, vocab, condition_dim = 5, 3, 6, 8, 7

    p = lm_layers.ConditionalRnnLm.Params()
    p.name = 'conditionalrnnlm'
    p.vocab_size = vocab
    p.emb.vocab_size = vocab
    p.emb.embedding_dim = dims
    p.embedding_dropout_keep_prob = 0.8
    p.embedding_dropout_seed = seed
    model_dim = dims + condition_dim
    p.rnns.cell_tpl.num_output_nodes = model_dim
    p.rnns.cell_tpl.num_input_nodes = model_dim
    p.softmax.input_dim = model_dim
    p.softmax.num_classes = vocab
    p.condition_dim = condition_dim

    with self.session(use_gpu=True) as sess:
      lm = p.cls(p)
      inputs = np.random.randint(vocab, size=[time, batch])
      targets = np.zeros([time, batch])
      targets[:-1] = inputs[1:]
      inputs = tf.constant(inputs, tf.int32)
      paddings = np.zeros([time, batch])
      paddings[-1] = 1.0
      paddings = tf.constant(paddings, tf.float32)
      targets = tf.constant(targets, tf.int32)
      condition = tf.constant(np.ones([batch, condition_dim]), tf.float64)
      sess.run(tf.global_variables_initializer())
      xent_output, state1 = lm.FPropDefaultTheta(
          inputs=inputs,
          paddings=paddings,
          state0=lm.zero_state(batch),
          condition=condition,
          labels=py_utils.NestedMap(
              class_weights=1 - paddings, class_ids=targets))

      xent_output_val, state1_val = sess.run([xent_output, state1])

      print('xent_output_val', xent_output_val)
      print('state1', state1_val)
      test_utils.CompareToGoldenSingleFloat(self, 2.17278885, xent_output_val.avg_xent)  # pyformat: disable pylint: disable=line-too-long
      self.assertAllEqual(xent_output_val.per_example_argmax,
                          np.argmax(xent_output_val.logits, axis=-1))

  def testBasicGrad(self):
    time, batch, dims, vocab, condition_dim = 5, 3, 6, 8, 7

    p = lm_layers.ConditionalRnnLm.Params()
    p.name = 'conditionalrnnlm'
    p.dtype = tf.float64
    p.vocab_size = vocab
    p.emb.vocab_size = vocab
    p.emb.embedding_dim = dims
    model_dim = dims + condition_dim
    p.rnns.cell_tpl.num_output_nodes = model_dim
    p.rnns.cell_tpl.num_input_nodes = model_dim
    p.softmax.input_dim = model_dim
    p.softmax.num_classes = vocab
    p.condition_dim = condition_dim

    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      lm = p.cls(p)
      np.random.seed(12345)
      inputs = np.random.randint(vocab, size=[time, batch])
      targets = np.zeros([time, batch])
      targets[:-1] = inputs[1:]
      inputs = tf.constant(inputs, tf.int32)
      paddings = np.zeros([time, batch])
      paddings[-1] = 1.0
      paddings = tf.constant(paddings, tf.float64)
      targets = tf.constant(targets, tf.int32)
      condition = tf.constant(np.ones([batch, condition_dim]), tf.float64)
      sess.run(tf.global_variables_initializer())
      xent_output, _ = lm.FPropDefaultTheta(
          inputs=inputs,
          paddings=paddings,
          state0=lm.zero_state(batch),
          condition=condition,
          labels=py_utils.NestedMap(
              class_weights=1 - paddings, class_ids=targets))

      lm_vars = lm.vars.Flatten()
      # Now add the backward graph.
      grads = tf.gradients(xent_output.avg_xent, lm_vars)

      for i, x in enumerate(grads):
        if isinstance(x, tf.IndexedSlices):
          grads[i] = tf.unsorted_segment_sum(x.values, x.indices,
                                             x.dense_shape[0])

      tf.global_variables_initializer().run()
      self.assertEqual(len(lm_vars), len(grads))
      for x, grad_x in zip(lm_vars, grads):
        grad_symbolic = sess.run(grad_x)
        grad_numeric = test_utils.ComputeNumericGradient(
            sess, xent_output.avg_xent, x, delta=1e-6)
        self.assertAllClose(grad_symbolic, grad_numeric, atol=0.005)


class MoeLmTest(test_utils.TestCase):

  def _MoeLmParams(self, vocab, shared_emb, add_postgating_rnn=True):
    p = lm_layers.MoeLm.Params()
    p.name = 'moelm'
    p.vocab_size = vocab
    p.emb.vocab_size = vocab
    p.emb.embedding_dim = 17
    p.shared_emb = shared_emb
    p.add_postgating_rnn = add_postgating_rnn
    p.rnns.cell_tpl.num_input_nodes = 17
    p.rnns.cell_tpl.num_output_nodes = 23
    p.number_of_experts = 4
    p.merge.vocab_size = vocab
    p.merge.rnns.cell_tpl.num_input_nodes = 23
    p.merge.rnns.cell_tpl.num_output_nodes = 32
    p.merge.softmax.input_dim = 32
    p.merge.softmax.num_classes = vocab
    return p

  def _GetData(self, vocab, time, batch):
    inputs = np.random.randint(vocab, size=[time, batch])
    targets = np.zeros([time, batch])
    targets[:-1] = inputs[1:]
    inputs = tf.constant(inputs, tf.int32)
    paddings = np.zeros([time, batch])
    paddings[-1] = 1.0
    paddings = tf.constant(paddings, tf.float32)
    targets = tf.constant(targets, tf.int32)
    labels = py_utils.NestedMap(class_weights=1 - paddings, class_ids=targets)
    return inputs, paddings, labels

  def testFPropNoSharedEmb(self):
    vocab, time, batch = 7, 13, 3
    p = self._MoeLmParams(vocab, False)

    with self.session(graph=tf.Graph()) as sess:
      np.random.seed(54321)
      tf.set_random_seed(123456)
      lm = p.cls(p)
      inputs, paddings, labels = self._GetData(vocab, time, batch)
      sess.run(tf.global_variables_initializer())
      xent_output, state1 = lm.FPropDefaultTheta(
          inputs=inputs,
          paddings=paddings,
          state0=lm.zero_state(batch),
          labels=labels)

      xent_output_val, state1_val = sess.run([xent_output, state1])

      print('xent_output_val', xent_output_val)
      print('state1', state1_val)
      test_utils.CompareToGoldenSingleFloat(self, 1.9460623, xent_output_val.avg_xent)  # pyformat: disable pylint: disable=line-too-long
      self.assertAllEqual(xent_output_val.per_example_argmax,
                          np.argmax(xent_output_val.logits, axis=-1))

  def testFPropSharedEmb(self):
    vocab, time, batch = 7, 13, 3
    p = self._MoeLmParams(vocab, False)

    with self.session(graph=tf.Graph()) as sess:
      np.random.seed(54321)
      tf.set_random_seed(123456)
      lm = p.cls(p)
      inputs, paddings, labels = self._GetData(vocab, time, batch)
      sess.run(tf.global_variables_initializer())
      xent_output, state1 = lm.FPropDefaultTheta(
          inputs=inputs,
          paddings=paddings,
          state0=lm.zero_state(batch),
          labels=labels)

      xent_output_val, state1_val = sess.run([xent_output, state1])

      print('xent_output_val', xent_output_val)
      print('state1', state1_val)
      test_utils.CompareToGoldenSingleFloat(self, 1.9460623, xent_output_val.avg_xent)  # pyformat: disable pylint: disable=line-too-long
      self.assertAllEqual(xent_output_val.per_example_argmax,
                          np.argmax(xent_output_val.logits, axis=-1))

  def testFPropNoPostGatingRNN(self):
    vocab, time, batch = 7, 13, 3
    p = self._MoeLmParams(vocab, False, False)

    with self.session(graph=tf.Graph()) as sess:
      np.random.seed(54321)
      tf.set_random_seed(123456)
      lm = p.cls(p)
      inputs, paddings, labels = self._GetData(vocab, time, batch)
      sess.run(tf.global_variables_initializer())
      xent_output, state1 = lm.FPropDefaultTheta(
          inputs=inputs,
          paddings=paddings,
          state0=lm.zero_state(batch),
          labels=labels)

      xent_output_val, state1_val = sess.run([xent_output, state1])

      print('xent_output_val', xent_output_val)
      print('state1', state1_val)
      test_utils.CompareToGoldenSingleFloat(self, 1.9443978, xent_output_val.avg_xent)  # pyformat: disable pylint: disable=line-too-long
      self.assertAllEqual(xent_output_val.per_example_argmax,
                          np.argmax(xent_output_val.logits, axis=-1))

  def testBProp(self):
    vocab, time, batch = 7, 4, 3
    p = self._MoeLmParams(vocab, True)
    p.dtype = tf.float64

    with self.session(graph=tf.Graph()) as sess:
      np.random.seed(54321)
      tf.set_random_seed(123456)
      lm = p.cls(p)
      inputs, paddings, labels = self._GetData(vocab, time, batch)
      sess.run(tf.global_variables_initializer())
      xent_output, _ = lm.FPropDefaultTheta(
          inputs=inputs,
          paddings=tf.cast(paddings, p.dtype),
          state0=lm.zero_state(batch),
          labels=labels)

      lm_vars = lm.vars.Flatten()
      # Now add the backward graph.
      grads = tf.gradients(xent_output.avg_xent, lm_vars)

      for i, x in enumerate(grads):
        if isinstance(x, tf.IndexedSlices):
          grads[i] = tf.unsorted_segment_sum(x.values, x.indices,
                                             x.dense_shape[0])

      tf.global_variables_initializer().run()
      self.assertEqual(len(lm_vars), len(grads))
      step = 11  # Speed up the test.
      for x, grad_x in zip(lm_vars, grads):
        grad_symbolic = sess.run(grad_x)
        grad_numeric = test_utils.ComputeNumericGradient(
            sess, xent_output.avg_xent, x, step=step, delta=1e-6)
        self.assertAllClose(
            grad_symbolic.reshape([-1])[::step],
            grad_numeric.reshape([-1])[::step])


class TransformerLmNoEmbeddingTest(test_utils.TestCase):

  def _testParams(self, dtype=tf.float32):
    model_dim, hidden_dim, vocab_size = 4, 6, 8
    p = lm_layers.TransformerLmNoEmbedding.Params()
    p.name = 'xformerlm'
    p.random_seed = 93820986
    p.dtype = dtype
    p.vocab_size = vocab_size
    p.model_dim = model_dim
    p.num_trans_layers = 3
    p.position_emb.embedding_dim = model_dim
    p.trans_tpl.source_dim = model_dim
    p.trans_tpl.tr_atten_tpl.num_attention_heads = 2
    p.trans_tpl.tr_fflayer_tpl.hidden_dim = hidden_dim
    p.softmax.input_dim = model_dim
    p.softmax.num_classes = vocab_size
    return p

  def _testInputs(self, dtype=tf.float32, last_padding=1.0):
    time, batch, model_dim, vocab_size = 5, 3, 4, 8
    np.random.seed(12345)
    inputs = np.random.normal(size=[time, batch, model_dim])
    inputs = tf.constant(inputs, dtype)
    paddings = np.zeros([time, batch])
    paddings[-1] = last_padding
    paddings = tf.constant(paddings, dtype)
    targets = tf.constant(
        np.random.randint(vocab_size, size=(time, batch)), tf.int32)
    return inputs, paddings, targets

  def testBasic(self):
    p = self._testParams(dtype=tf.float32)
    with self.session(use_gpu=True) as sess:
      lm = p.cls(p)
      inputs, paddings, targets = self._testInputs(dtype=tf.float32)
      sess.run(tf.global_variables_initializer())
      xent_output, _ = lm.FPropDefaultTheta(
          inputs=inputs,
          paddings=paddings,
          labels=py_utils.NestedMap(
              class_weights=1 - paddings, class_ids=targets))
      xent_output_val = sess.run(xent_output)
      print('xformer xent_output_val.avg_xent', xent_output_val.avg_xent)
      test_utils.CompareToGoldenSingleFloat(self, 1.91814, xent_output_val.avg_xent)  # pyformat: disable pylint: disable=line-too-long
      self.assertAllEqual(xent_output_val.per_example_argmax,
                          np.argmax(xent_output_val.logits, axis=-1))

  def testBasicGrad(self):
    p = self._testParams(dtype=tf.float64)
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      lm = p.cls(p)
      inputs, paddings, targets = self._testInputs(dtype=tf.float64)
      xent_output, _ = lm.FPropDefaultTheta(
          inputs=inputs,
          paddings=paddings,
          labels=py_utils.NestedMap(
              class_weights=1 - paddings, class_ids=targets))

      lm_vars = lm.vars.Flatten()
      # Now add the backward graph.
      grads = tf.gradients(xent_output.avg_xent, lm_vars)

      tf.global_variables_initializer().run()
      self.assertEqual(len(lm_vars), len(grads))
      for x, grad_x in zip(lm_vars, grads):
        grad_symbolic = sess.run(grad_x)
        grad_numeric = test_utils.ComputeNumericGradient(
            sess, xent_output.avg_xent, x, delta=1e-6)
        self.assertAllClose(grad_symbolic, grad_numeric, atol=0.005)

  def testStep(self):
    p = self._testParams(dtype=tf.float32)
    with self.session(use_gpu=True) as sess:
      lm = p.cls(p)
      inputs, paddings, _ = self._testInputs(dtype=tf.float32, last_padding=0.0)

      sess.run(tf.global_variables_initializer())
      xent_output, _ = lm.FPropDefaultTheta(inputs=inputs, paddings=paddings)
      logits1 = xent_output.logits

      time, batch = 5, 3
      prefix_states = lm.zero_state(batch)
      logits2 = []
      for i in range(time):
        l_i_out, prefix_states = lm.Step(lm.theta, inputs[i, :, :],
                                         paddings[i, :], prefix_states)
        logits2.append(l_i_out.logits)
      logits2 = tf.stack(logits2)

      tf.global_variables_initializer().run()
      logits1_v, logits2_v = sess.run([logits1, logits2])
      print('xformer logits1_v', logits1_v)
      print('xformer logits2_v', logits2_v)
      self.assertAllClose(logits1_v, logits2_v)


class TransformerLmTest(test_utils.TestCase):

  def testBasic(self):
    time, batch, dims, hidden_dim, vocab = 5, 3, 6, 4, 8

    p = lm_layers.TransformerLm.Params()
    p.name = 'transformerlm'
    p.vocab_size = vocab
    p.emb.vocab_size = vocab
    p.emb.embedding_dim = dims
    p.model_dim = dims
    p.num_trans_layers = 3
    p.position_emb.embedding_dim = dims
    p.trans_tpl.source_dim = dims
    p.trans_tpl.tr_atten_tpl.num_attention_heads = 2
    p.trans_tpl.tr_fflayer_tpl.hidden_dim = hidden_dim
    p.softmax.input_dim = dims
    p.softmax.num_classes = vocab

    with self.session(use_gpu=True) as sess:
      lm = p.cls(p)
      np.random.seed(12345)
      inputs = np.random.randint(vocab, size=[time, batch])
      targets = np.zeros([time, batch])
      targets[:-1] = inputs[1:]
      inputs = tf.constant(inputs, tf.int32)
      paddings = np.zeros([time, batch])
      paddings[-1] = 1.0
      paddings = tf.constant(paddings, tf.float32)
      targets = tf.constant(targets, tf.int32)
      sess.run(tf.global_variables_initializer())
      xent_output, _ = lm.FPropDefaultTheta(
          inputs=inputs,
          paddings=paddings,
          labels=py_utils.NestedMap(
              class_weights=1 - paddings, class_ids=targets))

      xent_output_val = sess.run(xent_output)

      print('xent_output_val', xent_output_val)
      test_utils.CompareToGoldenSingleFloat(self, 3.0489848, xent_output_val.avg_xent)  # pyformat: disable pylint: disable=line-too-long
      self.assertAllEqual(xent_output_val.per_example_argmax,
                          np.argmax(xent_output_val.logits, axis=-1))

  def testDropout(self):
    seed = 12345
    tf.set_random_seed(seed)
    np.random.seed(seed)

    time, batch, dims, hidden_dim, vocab = 5, 3, 6, 4, 8

    p = lm_layers.TransformerLm.Params()
    p.name = 'transformerlm'
    p.vocab_size = vocab
    p.emb.vocab_size = vocab
    p.emb.embedding_dim = dims
    p.model_dim = dims
    p.num_trans_layers = 3
    p.trans_tpl.source_dim = dims
    p.trans_tpl.tr_atten_tpl.num_attention_heads = 2
    p.trans_tpl.tr_fflayer_tpl.hidden_dim = hidden_dim
    p.softmax.input_dim = dims
    p.softmax.num_classes = vocab

    with self.session(use_gpu=True) as sess:
      lm = p.cls(p)
      inputs = np.random.randint(vocab, size=[time, batch])
      targets = np.zeros([time, batch])
      targets[:-1] = inputs[1:]
      inputs = tf.constant(inputs, tf.int32)
      paddings = np.zeros([time, batch])
      paddings[-1] = 1.0
      paddings = tf.constant(paddings, tf.float32)
      targets = tf.constant(targets, tf.int32)
      sess.run(tf.global_variables_initializer())
      xent_output, _ = lm.FPropDefaultTheta(
          inputs=inputs,
          paddings=paddings,
          labels=py_utils.NestedMap(
              class_weights=1 - paddings, class_ids=targets))

      xent_output_val = sess.run(xent_output)

      print('xent_output_val', xent_output_val)
      test_utils.CompareToGoldenSingleFloat(self, 3.038596, xent_output_val.avg_xent)  # pyformat: disable pylint: disable=line-too-long
      self.assertAllEqual(xent_output_val.per_example_argmax,
                          np.argmax(xent_output_val.logits, axis=-1))

  def testBasicGrad(self):
    time, batch, dims, hidden_dim, vocab = 5, 3, 6, 4, 8

    p = lm_layers.TransformerLm.Params()
    p.dtype = tf.float64
    p.name = 'transformerlm'
    p.vocab_size = vocab
    p.emb.vocab_size = vocab
    p.emb.embedding_dim = dims
    p.model_dim = dims
    p.num_trans_layers = 1
    p.trans_tpl.source_dim = dims
    p.trans_tpl.tr_atten_tpl.num_attention_heads = 2
    p.trans_tpl.tr_fflayer_tpl.hidden_dim = hidden_dim
    p.softmax.input_dim = dims
    p.softmax.num_classes = vocab

    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      lm = p.cls(p)
      np.random.seed(12345)
      inputs = np.random.randint(vocab, size=[time, batch])
      targets = np.zeros([time, batch])
      targets[:-1] = inputs[1:]
      inputs = tf.constant(inputs, tf.int32)
      paddings = np.zeros([time, batch])
      paddings[-1] = 1.0
      paddings = tf.constant(paddings, tf.float64)
      targets = tf.constant(targets, tf.int32)
      xent_output, _ = lm.FPropDefaultTheta(
          inputs=inputs,
          paddings=paddings,
          labels=py_utils.NestedMap(
              class_weights=1 - paddings, class_ids=targets))

      lm_vars = lm.vars.Flatten()
      grads = tf.gradients(xent_output.avg_xent, lm_vars)

      for i, x in enumerate(grads):
        if isinstance(x, tf.IndexedSlices):
          grads[i] = tf.unsorted_segment_sum(x.values, x.indices,
                                             x.dense_shape[0])

      tf.global_variables_initializer().run()
      self.assertEqual(len(lm_vars), len(grads))
      for x, grad_x in zip(lm_vars, grads):
        grad_symbolic = sess.run(grad_x)
        grad_numeric = test_utils.ComputeNumericGradient(
            sess, xent_output.avg_xent, x, delta=1e-6)
        self.assertAllClose(grad_symbolic, grad_numeric, atol=0.005)


class GPipeTransformerLmNoEmbeddingTest(test_utils.TestCase):

  def _testParams(self, dtype=tf.float32):
    model_dim, hidden_dim, vocab_size = 4, 6, 8
    p = lm_layers.GPipeTransformerLmNoEmbedding.Params()
    p.name = 'xformerlm'
    p.random_seed = 93820986
    p.dtype = dtype
    p.vocab_size = vocab_size
    p.model_dim = model_dim
    p.stack.num_encoder_layers = 4
    trans_tpl = p.stack.encoder_tpl
    trans_tpl.tr_atten_tpl.num_attention_heads = 2
    trans_tpl.tr_fflayer_tpl.hidden_dim = hidden_dim
    p.softmax.num_classes = vocab_size
    return p

  def _testInputs(self, dtype=tf.float32, last_padding=1.0):
    time, batch, model_dim, vocab_size = 5, 3, 4, 8
    np.random.seed(12345)
    inputs = np.random.normal(size=[time, batch, model_dim])
    inputs = tf.constant(inputs, dtype)
    paddings = np.zeros([time, batch])
    paddings[-1] = last_padding
    paddings = tf.constant(paddings, dtype)
    targets = tf.constant(
        np.random.randint(vocab_size, size=(time, batch)), tf.int32)
    return inputs, paddings, targets

  def testBasic(self):
    p = self._testParams(dtype=tf.float32)
    with self.session(use_gpu=True) as sess:
      lm = p.cls(p)
      assert p.stack.encoder_tpl.tr_atten_tpl.is_masked
      inputs, paddings, targets = self._testInputs(dtype=tf.float32)
      sess.run(tf.global_variables_initializer())
      xent_output, _ = lm.FPropDefaultTheta(
          inputs=inputs,
          paddings=paddings,
          labels=py_utils.NestedMap(
              class_weights=1 - paddings, class_ids=targets))
      xent_output_val = sess.run(xent_output)
      print('xformer xent_output_val.avg_xent', xent_output_val.avg_xent)
      test_utils.CompareToGoldenSingleFloat(self, 1.9776554,
                                            xent_output_val.avg_xent)
      self.assertAllEqual(xent_output_val.per_example_argmax,
                          np.argmax(xent_output_val.logits, axis=-1))

  def testBasicGrad(self):
    p = self._testParams(dtype=tf.float64)
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      lm = p.cls(p)
      inputs, paddings, targets = self._testInputs(dtype=tf.float64)
      xent_output, _ = lm.FPropDefaultTheta(
          inputs=inputs,
          paddings=paddings,
          labels=py_utils.NestedMap(
              class_weights=1 - paddings, class_ids=targets))

      lm_vars = lm.vars.Flatten()
      # Now add the backward graph.
      grads = tf.gradients(xent_output.avg_xent, lm_vars)

      sess.run(tf.global_variables_initializer())
      self.assertEqual(len(lm_vars), len(grads))
      for x, grad_x in zip(lm_vars, grads):
        grad_symbolic = sess.run(grad_x)
        grad_numeric = test_utils.ComputeNumericGradient(
            sess, xent_output.avg_xent, x, delta=1e-6)
        self.assertAllClose(grad_symbolic, grad_numeric, atol=0.005)


class GPipeTransformerLmTest(test_utils.TestCase):

  def _testParams(self, batch, dims, hidden_dim, vocab):
    p = lm_layers.GPipeTransformerLm.Params()
    p.name = 'transformerlm'
    p.vocab_size = vocab
    p.emb.vocab_size = vocab
    p.emb.embedding_dim = dims
    p.model_dim = dims
    p.stack.num_encoder_layers = 4
    trans_tpl = p.stack.encoder_tpl
    trans_tpl.tr_atten_tpl.num_attention_heads = 2
    trans_tpl.tr_fflayer_tpl.hidden_dim = hidden_dim
    p.softmax.num_classes = vocab
    return p

  def _SetupGraph(self, p, time, batch, vocab, return_grad=False):
    lm = p.cls(p)
    np.random.seed(12345)
    inputs = np.random.randint(vocab, size=[time, batch])
    targets = np.zeros([time, batch])
    targets[:-1] = inputs[1:]
    inputs = tf.constant(inputs, tf.int32)
    paddings = np.zeros([time, batch])
    paddings[-1] = 1.0
    paddings = tf.constant(paddings, tf.float64 if return_grad else tf.float32)
    targets = tf.constant(targets, tf.int32)
    xent_output, _ = lm.FPropDefaultTheta(
        inputs=inputs,
        paddings=paddings,
        labels=py_utils.NestedMap(
            class_weights=1 - paddings, class_ids=targets))
    if not return_grad:
      return xent_output

    lm_vars = lm.vars.Flatten()
    grads = tf.gradients(xent_output.avg_xent, lm_vars)
    for i, x in enumerate(grads):
      if isinstance(x, tf.IndexedSlices):
        grads[i] = tf.unsorted_segment_sum(x.values, x.indices,
                                           x.dense_shape[0])
    self.assertEqual(len(lm_vars), len(grads))
    return xent_output, lm_vars, grads

  def testBasic(self):
    time, batch, dims, hidden_dim, vocab = 5, 3, 6, 4, 8
    p = self._testParams(batch, dims, hidden_dim, vocab)
    xent_output = self._SetupGraph(p, time, batch, vocab)
    assert p.stack.encoder_tpl.tr_atten_tpl.is_masked
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      xent_output_val = sess.run(xent_output)

      print('xent_output_val', xent_output_val)
      test_utils.CompareToGoldenSingleFloat(self, 3.06884766,
                                            xent_output_val.avg_xent)
      self.assertAllEqual(xent_output_val.per_example_argmax,
                          np.argmax(xent_output_val.logits, axis=-1))

  def testBasicGrad(self):
    time, batch, dims, hidden_dim, vocab = 5, 3, 6, 4, 8
    p = self._testParams(batch, dims, hidden_dim, vocab)
    p.dtype = tf.float64
    xent_output, lm_vars, grads = self._SetupGraph(
        p, time, batch, vocab, return_grad=True)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      for x, grad_x in zip(lm_vars, grads):
        grad_symbolic = sess.run(grad_x)
        grad_numeric = test_utils.ComputeNumericGradient(
            sess, xent_output.avg_xent, x, delta=1e-6)
        self.assertAllClose(grad_symbolic, grad_numeric, atol=0.005)


if __name__ == '__main__':
  tf.test.main()
