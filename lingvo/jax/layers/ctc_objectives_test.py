# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for ctc_objectives."""

from absl.testing import absltest
import jax
from jax.experimental import jax2tf
import jax.numpy as jnp
from lingvo.jax import pytypes
from lingvo.jax import test_utils
from lingvo.jax.layers import ctc_objectives
import numpy as np
import tensorflow as tf

JTensor = pytypes.JTensor


def tf_ctc_loss(logits: np.ndarray,
                logits_paddings: np.ndarray,
                labels: np.ndarray,
                labels_paddings: np.ndarray,
                blank_id: int = 0):
  assert blank_id == 0

  def tf_ctc_loss_wrapper(logits, logits_paddings, labels, labels_paddings):
    labels = tf.cast(labels, tf.int32)
    logit_length = tf.cast(
        tf.reduce_sum(1.0 - logits_paddings, axis=-1), tf.int32)
    label_length = tf.cast(
        tf.reduce_sum(1.0 - labels_paddings, axis=-1), tf.int32)
    return tf.nn.ctc_loss(
        labels=labels,
        logits=logits,
        label_length=label_length,
        logit_length=logit_length,
        logits_time_major=False)

  return jax2tf.call_tf(tf_ctc_loss_wrapper)(logits, logits_paddings, labels,
                                             labels_paddings)


def average_ctc_loss(logprobs: JTensor, logprob_paddings: JTensor,
                     labels: JTensor, label_paddings: JTensor) -> JTensor:
  return jnp.average(
      ctc_objectives.ctc_loss(logprobs, logprob_paddings, labels,
                              label_paddings)[0])


def lengths_to_paddings(lengths: JTensor, maxlength: int) -> JTensor:
  indices = jnp.arange(maxlength).reshape((1,) * lengths.ndim + (maxlength,))
  lengths = jnp.expand_dims(lengths, axis=-1)
  elem_valid = indices < lengths
  return np.logical_not(elem_valid).astype(np.float32)


class CtcTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(1234)

  def test_fprop_bprop(self):
    """Smoke test."""
    inputsteps = 898
    outputsteps = 512
    batchsize = 8
    nclass = 50
    ndevice = jax.local_device_count()

    logprobs = jax.nn.log_softmax(
        np.random.randn(ndevice, batchsize, inputsteps, nclass))
    logprob_paddings = np.zeros((ndevice, batchsize, inputsteps))
    labels = np.random.uniform(
        1, nclass, size=(ndevice, batchsize, outputsteps)).astype(np.int32)
    lens = np.random.uniform(
        5, outputsteps, size=(
            ndevice,
            batchsize,
        )).astype(np.int32)
    label_paddings = lengths_to_paddings(lens, outputsteps)

    jax.pmap(jax.jit(ctc_objectives.ctc_loss))(logprobs, logprob_paddings,
                                               labels, label_paddings)

    jax.pmap(jax.jit(jax.grad(average_ctc_loss)))(logprobs, logprob_paddings,
                                                  labels, label_paddings)

  def test_with_one_to_one_alignment(self):
    # when inputsteps and outputsteps are equal, no phi will be allowed
    batchsize = 8
    steps = 50
    nclasses = 40
    logprobs = np.random.randn(batchsize, steps, nclasses)
    logprobs = jax.nn.log_softmax(logprobs)
    labels = np.random.uniform(
        1, nclasses, size=(batchsize, steps)).astype(np.int32)
    # This case only check the cases without same label repetition.
    # `test_repeat_with_one_to_one_alignment` below complements those cases.
    # Redraw samples for satisfying the constraint.
    for n in range(labels.shape[0]):
      for t in range(1, labels.shape[1]):
        while labels[n, t] == labels[n, t - 1]:
          labels[n, t] = np.random.uniform(1, nclasses)

    per_seq_loss, aux_vars = ctc_objectives.ctc_loss(
        logprobs, np.zeros(logprobs.shape[:2]), labels, np.zeros(labels.shape))

    for b in range(batchsize):
      p = 0.0
      for t in range(steps):
        p += logprobs[b, t, labels[b, t]]
      self.assertAllClose(jnp.array(-p), per_seq_loss[b])

      # Check logalpha interim variables
      # 1. All-phi path
      self.assertAllClose(aux_vars['logalpha_phi'][-1, b, 0],
                          jnp.sum(logprobs[b, :, 0]))
      # 2. After emitting all the labels
      self.assertAllClose(aux_vars['logalpha_emit'][-1, b, steps - 1],
                          -per_seq_loss[b])
      self.assertAllClose(aux_vars['logalpha_phi'][-1, b, -1], -per_seq_loss[b])

  def test_with_one_to_one_alignment_and_paddings(self):
    batch_size = 1
    nclasses = 8
    steps = 4
    logits = np.random.normal(size=[batch_size, steps, nclasses])
    logprobs = jax.nn.log_softmax(logits)
    labels = np.array([[1, 2, 3, 4]])
    logit_padding = np.array([[0, 0, 0, 1]], dtype=np.float32)
    label_padding = np.array([[0, 0, 0, 1]], dtype=np.float32)

    loss, gradients = jax.value_and_grad(average_ctc_loss)(logprobs,
                                                           logit_padding,
                                                           labels,
                                                           label_padding)
    expected_loss = -(logprobs[0, 0, 1] + logprobs[0, 1, 2] + logprobs[0, 2, 3])
    self.assertAllClose(expected_loss, loss, rtol=0.01, atol=0.01)

    expected_gradients = np.array(jax.nn.softmax(logits))
    expected_gradients[0, 0, 1] -= 1.0
    expected_gradients[0, 1, 2] -= 1.0
    expected_gradients[0, 2, 3] -= 1.0
    expected_gradients[0, 3, :] = 0.0
    self.assertAllClose(expected_gradients, gradients, rtol=0.01, atol=0.01)

  def test_against_tf_ctc_loss(self):
    batchsize = 8
    timesteps = 150
    labelsteps = 25
    nclasses = 400
    logits = np.random.randn(batchsize, timesteps, nclasses)
    logprobs = jax.nn.log_softmax(logits)
    logprob_paddings = np.zeros((batchsize, timesteps))
    labels = np.random.randint(
        1, nclasses, size=(batchsize, labelsteps)).astype(np.int32)
    label_paddings = np.zeros((batchsize, labelsteps))

    inputs = [logprobs, logprob_paddings, labels, label_paddings]

    jax_per_seq, unused_aux_vars = ctc_objectives.ctc_loss(*inputs)
    tf_per_seq = tf_ctc_loss(*inputs)
    self.assertAllClose(jax_per_seq.squeeze(), tf_per_seq.squeeze())

    average_tf_ctc_loss = lambda *args: jnp.average(tf_ctc_loss(*args))
    jax_dloss = jax.grad(average_ctc_loss)
    tf_dloss = jax.grad(average_tf_ctc_loss)

    jax_dlogits = jax_dloss(*inputs)
    tf_dlogits = tf_dloss(*inputs)
    # Relative error check is disabled as numerical errors explodes when a
    # probability computed from the input logits is close to zero.
    self.assertAllClose(jax_dlogits, tf_dlogits, rtol=0.0, atol=1e-4)

  def test_against_tf_ctc_loss_with_paddings(self):
    batchsize = 8
    timesteps = 150
    labelsteps = 25
    nclasses = 400

    logits = np.random.randn(batchsize, timesteps, nclasses)
    logprobs = jax.nn.log_softmax(logits)
    logprob_lens = np.random.randint(25, timesteps - 3, size=(batchsize,))
    logprob_paddings = lengths_to_paddings(logprob_lens, timesteps)

    labels = np.random.randint(
        1, nclasses, size=(batchsize, labelsteps)).astype(np.int32)
    label_lens = np.random.randint(10, labelsteps, size=(batchsize,))
    label_paddings = lengths_to_paddings(label_lens, labelsteps)

    inputs = [logprobs, logprob_paddings, labels, label_paddings]

    jax_per_seq, _ = ctc_objectives.ctc_loss(*inputs)
    tf_per_seq = tf_ctc_loss(*inputs)
    self.assertAllClose(jax_per_seq.squeeze(), tf_per_seq.squeeze())

  def test_repeat_with_one_to_one_alignment(self):
    batch_size = 2
    labels = np.array([
        [1, 2, 2, 3],
        [2, 3, 4, 4],
    ])
    label_lens = np.array([4, 4])
    label_paddings = lengths_to_paddings(label_lens, 4)
    logits = np.random.randn(batch_size, 5, 5)
    logprobs = jax.nn.log_softmax(logits)
    logprob_paddings = np.zeros(logprobs.shape[:2])

    jax_per_seq, unused_aux_vars = ctc_objectives.ctc_loss(
        logprobs, logprob_paddings, labels, label_paddings)

    expected_alignment = [
        [1, 2, 0, 2, 3],
        [2, 3, 4, 0, 4],
    ]

    for n in range(batch_size):
      expected_loss = -sum(logprobs[n, t, k]
                           for t, k in enumerate(expected_alignment[n]))
      self.assertAllClose(
          jnp.array(expected_loss), jax_per_seq[n], rtol=0.01, atol=0.05)


class CollapseRemoveBlankLabelTest(test_utils.TestCase):

  def test_collapse_repeated(self):
    collapsed, new_seq_lengths = ctc_objectives.collapse_and_remove_blanks(
        labels=jnp.array([[1, 3, 3, 3, 0], [1, 4, 4, 4, 0], [4, 2, 2, 9, 4]]),
        seq_length=jnp.array([4, 5, 5]))
    self.assertArraysEqual(new_seq_lengths, jnp.array([2, 2, 4]))
    self.assertArraysEqual(
        collapsed, jnp.array([[1, 3, 0, 0, 0], [1, 4, 0, 0, 0], [4, 2, 9, 4,
                                                                 0]]))

  def test_collapse_repeated_preserve_dtypes(self):
    collapsed, new_seq_lengths = ctc_objectives.collapse_and_remove_blanks(
        labels=jnp.array([[1, 3, 3, 3, 0], [1, 4, 4, 4, 0], [4, 2, 2, 9, 4]],
                         dtype=jnp.int16),
        seq_length=jnp.array([4, 5, 5], dtype=jnp.int16))
    self.assertEqual(new_seq_lengths.dtype, jnp.int16)
    self.assertEqual(collapsed.dtype, jnp.int16)
    self.assertArraysEqual(new_seq_lengths,
                           jnp.array([2, 2, 4]).astype(jnp.int16))
    self.assertArraysEqual(
        collapsed,
        jnp.array([[1, 3, 0, 0, 0], [1, 4, 0, 0, 0], [4, 2, 9, 4,
                                                      0]]).astype(jnp.int16))

  def test_collapse_repeated_extra_padding(self):
    collapsed, new_seq_lengths = ctc_objectives.collapse_and_remove_blanks(
        labels=jnp.array([[1, 3, 3, 3, 0, 0, 0], [1, 4, 4, 4, 0, 1, 2],
                          [4, 2, 2, 9, 4, 0, 0]]),
        seq_length=jnp.array([4, 5, 5]))
    self.assertArraysEqual(new_seq_lengths, jnp.array([2, 2, 4]))
    self.assertArraysEqual(
        collapsed,
        jnp.array([[1, 3, 0, 0, 0, 0, 0], [1, 4, 0, 0, 0, 0, 0],
                   [4, 2, 9, 4, 0, 0, 0]]))

  def test_collapse_repeated_front_repeats(self):
    collapsed, new_seq_lengths = ctc_objectives.collapse_and_remove_blanks(
        labels=jnp.array([[1, 1, 1, 2, 2], [1, 1, 1, 2, 2], [1, 1, 1, 2, 2]]),
        seq_length=jnp.array([5, 4, 3]))
    self.assertArraysEqual(new_seq_lengths, jnp.array([2, 2, 1]))
    self.assertArraysEqual(
        collapsed, jnp.array([[1, 2, 0, 0, 0], [1, 2, 0, 0, 0], [1, 0, 0, 0,
                                                                 0]]))

  def test_collapse_repeated_all_labels_the_same(self):
    collapsed, new_seq_lengths = ctc_objectives.collapse_and_remove_blanks(
        labels=jnp.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
        seq_length=jnp.array([4, 5, 1]))
    self.assertArraysEqual(new_seq_lengths, jnp.array([1, 1, 1]))
    self.assertArraysEqual(
        collapsed, jnp.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0,
                                                                 0]]))

  def test_collapse_repeated_with_blanks(self):
    collapsed, new_seq_lengths = ctc_objectives.collapse_and_remove_blanks(
        labels=jnp.array([[1, 0, 0, 2, 3], [1, 0, 1, 1, 2], [1, 0, 1, 0, 1]]),
        seq_length=jnp.array([5, 5, 5]))
    self.assertArraysEqual(new_seq_lengths, jnp.array([3, 3, 3]))
    self.assertArraysEqual(
        collapsed, jnp.array([[1, 2, 3, 0, 0], [1, 1, 2, 0, 0], [1, 1, 1, 0,
                                                                 0]]))

  def test_different_blank_id(self):
    collapsed, new_seq_lengths = ctc_objectives.collapse_and_remove_blanks(
        labels=jnp.array([[1, -1, -1, 2, 3], [1, -1, 1, 0, 0],
                          [1, -1, 1, 1, 0]]).astype(jnp.int32),
        seq_length=jnp.array([5, 3, 4]),
        blank_id=-1)
    self.assertArraysEqual(new_seq_lengths, jnp.array([3, 2, 2]))
    self.assertArraysEqual(
        collapsed, jnp.array([[1, 2, 3, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0,
                                                                 0]]))

  def test_first_item_is_blank(self):
    collapsed, new_seq_lengths = ctc_objectives.collapse_and_remove_blanks(
        labels=jnp.array([[0, 0, 1, 0, 0, 2, 3], [0, 0, 1, 0, 1, 1, 2],
                          [0, 0, 1, 0, 1, 0, 1]]),
        seq_length=jnp.array([7, 7, 7]))
    self.assertArraysEqual(new_seq_lengths, jnp.array([3, 3, 3]))
    self.assertArraysEqual(
        collapsed,
        jnp.array([[1, 2, 3, 0, 0, 0, 0], [1, 1, 2, 0, 0, 0, 0],
                   [1, 1, 1, 0, 0, 0, 0]]))


if __name__ == '__main__':
  absltest.main()
