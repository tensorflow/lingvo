# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for label_lib.py."""

from lingvo import compat as tf
from lingvo.core import test_utils
from lingvo.tasks.milan import labels as label_lib
import numpy as np

# Alias for the special "ignore" label value to improve readability below.
X = label_lib.IGNORE_PAIR_LABEL


class ExamplePairLabelerTest(test_utils.TestCase):

  def testCrossModal(self):

    batch = {
        'image_ids': tf.constant([1, 2, 3], dtype=tf.int64),
        'caption_ids': tf.constant([4, 2, 2], dtype=tf.int64),
    }
    inputs = label_lib.ExamplePairs.WithinBatch(
        batch, query_modality='image', result_modality='caption')

    labeler = label_lib.ExamplePairLabeler(
        drop_pairs_that_match=['image_ids', 'caption_ids'])

    expected_labels = [
        [1, 0, 0],
        [0, 1, X],
        [0, X, 1],
    ]
    self.assertAllEqual(expected_labels, labeler(inputs))


class MultiItemPairLabelerTest(test_utils.TestCase):

  def testMultipleResultsPerExample(self):

    # Simple batch of 3 examples with 2 items per example in the result
    # modality.
    batch_size = 3
    results_per_example = 2

    inputs = label_lib.ExamplePairs.WithinBatch(
        batch=dict(some_feature=tf.range(batch_size)),
        query_modality='q',
        result_modality='r')

    def example_pair_labeler(_):
      return tf.constant([
          [1, 0, 0],
          [0, 1, X],
          [0, X, 1],
      ], dtype=tf.int64)

    multi_item_labeler = label_lib.MultiItemExampleWrapper(
        example_pair_labeler,
        modality_batch_shapes=dict(
            q=tf.TensorShape([None]),
            r=tf.TensorShape([None, results_per_example])))
    labels = multi_item_labeler(inputs)
    self.assertEqual([batch_size, batch_size, results_per_example],
                     labels.shape.as_list())
    # [3, 3, 2]
    expected_labels = [
        # pyformat: disable
        [[1, 1], [0, 0], [0, 0]],
        [[0, 0], [1, 1], [X, X]],
        [[0, 0], [X, X], [1, 1]]
        # pyformat: enable
    ]
    self.assertAllEqual(expected_labels, labels)

  def testIntraModalLabels(self):
    # Simulate a batch of 4 examples with 2 items each in the 'text' modality.
    batch_size = 4
    items_per_example = 2
    modality = 'text'
    modality_shape = tf.TensorShape([batch_size, items_per_example])
    inputs = label_lib.ExamplePairs.WithinBatch(
        batch=dict(some_feature=tf.range(batch_size)),
        query_modality=modality,
        result_modality=modality)

    def example_pair_labeler(_):
      return tf.constant([
          [1, 0, 0, X],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
          [X, 0, 0, 1],
      ])

    labeler = label_lib.MultiItemExampleWrapper(
        example_pair_labeler, modality_batch_shapes={modality: modality_shape})
    labels = labeler(inputs)
    self.assertEqual(modality_shape + modality_shape, labels.shape)
    # The pairwise labels actually have rank 4 (twice the rank of ids), but we
    # compare them in matrix form for easier inspection. There are 8 items
    # total. Each should have a positive label for every other item from the
    # same example. Self-pairs should be ignored (they are neither positive
    # nor negative pairs), as well as pairs from duplicated examples.
    self.assertAllEqual([
        [X, 1, 0, 0, 0, 0, X, X],
        [1, X, 0, 0, 0, 0, X, X],
        [0, 0, X, 1, 0, 0, 0, 0],
        [0, 0, 1, X, 0, 0, 0, 0],
        [0, 0, 0, 0, X, 1, 0, 0],
        [0, 0, 0, 0, 1, X, 0, 0],
        [X, X, 0, 0, 0, 0, X, 1],
        [X, X, 0, 0, 0, 0, 1, X],
    ], tf.reshape(labels, [8, 8]))


class MultiLabelLossTest(test_utils.TestCase):

  def testOneHotLabels(self):
    """Tests that the loss equals softmax CE when the labels are one hot."""
    num_classes = 400
    batch_size = 7
    label_indices = np.random.randint(0, num_classes, size=(batch_size, 3))
    labels = tf.one_hot(label_indices, depth=num_classes, dtype=tf.float32)
    logits = np.random.uniform(size=(batch_size, 3, num_classes)) * 10 + 1e7
    logits_tensor = tf.convert_to_tensor(logits, dtype=tf.float32)

    losses = label_lib.MultiLabelContrastiveLoss(labels, logits_tensor)
    expected = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits_tensor)
    self.assertAllClose(expected, losses)

  def testIgnoreLabels(self):
    """Tests that pairs marked IGNORE_PAIR_LABEL are excluded from the loss."""
    x = label_lib.IGNORE_PAIR_LABEL
    labels = tf.constant([
        [0, 1, 0, x],
        [1, 0, 0, 0],
        [x, 0, x, 1],
    ],
                         dtype=tf.float32)
    logits = tf.constant([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                         dtype=tf.float32)

    losses = label_lib.MultiLabelContrastiveLoss(labels, logits)
    expected_losses = tf.stack([
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=1, logits=[1.0, 2.0, 3.0]),
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=0, logits=[1.0, 2.0, 3.0, 4.0]),
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=1, logits=[2.0, 4.0]),
    ])
    self.assertAllClose(expected_losses, losses)

  def testManyHotLabels(self):
    batch_size = 7
    num_classes = 400
    num_positive = 5

    # To help keep the test simple, we put the positive labels on the
    # first 'num_positive' classes in every example.
    labels = np.zeros((batch_size, num_classes), np.float32)
    labels[:, :num_positive] = 1.0

    logits = np.random.uniform(size=labels.shape).astype(np.float32) * 10 + 1e7
    losses = label_lib.MultiLabelContrastiveLoss(
        tf.convert_to_tensor(labels, dtype=tf.float32),
        tf.convert_to_tensor(logits, dtype=tf.float32))

    # Verify that the multi-label loss is equivalent to the average softmax
    # cross entropy of each positive pair vs. all negative pairs.
    negative_pair_logits = logits[:, num_positive:]

    one_vs_all_labels = np.zeros((batch_size, num_classes - num_positive + 1),
                                 np.float32)
    one_vs_all_labels[:, 0] = 1

    expected_loss_terms = []
    for i in range(num_positive):
      one_vs_all_logits = np.concatenate(
          [logits[:, i:(i + 1)], negative_pair_logits], axis=1)
      expected_loss_terms.append(
          tf.nn.softmax_cross_entropy_with_logits(
              labels=one_vs_all_labels, logits=one_vs_all_logits))
    expected_loss = tf.add_n(expected_loss_terms) / num_positive
    self.assertAllClose(expected_loss, losses)

  def testNoPositiveLabels(self):
    """Tests that the loss is zero for slices with no positive label."""
    batch_size = 7
    num_classes = 400
    losses = label_lib.MultiLabelContrastiveLoss(
        labels=tf.zeros((batch_size, num_classes)),
        logits=tf.zeros((batch_size, num_classes)))
    self.assertAllClose(losses, tf.zeros(batch_size))


if __name__ == '__main__':
  tf.test.main()
