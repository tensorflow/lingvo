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
"""Utilities for labeling batches and computing loss.

In dual encoder tasks, the concept of "labels" is more complicated than in
traditional supervised learning tasks. This is for two main reasons:

 - Labels have to be computed on the fly. This is because dual encoder labels
   describe pairwise relationships between (and within) examples in a batch.
   As such, they aren't simply fixed annotations attached to each example.
 - Determining some of these relationships requires knowledge about the dataset
   format and what its features mean. There is no one-size-fits-all computation
   that can generate labels for all datasets.

Milan's solution is to encapsulate all the logic required to compute labels
in dataset-specific *label functions*. Label function instances typically
assume input examples have a very particular format and interpretation. Upstream
code is responsible for ensuring the labeler only gets called on the specific
dataset it describes. Implementations in this file are templates that can be
configured for many common datasets.

The structure of datasets implicitly defines some relationships -- but
only some. In particular, *within-example* relationships are assumed to be
positive, but *between-example* relationships are a priori unknown. For
instance, consider an image-captioning dataset that stores an image and caption
per example. Within each example, we assume the caption describes the image and
vice-versa; but, without any extra information, we know nothing about how images
or captions from different examples relate to each other.

In practice, most between-example pairs are interpreted as negative (unrelated).
However, there are at least two commonly occurring cases where a negative label
would be incorrect:

 - Linked positive examples: Some datasets distribute positive pairs among
   multiple examples. For instance, in a captioning dataset with N captions per
   image, co-captions might be flattened into N separate (image + caption)
   examples. (Side note: Milan supports "multi-item" examples, which makes such
   flattening unnecessary.)

 - Duplicates: Training batches sampled with replacement may contain duplicates,
   meaning examples that are identical copies of others, either in whole or
   in a modality being modeled.
   Note the chance of getting exact duplicates generally increases with the
   global batch size and the number of independent TPU infeeds (usually one per
   core or TPU host).

While a negative label is incorrect in both cases, neither can be identified
through generic, dataset-independent means. Instead, some knowledge of the
dataset is necessary. For instance, if we know there is an `example_id` feature
that uniquely identifies each example, then deduplication of examples becomes
easy.


Label functions
---------------
In general, a label function is just a function with signature
`ExamplePairs` -> labels Tensor.
The `ExamplePairs` input specifies the examples to label and the query and
result modalities. Other details like feature names, a method for detecting
duplicates, etc. are expected to be captured within the function itself.

Labels are trinary-valued: they can either be positive (1), negative (0), or a
special `IGNORE_PAIR_LABEL` value that indicates the query-result pair should be
ignored during training. The last can be used to mark duplicates, or more
generally cases where the query-result relationship is unknown or should not be
used.

Implementations in this file are templates that can be configured for many
common datasets.
"""

from typing import Callable, Iterable, Mapping, Union

import attr
from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.tasks.milan import tpu_utils
from lingvo.tasks.milan import utils


@attr.s
class ExamplePairs:
  """Defines query-result example pairs to be labeled.

  `query_examples` and `result_examples` are the input batches of the
  queries and results being labeled. These batches contain full examples, as
  read from the original dataset, after any applicable preprocessing.
  `result_examples` is always a superset of `query_examples`. It contains
  identical copies of all `query_examples` plus zero or more other randomly
  chosen examples.

  `correspondences` indicates where each of the `query_examples`
  was copied inside `result_examples` when the `ExamplePairs` was constructed.
  It has shape [query_batch_size, result_batch_size] and is typically a
  rectangular (padded) identity matrix.

  `query_modality` and `result_modality` are strings identifying the query
  and result modalities. These define which slice or view of each
  example constitutes a query or result.
  """

  query_examples = attr.ib()
  query_modality: str = attr.ib()
  result_examples = attr.ib()
  result_modality: str = attr.ib()
  correspondences: tf.Tensor = attr.ib()

  # Number of examples in `query_examples`, inferred from tensor shapes.
  query_batch_size: int = attr.ib(init=False)
  # Number of examples in `result_examples`, inferred from tensor shapes.
  result_batch_size: int = attr.ib(init=False)

  @classmethod
  def WithinBatch(cls, batch, **kwargs) -> 'ExamplePairs':
    """Creates an instance representing all example pairs within `batch`.

    Args:
      batch: Dict of input examples; this same set of examples represents both
        the query_examples and result_examples.
      **kwargs: kwargs to forward to ExamplePairs constructor.

    Returns:
      `ExamplePairs`
    """
    correspondences = tf.eye(utils.InferBatchSize(batch), dtype=tf.bool)
    return ExamplePairs(
        query_examples=batch,
        result_examples=batch,
        correspondences=correspondences,
        **kwargs)

  @classmethod
  def BetweenLocalAndGlobalBatches(cls, local_batch,
                                   **kwargs) -> 'ExamplePairs':
    """Creates an instance representing (local, global) example pairs."""
    local_batch = py_utils.NestedMap(local_batch)
    global_batch = tpu_utils.ConcatenateAcrossReplicas(local_batch)
    correspondences = tf.eye(
        utils.InferBatchSize(local_batch),
        utils.InferBatchSize(global_batch),
        dtype=tf.bool)
    return ExamplePairs(
        query_examples=local_batch,
        result_examples=global_batch,
        correspondences=correspondences,
        **kwargs)

  def __attrs_post_init__(self):
    self.query_batch_size = utils.InferBatchSize(self.query_examples)
    self.result_batch_size = utils.InferBatchSize(self.result_examples)

    self.correspondences.shape.assert_is_compatible_with(
        [self.query_batch_size, self.result_batch_size])
    assert self.correspondences.dtype == tf.bool


# Type describing general label functions.
LabelFnType = Callable[[ExamplePairs], tf.Tensor]

IGNORE_PAIR_LABEL = -1


class ExamplePairLabeler:
  """A simple labeler for single-item examples.

  This labeler provides a mechanism to drop (ignore) example pairs based on
  equality of scalar features. Typically these features are integer ids
  (or similar) that uniquely identify the example or one of its modalities.
  Dropping (distinct) pairs with the same ID is a simple form of deduplication.

  More precisely:
   - Queries and results from the same example (according to `correspondences`)
     always get a positive label.
   - By default, all other pairs get a negative label.
   - Negative labels are replaced by IGNORE if the examples are equal w.r.t.
     any feature named in `drop_pairs_that_match`. These features should be
     scalars, one per example.

  This labeler assumes examples only have one item per modality (e.g. one image
  and one caption), and is only suitable for datasets that satisfy that
  condition.
  """

  def __init__(self, drop_pairs_that_match: Union[str, Iterable[str]] = ()):
    if isinstance(drop_pairs_that_match, str):
      drop_pairs_that_match = [drop_pairs_that_match]
    self._drop_on_match = list(drop_pairs_that_match)

  def __call__(self, inputs: ExamplePairs) -> tf.Tensor:
    drop_masks = []
    for feature_name in self._drop_on_match:
      query_ids = inputs.query_examples.get(feature_name)
      result_ids = inputs.result_examples.get(feature_name)
      if query_ids is None:
        raise ValueError('No feature {} in query batch'.format(feature_name))
      if result_ids is None:
        raise ValueError('No feature {} in result batch'.format(feature_name))
      query_ids.shape.assert_is_compatible_with([inputs.query_batch_size])
      result_ids.shape.assert_is_compatible_with([inputs.result_batch_size])
      drop_masks.append(tf.equal(query_ids[:, None], result_ids[None, :]))

    labels = tf.cast(inputs.correspondences, dtype=tf.int32)
    if drop_masks:
      any_drop_condition_met = tf.reduce_any(drop_masks, axis=0)
      labels = _IgnorePairsWhere(
          ~inputs.correspondences & any_drop_condition_met, labels)
    return labels


def _BroadcastExamplePairLabelsToAllItemPairs(example_pair_labels: tf.Tensor,
                                              queries_shape: tf.TensorShape,
                                              results_shape: tf.TensorShape):
  """Propagates each example-pair label to all pairs of their items.

  Args:
    example_pair_labels: Labels tensor for example pairs, shape
      [query_batch_size, result_batch_size].
    queries_shape: Batch shape of the query examples. Must start with
      `query_batch_size`.
    results_shape: Batch shape of the query examples. Must start with
      `result_batch_size`.

  Returns:
    A labels tensor with shape `queries_shape + results_shape`.
  """
  example_pair_labels.shape.assert_has_rank(2)
  queries_shape.assert_is_fully_defined()
  results_shape.assert_is_fully_defined()
  # Expand [q, r] to [q, 1, ..., r, 1, ...]
  all_slice = slice(None, None, None)
  reshape_slice = ((all_slice,) + (None,) * (queries_shape.rank - 1) +
                   (all_slice,) + (None,) * (results_shape.rank - 1))
  return tf.broadcast_to(example_pair_labels[reshape_slice],
                         queries_shape + results_shape)


def _IgnorePairsWhere(condition, labels):
  return tf.where(condition, tf.broadcast_to(IGNORE_PAIR_LABEL, labels.shape),
                  labels)


class MultiItemExampleWrapper:
  """Adapts a labeler for single-item examples into one for multi-item examples.

  `MultiItemExampleWrapper` wraps a labeler for single-item examples (one that
  assumes examples have a single item per modality), so that it can label
  multi-item examples (i.e. where each example holds Q >= 1 query and R >= 1
  result items). The former produces a label for each of
  query_batch_size * result_batch_size pairs; the latter produces one for each
  of query_batch_size * Q * result_batch_size * R pairs.

  `modality_shapes` specifies the number of items per modality in each example
  (Q and R, in this example).

  This wrapper employs some simple semantics to expand labels:

    - All item pairs inherit the label of the parent example pair.
    - When the query and result modalities are the same (intra-modal retrieval),
      item self pairs are labeled "ignore". This is so items aren't used as
      their own targets during training.
  """

  def __init__(self, example_pair_labeler: LabelFnType,
               modality_batch_shapes: Mapping[str, tf.TensorShape]):
    self._example_pair_labeler = example_pair_labeler
    self._modality_batch_shapes = dict(modality_batch_shapes)

  def __call__(self, inputs: ExamplePairs):
    """Labels item pairs in `inputs`."""

    # Generate labels for the example pairs. If examples only have one item in
    # each modality, everything else below is a no-op.
    example_pair_labels = self._example_pair_labeler(inputs)
    example_pair_labels.shape.assert_is_compatible_with(
        [inputs.query_batch_size, inputs.result_batch_size])

    query_batch_shape = utils.ResolveBatchDim(
        self._modality_batch_shapes[inputs.query_modality],
        inputs.query_batch_size)
    result_batch_shape = utils.ResolveBatchDim(
        self._modality_batch_shapes[inputs.result_modality],
        inputs.result_batch_size)

    # Broadcast example-level labels to all pairs of their items.
    item_pair_labels = _BroadcastExamplePairLabelsToAllItemPairs(
        example_pair_labels, query_batch_shape, result_batch_shape)

    if inputs.query_modality == inputs.result_modality:
      # Intra-modal retrieval. Give self pairs the "ignore" label so that items
      # aren't used as their own targets during training.
      # For this case we require the batch shape to be rank 2.
      #  - If rank == 1, each example only contains a single item, so the only
      #    within-example pairs are the self pairs. Once these are dropped,
      #    there are typically no positively labeled pairs, meaning there is
      #    no training signal. (The exception is if two *distinct* examples
      #    are given a positive label.) Rather than hoping the trainer will do
      #    something smart in this weird corner case, we simply die if
      #    rank == 1.
      #  - Any rank > 1 is sufficient to get around the above problem, but
      #    currently there's no use case for any ranks above 2.
      query_batch_shape.assert_has_rank(2)
      n = tf.compat.dimension_value(query_batch_shape[1])
      assert n > 1
      # Item self-pairs are those at indices [q, i, r, j] where
      #   - Examples q and r are the same example
      #   - i == j refer to the same item in the example
      is_item_self_pair = (
          # [q, 1, r, 1]
          inputs.correspondences[:, None, :, None]
          # [1, n, 1, n]
          & tf.eye(n, dtype=tf.bool)[None, :, None, :])
      item_pair_labels = _IgnorePairsWhere(is_item_self_pair, item_pair_labels)
    return item_pair_labels


def MultiLabelContrastiveLoss(labels, logits, axis: int = -1):
  """Computes a multi-label generalization of softmax cross entropy loss.

  This loss generalizes softmax cross entropy in the following sense.

  - If `labels` are one-hot (over `axis`), this loss is equivalent to softmax
    cross entropy. Note in this case the per-example loss can be interpreted as
    -log(p(positive_class)). Here p() is a distribution of over C classes,
    namely 1 positive class and C-1 negative classes.

  - In general, if `labels` are N-hot, this function computes the loss
    `sum_i{ -log(p_i(positive_class_i)) } / N`
    where p_i() is a distribution over the i'th positive class and the C-N
    negative classes.

  - As a special case, this function returns a loss of zero for any slice of
    `labels` that contains *no* positive label.

  Note unlike `tf.nn.softmax_cross_entropy_with_logits()`, this function does
  not support "soft" labels. Positive and negative labels must be represented as
  1 and 0, respectively. Setting a label to any other value causes the example-
  class pair to be ignored in the loss calculation. This is intended as a
  feature, to give callers fine-grained control over which pairs are used in
  the loss.

  Args:
    labels: Tensor of labels. Must have the same shape as `logits`.
    logits: Tensor of logits (scores). Must have the same shape as `labels`.
    axis: The class dimension, i.e. the one over which probability distributions
      are normalized.

  Returns:
    A Tensor of per-example losses. Has the same type as `logits`, and the same
    shape, except without `axis`.

    Typically `labels` and `logits` are both [batch_size, num_classes], in
    which case the result is [batch_size].
  """

  labels.shape.assert_is_compatible_with(logits.shape)

  # Set logits for non-negative pairs to -inf so they're effectively ignored.
  is_negative_pair = tf.equal(labels, 0)
  negative_pair_logits = tf.where(is_negative_pair, logits,
                                  tf.broadcast_to(float('-inf'), logits.shape))
  # Compute binary logits, log(p / (1-p)). Shift inputs by the max negative-pair
  # score to improve numerical precision. The reason is that
  #   tf.reduce_logsumexp(x) == max(x) + log(sum_i(exp(x[i] - max(x))))
  # and if the max is sufficiently large the second term disappears as round-off
  # error.
  adjustment = tf.reduce_max(negative_pair_logits, axis=axis, keepdims=True)

  binary_logits = (logits - adjustment) - tf.reduce_logsumexp(
      negative_pair_logits - adjustment, axis=axis, keepdims=True)

  # Accumulate the losses of each positive sample vs. all negative ones. Note
  # -log_sigmoid == sigmoid_cross_entropy_with_logits in the special case that
  # all labels are 1.
  is_positive_pair = tf.cast(tf.equal(labels, 1), binary_logits.dtype)
  losses = is_positive_pair * -tf.math.log_sigmoid(binary_logits)
  num_positives = tf.reduce_sum(is_positive_pair, axis=axis)
  return tf.reduce_sum(losses, axis=axis) / tf.math.maximum(num_positives, 1)
