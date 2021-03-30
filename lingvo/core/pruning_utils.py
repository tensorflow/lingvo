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
"""Utilities for pruning."""

import lingvo.compat as tf
from lingvo.core import py_utils

from model_pruning.python import pruning


def AddToPruningCollections(weight,
                            mask,
                            threshold,
                            gradient=None,
                            old_weight=None,
                            old_old_weight=None):
  """Add mask, threshold, and weight vars to their respective collections."""
  if mask not in tf.get_collection(pruning.MASK_COLLECTION):
    tf.add_to_collection(pruning.WEIGHT_COLLECTION, weight)
    tf.add_to_collection(pruning.MASK_COLLECTION, mask)
    tf.add_to_collection(pruning.THRESHOLD_COLLECTION, threshold)

    # Add gradient, old_weight, and old_old_weight to collections approximating
    # gradient and hessian, where old_weight is the weight tensor one step
    # before and old_old_weight is the weight tensor two steps before.
    if gradient is not None:
      assert old_weight is not None
      assert old_old_weight is not None
      tf.add_to_collection(pruning.WEIGHT_GRADIENT_COLLECTION, gradient)
      tf.add_to_collection(pruning.OLD_WEIGHT_COLLECTION, old_weight)
      tf.add_to_collection(pruning.OLD_OLD_WEIGHT_COLLECTION, old_old_weight)


def UsePruningInterface(pruning_hparams_dict):
  if not pruning_hparams_dict:
    return False
  prune_option = pruning_hparams_dict.get('prune_option', 'weight')
  return prune_option == 'compression'


def ApplyCompression(params):
  """Returns a bool indicating whether compression library is to be used."""
  return not params.apply_pruning and params.pruning_hparams_dict is not None


def FlatEmbeddingLookup(emb_table,
                        flat_ids,
                        vocab_size,
                        matmul_axis=1,
                        fprop_mode='matmul'):
  """Performs embedding lookup operation.

  Args:
    emb_table: tf.Tensor containing the embedding vectors.
    flat_ids: tf.Tensor of shape (number_ids,).
    vocab_size: vocabulary size of the embedding table, int.
    matmul_axis: the axis of flat_ids that is used for matmul, int.
    fprop_mode: embedding lookup option, should be 'matmul' or 'gather'.

  Returns:
    Embedding lookup result.
  """
  if fprop_mode == 'matmul':
    lhs = tf.equal(
        tf.expand_dims(flat_ids, matmul_axis),
        tf.range(vocab_size, dtype=flat_ids.dtype))
    return tf.matmul(tf.cast(lhs, emb_table.dtype), emb_table)
  elif fprop_mode == 'gather':
    return tf.nn.embedding_lookup(emb_table, flat_ids)
  else:
    raise ValueError(
        'FlatEmbeddingLookup(): fprop_mode {} is not supported.'.format(
            fprop_mode))


class PruningOp(object):
  """A pruning op object.

  This class encapsulates the methods that are needed for pruning (and
  compression) so that both pruning and compression can be called in lingvo
  using the same API.
  """

  _pruning_hparams_dict = {}
  _global_step = None
  _pruning_obj = None
  _pruning_hparams = None

  @classmethod
  def Setup(cls, pruning_hparams_dict, global_step):  # pylint:disable=invalid-name
    """Set up the pruning op with pruning hyperparameters and global step.

    Args:
      pruning_hparams_dict: a dict containing pruning hyperparameters;
      global_step: global step in TensorFlow.
    """
    if cls._pruning_obj is not None:
      pass
    assert pruning_hparams_dict is not None
    assert isinstance(pruning_hparams_dict, dict)
    cls._pruning_hparams_dict = pruning_hparams_dict
    cls._global_step = global_step
    cls._pruning_hparams = pruning.get_pruning_hparams().override_from_dict(
        pruning_hparams_dict)
    cls._pruning_obj = pruning.Pruning(
        spec=cls._pruning_hparams, global_step=global_step)

  # pylint:disable=unused-argument
  @classmethod
  def ApplyPruning(cls, pruning_hparams_dict, lstmobj, weight_name, wm_pc,
                   dtype, scope):
    if not cls._pruning_obj:
      train_global_step = py_utils.GetGlobalStep()
      if train_global_step is None:
        train_global_step = 0
      cls.Setup(pruning_hparams_dict, global_step=train_global_step)
    return None
  # pylint:enable=unused-argument

  @classmethod
  def GetMixResult(cls, theta, concat, lstmobj):  # pylint:disable=unused-argument
    """Compute the mix result.

    Args:
      theta: a theta object in the LSTM cells;
      concat: Tensor, concat of previous output and current state vector;
      lstmobj: a LSTM cell object.

    Returns:
      result Tensor.

    Raises:
      NotImplementedError if prune_option is not 'weight',
      'first_order_gradient', or 'second_order_gradient'.
    """
    return tf.matmul(
        concat,
        lstmobj.QWeight(tf.multiply(theta.wm, theta.mask, 'masked_weight')))

  @classmethod
  def GetLastCompressionOp(cls):
    # Currently only pruning is supported and no compression op should be
    # returned.
    return None

  @classmethod
  def GetEmbeddingLookupResult(cls,
                               theta,
                               flat_ids,
                               fprop_mode='gather',
                               layer_obj=None):  # pylint:disable=unused-argument
    """Get embedding lookup result.

    Args:
      theta: a theta object in the LSTM cells;
      flat_ids: tf.Tensor of shape (number_ids,).
      fprop_mode: embedding lookup option, should be 'matmul' or 'gather'.
      layer_obj: an EmbeddingLayer object.

    Returns:
      Embedding lookup result.

    Raises:
      NotImplementedError if prune_option is not 'weight',
      'first_order_gradient', or 'second_order_gradient'.
    """
    if cls._pruning_hparams.prune_option in [
        'weight', 'first_order_gradient', 'second_order_gradient'
    ]:
      return FlatEmbeddingLookup(
          emb_table=theta.wm,
          flat_ids=flat_ids,
          vocab_size=theta.wm.shape[0],
          fprop_mode=fprop_mode)
    else:
      raise NotImplementedError()

  @classmethod
  def GetPruningUpdate(cls):  # pylint:disable=invalid-name
    # for pruning, it returns pruning_obj.conditional_mask_update_op()
    return cls._pruning_obj.conditional_mask_update_op()

  @classmethod
  def ApplyTensorflowUpdate(cls):  # pylint:disable=invalid-name
    if not cls._pruning_obj:
      return False
    hparams = cls._pruning_obj.get_spec()
    return (hparams.prune_option in [
        'weight', 'first_order_gradient', 'second_order_gradient'
    ] or hparams.update_option == 0 or hparams.update_option == 2)

  @classmethod
  def ApplyPythonUpdate(cls):
    if not cls._pruning_obj:
      return False
    hparams = cls._pruning_obj.get_spec()
    return hparams.update_option == 1 or hparams.update_option == 2

  @classmethod
  def ApplyTensorflowAndPythonUpdate(cls):
    """Returns True if both Tensorflow and Python updates need to run."""
    if not cls._pruning_obj:
      return False
    hparams = cls._pruning_obj.get_spec()
    return hparams.update_option == 2

  @classmethod
  def RunPythonUpdate(cls, session, global_step):  # pylint:disable=unused-argument
    return
