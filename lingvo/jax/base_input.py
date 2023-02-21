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
"""Base classes for the lingvo Jax input layers."""

import copy
from typing import List, Optional

from absl import logging
from lingvo.core import cluster_factory
from lingvo.core import datasource
from lingvo.jax import py_utils
from lingvo.jax import pytypes
import tensorflow.compat.v2 as tf

NestedMap = py_utils.NestedMap
NestedJTensor = pytypes.NestedJTensor
InstantiableParams = py_utils.InstantiableParams
ParamsT = pytypes.ParamsT


class BaseInputParams(InstantiableParams):
  """A convenient base type for the params of a dataset."""

  def __init__(self, cls) -> None:
    super().__init__(cls)
    self.Define('name', 'input', 'Name of this input dataset.')

    self.Define(
        'batch_size', None, 'The (Jax per process) Batch size. '
        'Each call to get_next() returns a batch with this '
        'batch size.')

    # Sharding behavior.
    self.Define(
        'num_infeed_hosts', 1,
        'Usually set to jax.process_count(). Implementation must '
        'ensure that the data is sharded into this many shard.')
    self.Define(
        'infeed_host_index', 0,
        'Usually set to jax.process_index(). Implementation must '
        'ensure that each instance returns a shard with this index.')

    # Deterministic randomness.
    self.Define(
        'input_random_seed', None,
        'If set, implementation must ensure that this is used to seed '
        'randomness, e.g. when shuffling in a deterministic manner.')

    self.Define(
        'reset_for_eval', False,
        'If set, eval will continue until tf.errors.OutOfRange is raised, '
        'and reset() will called for each eval. Implementation must ensure that'
        ' all variant p.infeed_host_index instances raise after the same number'
        ' of calls to get_next() to ensure synchronization across hosts. If not'
        ' set, get_next() must never raise.')
    self.Define(
        'eval_loop_num_batches', 1,
        'Num of batches to process per eval loop. Must be >= 1. This value'
        ' is ignored if reset_for_eval is set True, in which case, this value'
        ' is dynamically determined by the number of available batches. If '
        ' reset_for_eval is set to False, then each eval loop will process'
        ' this many batches. Metrics over those batches will be aggregated'
        ' and then reported.')
    self.Define('is_training', False,
                'Whether or not this dataset is used for model traning.')


class BaseInput:
  """Base class for Jax input classes.

  During Lingvo Jax's train, on each host an input instance will be
  created (input_p.Instantiate()), and then get_next() is iteratively
  called in eager mode to generate one batch of data for each step
  of train/eval/etc.

  If supported, for eval, reset() is called after each eval step.
  See p.reset_for_eval below.

  A tf.data based input should inherit this class directly and implement
  get_next() and reset(). For an example of how to handle sharding for both
  training and eval data, please refer to the implementation of
  TFRecordBertInput at tasks/lm/input_generator.py.

  If there is already an Lingvo TF input generator that one would like to
  use directly, please use LingvoInputAdaptor below.
  """
  _VALIDATE_BATCH_SIZE_NOT_NONE = True

  @classmethod
  def Params(cls) -> InstantiableParams:  # pylint:disable=invalid-name
    """Common Params for all inputs."""
    return BaseInputParams(cls)

  def __init__(self, p: ParamsT) -> None:
    if self._VALIDATE_BATCH_SIZE_NOT_NONE and (p.batch_size is None):
      raise ValueError('Must specify p.batch_size.')
    self._params = p.Copy()

  @property
  def params(self) -> ParamsT:
    return self._params

  def get_next(self) -> NestedJTensor:
    raise NotImplementedError

  def reset(self) -> None:
    pass

  def ids_to_strings(self,
                     ids: pytypes.NpTensor,
                     lengths: pytypes.NpTensor,
                     key: Optional[str] = None) -> List[str]:
    """Converts int ids into strings.

    Args:
      ids: A matrix of shape [batch, seqlen], each row is a sequence to be
        converted.
      lengths: A vector of shape [batch]. lens[i] is the sequence length of the
        i-th row. Only the first lens[i] tokens in ids[i, :] are valid tokens.
      key: Optional argument to specify whether a tokenizer to use is the source
        or target. This is useful for example in a sequence model where the
        source and targets have different tokenizers. For the source corpus the
        key should be `src` while for the target corpus the key should be `tgt`.

    Returns:
      A list strings of shape [batch]. The converted texts.
    """
    raise NotImplementedError


class LingvoInputAdaptor(BaseInput):
  """Syntactic sugar for adapting a Lingvo style input for Jax.

  This should be able to wrap any Lingvo TF input generator to be used in
  Lingvo Jax. Remember to set `p.is_training=True` on the training dataset.

  Some usage caveats below.

  For eval, `p.num_samples` or other similar params like samples_per_summary are
  completely ignored by Lingvo Jax. Caller should instead set `p.num_batches` to
  (p.num_samples // batch_size) with `p.reset_for_eval=True` so that each eval
  step reads (approximately) one epoch of eval data. This might not be needed if
  the input already is finite (e.g. with p.repeat_count=1).

  When multiple infeed hosts are used, one must take care to ensure that the
  Lingvo input either already uses InfeedContextScope for proper sharding, or
  alternatively do not use the same random seed on all hosts. In other words,
  one must avoid the failure case where each host emits identical training data.
  See also p.allow_fixed_file_random_seed below.
  """
  _VALIDATE_BATCH_SIZE_NOT_NONE = False
  _VALIDATE_BATCH_SIZE_NONE = True

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('input', None, 'Params of a Lingvo input generator.')
    p.Define(
        'num_batches', None,
        'If specified and positive, raises tf.errors.OutOfRange after this many'
        ' batches have been produced. This forces a raise after get_next() is '
        'called this many times, to support p.reset_for_eval=True.')
    p.Define(
        'allow_fixed_file_random_seed', False,
        'If not set, disallows a fixed, non-zero p.input.file_random_seed. '
        'We disallow by default to avoid having identical input batches across '
        'different infeed hosts. If set, random seeds are adjusted by '
        'p.infeed_host_index to ensure different random seeds.')
    p.Define(
        'cluster_do_eval', False,
        'Whether to set cluster.do_eval to True for non-training data. '
        'Note that if set to True, this will change '
        'cluster.require_sequential_input_order to True as a result. '
        'Ignored  when p.is_training is True.')
    return p

  def __init__(self, p):
    if self._VALIDATE_BATCH_SIZE_NONE and p.batch_size is not None:
      raise ValueError('LingvoInputAdaptor does not support p.batch_size. '
                       'Please specify batch size on p.input, e.g. with '
                       'p.input.bucket_batch_limit = [4] or '
                       'p.input.args.batch=4, depeding the Lingvo input '
                       f'used. Currently: p.batch_size={p.batch_size}, '
                       'it must be None.')
    super().__init__(p)
    self._cluster = copy.deepcopy(cluster_factory.Current())
    # For Lingvo's Cluster context that may impact the behavior of this input
    # generator, we always set use_tpu to True, and optionally set do_eval
    # for non-training data when configured to do so. All other Cluster params
    # use the default value.
    self._cluster.params.xla_device = 'tpu'
    self._cluster.params.enable_asserts = False
    # This indirectly sets cluster.require_sequential_input_order as well.
    self._cluster.params.do_eval = (not p.is_training and p.cluster_do_eval)
    self._initialize()

  def _initialize(self) -> None:
    """Initializes the relevant fields of this adaptor input."""
    p = self.params
    if hasattr(p.input, 'file_random_seed') and p.input.file_random_seed:
      if not p.allow_fixed_file_random_seed:
        raise ValueError(
            'Training data using fixed non-zero file_random_seed: '
            f'p.input.file_random_seed={p.input.file_random_seed}. '
            'This means each host *might* infeed identical batches. You can set '
            'p.input.file_random_seed = 0, or if certain this is intended, '
            'suppress this error by setting p.allow_fixed_file_random_seed = '
            'True.')
      # Make sure each host uses a different random seed.
      p.input.file_random_seed += p.infeed_host_index
    # We make self.input public so that users can access its methods like
    # IdsToStrings if needed.
    with py_utils.infeed_context_scope(
        infeed_host_index=p.infeed_host_index,
        num_infeed_hosts=p.num_infeed_hosts), self._cluster:
      self.input = p.input.Instantiate()

    if hasattr(self.input, 'datasource') and isinstance(
        self.input.datasource, datasource.TFDatasetSource):
      # For the special case when the input is implemented by a tf.data.Dataset,
      # call eagerly. Using tf.function may result in returning duplicate
      # batches.
      self._get_next_fn = self._get_batch
    else:
      self._get_next_fn = tf.function(self._get_batch)
    self._num_batches_produced = 0

  def _get_batch(self) -> NestedMap:
    p = self.params
    with py_utils.infeed_context_scope(
        infeed_host_index=p.infeed_host_index,
        num_infeed_hosts=p.num_infeed_hosts), self._cluster:
      ret = self.input.GetPreprocessedInputBatch()
    # Remove unsupported string (byte) array from input.
    return ret.Filter(lambda v: v.dtype != tf.string)

  def get_next(self) -> NestedJTensor:
    p = self.params
    if p.num_batches is not None and p.num_batches > 0:
      if self._num_batches_produced >= p.num_batches:
        raise tf.errors.OutOfRangeError(
            node_def=None,
            op=None,
            message=f'num_batches exceeding {self._num_batches_produced}')
      self._num_batches_produced += 1
    ret = self._get_next_fn()
    return tf.nest.map_structure(lambda x: x.numpy(), ret)

  def reset(self) -> None:
    if hasattr(self.input, 'datasource') and isinstance(
        self.input.datasource, datasource.TFDatasetSource):
      self.input.datasource.Reset()
      # reset counter to 0.
      self._num_batches_produced = 0
      return
    # reinstantiate the input and retrace self._get_batch.
    self._initialize()

  def ids_to_strings(self,
                     ids: pytypes.NpTensor,
                     lengths: pytypes.NpTensor,
                     key: Optional[str] = None) -> List[str]:
    """Converts int ids into strings."""
    bytes_list = self.input.IdsToStrings(ids, lengths, key=key).numpy()
    return [b.decode('utf-8') for b in bytes_list]


class LingvoInputAdaptorNewBatchSize(LingvoInputAdaptor):
  """A similar adapter as LingvoInputAdaptor supporting a new batch size.

  LingvoInputAdaptor uses the batch size specified by the underlying Lingvo
  input. This class, however, allows specifying a smaller p.batch_size.
  This can be useful when the Lingvo input expects a large batch size,
  but the user wants a smaller batch size, e.g. when the Lingvo input uses
  a fixed packing factor to do packing, which can more efficiently pack with
  more data.

  We require that the batch size of the underlying Lingvo input must divide
  p.batch_size. Internally this class acts as a cache, retrieving the large
  batches from the parent class size, and consuming it by slicing it to the
  smaller batch size specified by the user.

  Example usage:
      p = ChangeBatchSizeInput.Params().Set(...)
      p.input.packing_factor = 3.5
      p.input.bucket_batch_limit = [4096]
      p.batch_size = 4
  """
  _VALIDATE_BATCH_SIZE_NOT_NONE = True
  _VALIDATE_BATCH_SIZE_NONE = False

  def __init__(self, p):
    super().__init__(p)
    self._current_batch = super().get_next()
    self._inner_batch_size = next(iter(self._current_batch.values())).shape[0]  # pytype: disable=attribute-error  # jax-ndarray
    logging.info(
        'The wrapped Lingvo input has batch size %d, the actual input '
        'has batch size %d.', self._inner_batch_size, p.batch_size)
    if self._inner_batch_size % p.batch_size != 0:
      raise ValueError(f'Lingvo input batch size {self._inner_batch_size} '
                       'must be a multiple of p.batch_size={p.batch_size}.')
    self._current_batch_index = 0

  def get_next(self):
    p = self.params
    if self._current_batch_index >= self._inner_batch_size:
      self._current_batch = super().get_next()
      self._current_batch_index = 0

    def _get_subrows(b):
      start = self._current_batch_index
      return b[start:start + p.batch_size]

    ret = tf.nest.map_structure(_get_subrows, self._current_batch)
    self._current_batch_index += p.batch_size
    return ret

  def reset(self):
    super().reset()
    self._current_batch = super().get_next()
    self._current_batch_index = 0
