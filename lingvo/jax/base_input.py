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
"""Base classes for the lingvo Jax input layers."""

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

  @classmethod
  def Params(cls) -> InstantiableParams:  # pylint:disable=invalid-name
    """Common Params for all inputs."""
    return BaseInputParams(cls)

  def __init__(self, p: ParamsT) -> None:
    if p.batch_size is None:
      raise ValueError('Must specify p.batch_size')
    self._params = p.Copy()

  @property
  def params(self) -> ParamsT:
    return self._params

  def get_next(self) -> NestedJTensor:
    raise NotImplementedError

  def reset(self) -> None:
    pass


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
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('input', None, 'Params of a Lingvo input generator.')
    p.Define(
        'num_batches', None,
        'If specified and positive, raises tf.errors.OutOfRange after this many'
        ' batches have been produced. This forces a raise after get_next() is '
        'called this many times, to support p.reset_for_eval=True.')
    return p

  def __init__(self, p):
    p.batch_size = -1  # unused
    super().__init__(p)
    self._initialize()

  def _initialize(self) -> None:
    """Initializes the relevant fields of this adaptor input."""
    p = self.params
    # We make self.input public so that users can access its methods like
    # IdsToStrings if needed.
    with py_utils.infeed_context_scope(
        infeed_host_index=p.infeed_host_index,
        num_infeed_hosts=p.num_infeed_hosts):
      self.input = p.input.Instantiate()

    if hasattr(self.input, 'datasource') and isinstance(
        self.input.datasource, datasource.TFDatasetSource):
      # For the special case when the input is implemented by a tf.data.Dataset,
      # use it directly. Otherwise roundtrip adaptions may result in returning
      # duplciate batches.
      self._get_next_fn = self.input.datasource.GetNext
    else:
      self._get_next_fn = tf.function(self._get_batch)
    self._num_batches_produced = 0

  def _get_batch(self) -> NestedMap:
    p = self.params
    with py_utils.infeed_context_scope(
        infeed_host_index=p.infeed_host_index,
        num_infeed_hosts=p.num_infeed_hosts):
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
      return
    # reinstantiate the input and retrace self._get_batch.
    self._initialize()
