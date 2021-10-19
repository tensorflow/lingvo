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

from lingvo.jax import py_utils
from lingvo.jax import pytypes
import tensorflow.compat.v2 as tf

NestedMap = py_utils.NestedMap
NestedJTensor = pytypes.NestedJTensor
InstantiableParams = py_utils.InstantiableParams
ParamsT = pytypes.ParamsT


class BaseInput:
  """Base class for Jax input classes.

  During Lingvo Jax's train, on each host an input instance will be
  created (input_p.Instantiate()), and then GetNext() is iteratively
  called in eager mode to generate one batch of data for each step
  of train/eval/etc.

  If supported, for eval, Reset() is called after each eval step.
  See p.reset_for_eval below.
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Common Params for all inputs."""
    p = InstantiableParams(cls)
    p.Define('name', 'input', 'Name of this input.')

    p.Define(
        'batch_size', None, 'The (Jax per process) Batch size. '
        'Each call to GetNext() returns a batch with this '
        'batch size.')

    # Sharding behavior.
    p.Define(
        'num_infeed_hosts', 1,
        'Usually set to jax.process_count(). Implementation must '
        'ensure that the data is sharded into this many shard.')
    p.Define(
        'infeed_host_index', 0,
        'Usually set to jax.process_index(). Implementation must '
        'ensure that each instance returns a shard with this index.')

    # Deterministic randomness.
    p.Define(
        'input_random_seed', None, 'If set, implementation must '
        'ensure that this is used to seed randomness, e.g. '
        'when shuffling in a deterministic manner.')

    p.Define(
        'reset_for_eval', False,
        'If set, eval will continue until OutOfRange is raised, '
        'and Reset() will called for each eval. Implementation '
        'must ensure that all variant p.infeed_host_index '
        'instances raise after the same number of calls to '
        'GetNext() to ensure synchronization across hosts. '
        'If not set, GetNext() must never raise.')
    return p

  def __init__(self, p: ParamsT) -> None:
    if p.batch_size is None:
      raise ValueError('Must specify p.batch_size')
    self._params = p.Copy()

  @property
  def params(self) -> ParamsT:
    return self._params

  def GetNext(self) -> NestedJTensor:
    raise NotImplementedError

  def Reset(self) -> None:
    pass


class LingvoInputAdaptor(BaseInput):
  """Syntactic sugar for adapting a Lingvo style input for Jax."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('input', None, 'Params of a Lingvo input generator.')
    return p

  def __init__(self, p):
    p.batch_size = -1  # unused
    p.reset_for_eval = False
    super().__init__(p)

    with py_utils.InfeedContextScope(
        infeed_host_index=p.infeed_host_index,
        num_infeed_hosts=p.num_infeed_hosts):
      self.input = p.input.Instantiate()

  @tf.function
  def _GetBatch(self) -> NestedMap:
    p = self.params
    with py_utils.InfeedContextScope(
        infeed_host_index=p.infeed_host_index,
        num_infeed_hosts=p.num_infeed_hosts):
      ret = self.input.GetPreprocessedInputBatch()
    # Remove unsupported string (byte) array from input.
    return ret.Filter(lambda v: v.dtype != tf.string)

  def GetNext(self) -> NestedJTensor:
    ret = self._GetBatch()
    return tf.nest.map_structure(lambda x: x.numpy(), ret)
