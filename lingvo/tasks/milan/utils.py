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
"""Generic utility functions."""

import collections
import copy

from typing import List, Optional

from absl import logging
from lingvo import compat as tf
from lingvo.core import builder_layers
from lingvo.core import py_utils


def BatchMajorToTimeMajor(tensor):
  """Transposes axes 0 and 1 of `tensor`, keeping all others the same."""
  perm = [1, 0] + list(range(tensor.shape.ndims))[2:]
  return tf.transpose(tensor, perm=perm)


def MakeFnLayer(fn, name: Optional[str] = None):
  """Decorator that wraps a python function as a lingvo `FnLayer`.

  Example::

    @MakeFnLayer
    def add_two_things(x, y):
      return x + y

    layer = add_two_things.Instantiate()
    output = layer(tf.constant([1, 2, 3], tf.constant([4, 5, 6]))

  To wrap a lambda, call this as a standard function and provide a 'name'::

    layer_params = MakeFnLayer(lambda x, y: x + y, name='add_two_things')

  Args:
    fn: Function to be wrapped. Can be anything that take Tensor input(s) and
      produces Tensor output(s).
    name: Name of the wrapped layer; defaults to the name of `fn`. Must be set
      if `fn` is a lambda.

  Returns:
    Params of the `FnLayer` wrapper.
  """
  name = name or fn.__name__
  if name == '<lambda>':
    raise ValueError('MakeFnLayer: must set "name" when wrapping a lambda.')
  return builder_layers.FnLayer.Params().Set(name=name, fn=fn)


def GetFromNestedMapOrDie(nested_map, key):
  """Returns the value of `key` if it exists, otherwise dies with a nice error.

  Args:
    nested_map: The input `NestedMap`.
    key: The key to retrieve. May use dots to refer to a `nested.field_name`.
  """
  value = nested_map.Get(key)
  if value is None:
    all_keys = [k for k, _ in nested_map.FlattenItems()]
    raise ValueError(
        'No feature {} in input batch. Available features are: {}'.format(
            key, all_keys))
  return value


class Selector:
  """Selects an arbitrarily-structured set of fields from a NestedMap.

  A `Selector` selects one or more fields from a NestedMap and returns them in a
  user-defined structure. One use is to restructure inputs so they can be fed
  as inputs to a layer that expects a different format.

  The simplest selector extracts a single field from the input::

    batch = NestedMap(foo=NestedMap(a=1, b=2), c=3)
    selector = Selector('foo.a')
    selector(batch) ==> 1

  In general, selectors can extract any number of fields and return them in a
  user-defined structure::

    # Select multiple elements as a tuple.
    tuple_selector = Selector(('foo.a', 'c'))
    tuple_selector(batch) ==> (1, 3)

    # Select multiple elements arranged in a nested map structure.
    fancy_selector = Selector(
        NestedMap(x='foo.a', y='c', bar=NestedMap(baz='foo.b')))
    fancy_selector(batch) ==> NestedMap(x=1, y=3, bar=NestedMap(baz=2))
  """

  def __init__(self, spec):
    """Creates an instance using the fields and structure specified in `spec`.

    Args:
      spec: The field(s) to select, as a string field name or a (possibly
        nested) structure containing field names as values. The structure may be
        in any format supported by `tf.nest`, or a lingvo `NestedMap`.
    """
    self._spec = copy.deepcopy(spec)
    self._flat_source_keys = tf.nest.flatten(self._spec)
    for key in self._flat_source_keys:
      py_utils.NestedMap.CheckKey(key)

  def __call__(self, nested_map: py_utils.NestedMap):
    flat_values = [
        GetFromNestedMapOrDie(nested_map, key) for key in self._flat_source_keys
    ]
    if isinstance(self._spec, py_utils.NestedMap):
      return self._spec.Pack(flat_values)
    else:
      return tf.nest.pack_sequence_as(self._spec, flat_values)


def InferBatchSize(batch) -> int:
  """Returns the batch size of the given Tensor or Tensors.

  Args:
    batch: A Tensor or structure of Tensors representing a batch of examples.

  Returns:
     An int, the batch size.
  Raises:
    ValueError: if `batch` contains no Tensors or the batch size (shape dim 0)
      is not statically known.
  """
  tensors = tf.nest.flatten(batch)
  if not tensors:
    raise ValueError('Can\'t infer batch size: batch has no tensors.')
  batch_size = tensors[0].shape.as_list()[0]
  if batch_size is None:
    raise ValueError('Can\'t infer batch size: input tensor has dynamic shape.')
  # TODO(austinwaters): Check that others tensors agree?
  return batch_size


def ResolveBatchDim(shape: tf.TensorShape, batch_size: int) -> tf.TensorShape:
  """Returns `shape` with the None batch dimension replaced by `batch_size`."""
  # NB: Dies if dim 0 is already known but not equal to batch_size.
  return shape.merge_with([batch_size] + [None] * (shape.rank - 1))


class BatchFlattener:
  """Reshapes tensors for functions/layers that expect a single batch dimension.

  Example:
  Suppose a sequence `encoder` has signature ::

    [batch_size, max_length] -> [batch_size, max_length, feature_dim]

  and we want to run it on inputs with shape `[batch_size, n, max_length]`, i.e.
  where each example has `n` sequences.

  `BatchFlattener` can handle the mess of flattening the batch
  structure of the inputs and restoring it on the outputs::

    inputs = ...  # [batch_size, n, max_length]
    bf = BatchFlattener([None, n])
    flat_inputs = bf.Flatten(inputs)  # [batch_size * n, max_length]
    flat_outputs = encoder(flat_inputs)  # [batch_size * n, feature_dim]
    outputs = bf.Unflatten(flat_outputs)  # [batch_size, n, feature_dim]

  In general, arguments to `Flatten()` and `Unflatten()` may be single Tensors
  or (nearly) arbitrary structures of Tensors, including `NestedMap`.
  """

  def __init__(self, batch_shape):
    batch_shape = tf.TensorShape(batch_shape)
    assert batch_shape.rank >= 1
    batch_shape[1:].assert_is_fully_defined()
    self._batch_shape = batch_shape
    self._is_no_op = batch_shape == tf.TensorShape([None])

  @property
  def batch_shape(self):
    return self._batch_shape

  def Flatten(self, tensors):
    """Reshapes `tensors`, collapsing the leading `batch_shape` to one dim.

    The shapes of all `tensors` must begin with `batch_shape`.

    Args:
      tensors: A tensor or structure of tensors to be reshaped.

    Returns:
      The reshaped tensors, with `batch_shape.rank` - 1 fewer dimensions, in the
      same format (tensor, list, dict) as the input.
    """
    if self._is_no_op:
      return tensors
    flat_tensors = [
        tf.reshape(t, self._GetFlatShape(t)) for t in tf.nest.flatten(tensors)
    ]
    return self._PackAs(tensors, flat_tensors)

  def Unflatten(self, flat_tensors):
    """The inverse of Flatten(); expands the leading dim to `batch_shape`.

    Args:
      flat_tensors: A tensor or structure of tensors to be reshaped.

    Returns:
      The reshaped tensors, with `batch_shape.rank` - 1 more dimensions, in the
      same format (tensor, list, dict) as the input.
    """
    if self._is_no_op:
      return flat_tensors
    batch_shape = self._batch_shape.as_list()
    if batch_shape[0] is None:
      batch_shape[0] = -1

    unflattened_tensors = [
        tf.reshape(flat_tensor, batch_shape + flat_tensor.shape.as_list()[1:])
        for flat_tensor in tf.nest.flatten(flat_tensors)
    ]
    return tf.nest.pack_sequence_as(flat_tensors, unflattened_tensors)

  def _PackAs(self, structure, values):
    if isinstance(structure, py_utils.NestedMap):
      return structure.Pack(values)
    else:
      return tf.nest.pack_sequence_as(structure, values)

  def _GetFlatShape(self, tensor):
    # Fill in leading batch dim if necessary.
    batch_rank = self._batch_shape.rank
    merged_batch_shape = tensor.shape[:batch_rank].merge_with(self._batch_shape)
    logging.info('Inferred batch shape %s', merged_batch_shape)
    return [merged_batch_shape.num_elements()] + tensor.shape[batch_rank:]


def FlattenBatch(tensors, batch_shape):
  return BatchFlattener(batch_shape).Flatten(tensors)


def UnflattenBatch(flat_tensors, batch_shape):
  return BatchFlattener(batch_shape).Unflatten(flat_tensors)


def CollectRegularizationLosses(layer) -> List[tf.Tensor]:
  """Collects regularization losses from `layer` or its Keras-like children.

  This function assumes (some) lingvo layers expose regularziation losses
  through a `losses` attribute, as `tf.keras.Layer`s do. When present, `losses`
  is assumed to be a list of regularization losses for the layer and all of its
  children.

  Args:
    layer: The root lingvo layer to collect regularization losses over.

  Returns:
    A list of scalar Tensors containing regularization losses.
  """

  losses = []
  deque = collections.deque([layer])
  while deque:
    child = deque.popleft()
    if hasattr(child, 'losses'):
      losses += child.losses
    else:
      deque.extend(tf.nest.flatten(child.children))
  return losses


def PadOrTrimDimension(tensor: tf.Tensor, new_size: int,
                       axis: int) -> tf.Tensor:
  tensor.shape.with_rank_at_least(abs(axis))
  shape = py_utils.GetShape(tensor)
  return py_utils.PadOrTrimTo(tensor,
                              shape[:axis] + [new_size] + shape[axis + 1:])
