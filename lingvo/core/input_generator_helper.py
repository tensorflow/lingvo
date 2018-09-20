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
"""Helper functions for model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from six.moves import range
import tensorflow as tf

from lingvo.core import py_utils


def ComputeSplits(batch_size, num_splits):
  """Creates a tensor of size num_splits of number of values per split.

  Assigns each split floor(batch_size/num_splits) and round-robins
  the remainder (if any) to each split.

  Example::

    batch_size: [5]
    num_splits: 3
    returns: [2, 2, 1]

  Args:
    batch_size: tensor of rank 0, size of tensor to be split
    num_splits: number of splits to split tensor into
  Returns:
    tensor of length num_splits containing sizes of each split
  """
  values = tf.tile(
      tf.div([batch_size], num_splits),
      tf.constant(
          [num_splits], dtype=tf.int32))
  mods = tf.tile(tf.constant([1]), tf.mod([batch_size], num_splits))
  zeros = tf.tile(tf.constant([0]),
                  tf.subtract(tf.shape(values), tf.shape(mods)))
  mods = tf.concat([mods, zeros], 0)
  ret = tf.add(values, mods)
  # for some reason TF erases shape information if num_splits is 1
  if num_splits == 1:
    ret.set_shape([1])
  return ret


def SplitTensors(xs, num_splits):
  """Splits tensors in `xs` evenly into num_splits along the 1st dimenion.

  Args:
    xs: A tuple of tensors. Each tensor's 1st dimension is the same size.
    num_splits: A python integer.

  Returns:
    A tuple of lists of tensors, num elements in the tuple = len(xs).

    i-th element in each list corresponds to i-th split of each tensor in xs
    along the first dimension of each tensor.
  """
  # assert first dim of all tensors in xs is equal
  batch_dims = [tf.shape(x)[0] for x in xs]
  all_batch_dims = tf.stack(batch_dims)

  all_batch_dims = py_utils.with_dependencies([
      py_utils.assert_equal(
          all_batch_dims,
          tf.shape(xs[0])[0],
          message='first dim of tensors in xs must match'),
      py_utils.assert_greater_equal(
          tf.shape(xs[0])[0],
          num_splits,
          message='first dim of tensors in xs must be greater than num_splits')
  ], all_batch_dims)

  splits = ComputeSplits(tf.shape(xs[0])[0], num_splits)
  # add the above assertion into the compute graph
  splits = py_utils.with_dependencies([all_batch_dims], splits)
  split_xs = [tf.split(axis=0, num_or_size_splits=splits, value=x) for x in xs]

  return split_xs


def SplitDictOfTensors(t_dict, num_splits):
  """Splits tensors in `t_dict` evenly into `num_splits` along the 1st dimenion.

  Args:
    t_dict: A dictionary of tensors. Each tensor's 1st dimension is the same
      size.
    num_splits: A python integer.

  Returns:
    A list of dictionaries of tensors, num elements in the list = num_splits

    i-th dictionary in the list corresponds to i-th split of each tensor
    along the first dimension of each tensor for each key in the original dict.
  """
  keys = []
  values = []
  for k, v in sorted(six.iteritems(t_dict)):
    keys.append(k)
    values.append(v)

  splits = SplitTensors(tuple(values), num_splits)

  assert all(len(lst) == len(splits[0]) for lst in splits)

  ret_list = []
  for s in range(num_splits):
    d = {}
    for k in range(len(splits)):
      d[keys[k]] = splits[k][s]
    ret_list.append(d)

  return ret_list
