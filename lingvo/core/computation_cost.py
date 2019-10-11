# Lint as: python2, python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities to estimate computation costs of layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lingvo import compat as tf
from lingvo.core import bn_layers  # for AddingAccumulator

COST_METRICS = {
    'flops': tf.int64,
}


def Prepare(layer):
  """Registers cost accumulators for layer and each of its child layers.

  This function should be called before FProp().

  Args:
    layer: the root layer.
  """

  def _Traverse(layer):
    """Adds accumulators to layer and its descendant layers."""
    if isinstance(layer, (list, tuple)):
      for layer_i in layer:
        _Traverse(layer_i)
      return
    with tf.name_scope(layer.params.name):
      for cost_metric_name in COST_METRICS:
        dtype = COST_METRICS[cost_metric_name]
        layer.RegisterAccumulator(
            cost_metric_name,
            bn_layers.AddingAccumulator(shape=[], dtype=dtype))
      for _, child in sorted(layer.children.items()):
        _Traverse(child)

  _Traverse(layer)


def _HasAccumulator(layer, cost_metric_name):
  assert cost_metric_name in COST_METRICS
  return cost_metric_name in layer.accumulators


def _GetAccumulator(layer, cost_metric_name):
  assert cost_metric_name in COST_METRICS
  if cost_metric_name not in layer.accumulators:
    raise ValueError('Prepare was not called for %s' % layer.path)
  return layer.accumulators[cost_metric_name]


def Add(layer, cost_metric_name, cost):
  if not _HasAccumulator(layer, cost_metric_name):
    return
  _GetAccumulator(layer, cost_metric_name).Update(cost)


def Get(layer, cost_metric_name):
  """Returns the aggregated cost metric."""
  return _GetAccumulator(layer, cost_metric_name).GetValue()
