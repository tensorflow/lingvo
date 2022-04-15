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
"""A suite of metric classes to compute aggregate stats across TPU hosts."""

import abc
import collections
import logging
from typing import Optional

import jax
import jax.numpy as jnp
from lingvo.jax import py_utils
from lingvo.jax import summary_utils
import numpy as np

InstantiableParams = py_utils.InstantiableParams


class BaseMetrics(metaclass=abc.ABCMeta):
  """Abstract base class for all tasks."""

  @classmethod
  def Params(cls):  # pylint:disable=invalid-name
    p = InstantiableParams(cls)
    p.Define('name', None, 'Name of the metric')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    self._params = params.Copy()
    self._metrics = collections.defaultdict(list)

  @property
  def params(self) -> InstantiableParams:
    """Returns the params upon which this layer is built."""
    return self._params

  def store(self, batch_metrics):
    for k in batch_metrics:
      self._metrics[k].append(batch_metrics[k])

  @abc.abstractmethod
  def update(self, *args, **kwargs):
    pass

  @abc.abstractmethod
  def finalize(self):
    pass

  def summarize(self, step_i, prefix):
    metrics = self.finalize()
    for k, v in metrics.items():
      value, weight = v
      logging.info('  %s=%f (weight=%f)', k, value, weight)
      summary_utils.write_summary_tensor(step_i, f'{prefix}/{k}', value,
                                         summary_utils.SummaryType.SCALAR)
      summary_utils.write_summary_tensor(step_i, f'{prefix}/{k}-weight', weight,
                                         summary_utils.SummaryType.SCALAR)


def _pmap_aggregate_metrics(f,
                            batch_metrics,
                            metric_keys,
                            reshard: bool,
                            pmap_axis_name: str = 'batch'):
  """Aggregate a dict of metrics over all replicas.

  Args:
    f: A callable that is used to aggregate the metrics across tpus().
      The function signature of f should following convention: f(value, weight,
        pmap_axis_name) For example to compute the mean across TPU replicas
        def _pmap_mean(value, weight, axis_name): sum_value = jax.lax.psum(value
          * weight, axis_name) sum_weight = jax.lax.psum(weight, axis_name)
          return (sum_value / (sum_weight + 1e-8), sum_weight)
    batch_metrics: dictionary of items to aggregate over.
    metric_keys: the set of keys to aggregate over. If None, will aggregate over
      all.
    reshard: boolean to indicate whether to reshard before aggregation.
    pmap_axis_name: Data parallel axis name for jax.pmap,psum, etc operations.

  Returns:
    Aggregated across TPU version of the metrics dict.
  """

  # Reshard for sum over devices
  def _reshard(batch_metrics):
    reshard_metrics = type(batch_metrics)()
    for k, v in batch_metrics.items():
      value, weight = v
      assert weight.ndim == 0
      new_value = jnp.stack([jnp.array(value)] * jax.local_device_count())
      new_weight = jnp.ones(
          shape=(jax.local_device_count(),),
          dtype=weight.dtype) * weight / jax.local_device_count()
      reshard_metrics[k] = (new_value, new_weight)
    return reshard_metrics

  # aggregate across replicas
  def _aggregate(metrics_dict):
    metrics = type(metrics_dict)()
    for k, v in metrics_dict.items():
      if metric_keys and k not in metric_keys:
        continue
      value, weight = v
      metrics[k] = f(value, weight, pmap_axis_name)
    return metrics

  if reshard:
    pmap_aggregate = jax.pmap(_aggregate, pmap_axis_name, out_axes=None)
    return pmap_aggregate(_reshard(batch_metrics))
  else:
    return _aggregate(batch_metrics)


def _vmap_aggregate_metrics(f, metrics_dict):
  """Aggregate a dict of metrics over all recorded batches.

  Args:
    f: A Callable that computes the aggregate over a vector of metrics. For
      example to compute the mean, we sum over the input vector of weights
        def _vmap_mean(values, weights): sum_metric_weights = np.sum(weights)
          weighted_average = np.sum(values * weights) / sum_metric_weights
          return (weighted_average, sum_metric_weights
    metrics_dict: Dictionary of metrics each containing a vector of values and
      associated weights

  Returns:
      Aggregated metrics.
  """
  metrics = {}
  for k in metrics_dict.keys():
    values = jnp.stack([metric[0] for metric in metrics_dict[k]])
    weights = jnp.stack([metric[1] for metric in metrics_dict[k]])
    metrics[k] = f(values, weights)
  return metrics


class MeanMetrics(BaseMetrics):
  """Computes the mean of the metrics over devices."""

  @classmethod
  def Params(cls) -> InstantiableParams:  # pylint:disable=invalid-name
    """Task parameters."""
    p = super().Params()
    p.Define('metric_keys', None,
             'List of metrics that will be aggregated and logged.')
    return p

  def aggregate(self, batch_metrics, reshard: Optional[bool] = False):
    p = self.params

    def _pmap_mean(value, weight, axis_name):
      sum_value = jax.lax.psum(value * weight, axis_name)
      sum_weight = jax.lax.psum(weight, axis_name)
      return (sum_value / (sum_weight + 1e-8), sum_weight)

    return _pmap_aggregate_metrics(
        _pmap_mean,
        batch_metrics,
        p.metric_keys,
        reshard,
        pmap_axis_name='batch')

  def update(self, batch_metrics) -> None:
    """Add per batch metrics to the metrics dict.

    Sum over shards with pmap.

    Args:
      batch_metrics: per batch metrics (unsharded) - e.g. output of
        process_decode_out()
    """
    batch_metrics = self.aggregate(batch_metrics, reshard=True)
    self.store(batch_metrics)

  def finalize(self):
    """Finalize aggregation over all batches and returns the metrics."""

    def _vmap_mean(values, weights):
      sum_metric_weights = jnp.sum(weights)
      weighted_average = jnp.sum(values * weights, axis=0) / sum_metric_weights
      return (weighted_average, sum_metric_weights)

    metrics = _vmap_aggregate_metrics(_vmap_mean, self._metrics)
    self._metrics = collections.defaultdict(list)
    return metrics


class MaxMetrics(BaseMetrics):
  """Computes the max over sharded metrics."""

  @classmethod
  def Params(cls) -> InstantiableParams:  # pylint:disable=invalid-name
    """Task parameters."""
    p = super().Params()
    p.Define('metric_keys', None,
             'List of metrics that will be aggregated and logged.')
    return p

  def aggregate(self, batch_metrics, reshard: Optional[bool] = False):
    p = self.params

    def _pmap_max(value, weight, axis_name):
      max_value = jax.lax.pmax(value, axis_name)
      sum_weight = jax.lax.psum(weight, axis_name)
      return (max_value, sum_weight)

    return _pmap_aggregate_metrics(
        _pmap_max,
        batch_metrics,
        p.metric_keys,
        reshard,
        pmap_axis_name='batch')

  def update(self, batch_metrics) -> None:
    """Add per batch max metrics to the metrics dict.

    Args:
      batch_metrics: per batch metrics (unsharded) - e.g. output of
        process_decode_out()
    """
    batch_metrics = self.aggregate(batch_metrics, reshard=True)
    self.store(batch_metrics)

  def finalize(self):
    """Finalize aggregation over all batches and returns the metrics."""

    def _vmap_max(values, weights):
      sum_metric_weights = np.sum(weights, axis=0)
      max_value = np.max(values, axis=0)
      return (max_value, sum_metric_weights)

    metrics = _vmap_aggregate_metrics(_vmap_max, self._metrics)
    self._metrics = collections.defaultdict(list)
    return metrics


class HistogramMetrics(BaseMetrics):
  """Compute aggregate single scalar statistics over sharded batches."""

  @classmethod
  def Params(cls) -> InstantiableParams:  # pylint:disable=invalid-name
    """Task parameters."""
    p = super().Params()
    p.Define('histogram_key', None, 'Key which contains the histogram data.')
    return p

  def aggregate(self, batch_metrics, reshard: Optional[bool] = False):
    p = self.params

    def _pmap_sum(value, weight, axis_name):
      value = jax.lax.psum(value, axis_name)
      weight = jax.lax.psum(weight, axis_name)
      return (value, weight)

    return _pmap_aggregate_metrics(
        _pmap_sum,
        batch_metrics, [p.histogram_key],
        reshard,
        pmap_axis_name='batch')

  def update(self, batch_metrics) -> None:
    """Add per batch metrics to the metrics dict.

    Sum over shards with pmap.

    Args:
      batch_metrics: per batch metrics (unsharded) - e.g. output of
        process_decode_out()
    """
    batch_metrics = self.aggregate(batch_metrics, reshard=True)
    self.store(batch_metrics)

  def finalize(self):
    """Finalize aggregation over all batches and returns the metrics."""
    metrics = {}
    for k in self._metrics.keys():
      metric_values = np.stack([metric[0] for metric in self._metrics[k]])
      metric_weights = np.stack([metric[1] for metric in self._metrics[k]])
      sum_metric_weights = np.sum(metric_weights)
      histogram = np.sum(metric_values, axis=0)
      num_groups = histogram.shape[0] if histogram.ndim > 0 else 1
      normalizer = np.sum(histogram) / num_groups

      # [g, c]
      probs = histogram / jnp.maximum(normalizer, 1.0)
      log_probs = jnp.log(jnp.maximum(1.0e-30, probs))
      # [g]
      sum_plogp = jnp.sum(log_probs * probs, -1)
      pplx = jnp.mean(jnp.exp(-sum_plogp))
      entropy = jnp.log(pplx)

      metrics[k + '_pplx'] = (pplx, sum_metric_weights)
      metrics[k + '_entropy'] = (entropy, sum_metric_weights)

      onehot = jnp.greater(histogram, 0).astype(jnp.float32)
      avg_num_covered_words = jnp.mean(jnp.sum(onehot, -1))
      num_classes = histogram.shape[-1]
      metrics[k + '_coverage'] = (avg_num_covered_words / num_classes,
                                  sum_metric_weights)
    return metrics


class CompositeMetrics(BaseMetrics):
  """Compute aggregate single scalar statistics over sharded batches."""

  @classmethod
  def Params(cls) -> InstantiableParams:  # pylint:disable=invalid-name
    """Task parameters."""
    p = super().Params()
    p.Define('metrics_p', None,
             'List of metrics that will be aggregated and logged.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    self.metrics_calcs = [m.Instantiate() for m in p.metrics_p]

  def aggregate(self, batch_metrics, reshard: Optional[bool] = False):
    all_metrics = collections.defaultdict()
    for m in self.metrics_calcs:
      metrics = m.aggregate(batch_metrics, reshard)
      for k, v in metrics.items():
        all_metrics[k] = v
    return all_metrics

  def update(self, batch_metrics) -> None:
    """Add per batch metrics to the metrics dict.

    Sum over shards with pmap.

    Args:
      batch_metrics: per batch metrics (unsharded) - e.g. output of
                     process_decode_out()
    """
    for m in self.metrics_calcs:
      m.update(batch_metrics)

  def finalize(self):
    """Finalize aggregation over all batches and returns the metrics."""
    metrics = {}
    for m in self.metrics_calcs:
      finalized = m.finalize()
      for k, v in finalized.items():
        metrics[k] = v
    return metrics
