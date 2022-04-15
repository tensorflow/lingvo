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
"""Tests for base_metrics."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from lingvo.jax import base_metrics
from lingvo.jax import py_utils
from lingvo.jax import test_utils

NestedMap = py_utils.NestedMap


class BaseMetricsTest(test_utils.TestCase):

  def _decode(self, feats):
    b, t, d = feats.shape
    mean_val = jnp.mean(feats)
    max_val = jnp.max(feats)
    min_val = jnp.min(feats)
    frames = jnp.reshape(feats, [b * t, d])
    val = jnp.where(frames > 0.8, size=b * t * d)
    hist = jnp.zeros([d])
    hist = hist.at[val[1]].add(1)
    hist = hist.at[0].set(1)
    nframes = jnp.array(b * t)
    metrics = {
        'mean': (mean_val, nframes),
        'max': (max_val, nframes),
        'min': (min_val, nframes),
        'hist': (hist, nframes)
    }
    return metrics

  def test_reshard_metrics(self):
    feats = jax.random.uniform(jax.random.PRNGKey(1234), [8, 10, 100, 128])

    mean_metrics_p = base_metrics.MeanMetrics.Params().Set(metric_keys=['mean'])
    max_metrics_p = base_metrics.MaxMetrics.Params().Set(metric_keys=['max'])
    hist_metrics_p = base_metrics.HistogramMetrics.Params().Set(
        histogram_key='hist')
    composite_p = base_metrics.CompositeMetrics.Params().Set(
        metrics_p=[mean_metrics_p, max_metrics_p, hist_metrics_p])
    composite = composite_p.Instantiate()

    for i in range(8):
      batch_metrics = self._decode(feats[i])
      composite.update(batch_metrics)
    metrics = composite.finalize()

    self.assertAllClose(metrics['mean'][0], jnp.mean(feats))
    self.assertAllClose(metrics['max'][0], jnp.max(feats))
    self.assertArraysEqual(metrics['hist_coverage'][0], jnp.array(1.0))

  def test_aggregate_metrics(self):
    feats = jax.random.uniform(jax.random.PRNGKey(1234), [1, 10, 100, 128])

    mean_metrics_p = base_metrics.MeanMetrics.Params().Set(metric_keys=['mean'])
    max_metrics_p = base_metrics.MaxMetrics.Params().Set(metric_keys=['max'])
    composite_p = base_metrics.CompositeMetrics.Params().Set(
        metrics_p=[mean_metrics_p, max_metrics_p])
    composite = composite_p.Instantiate()

    def _decode_step(feats):
      metrics = self._decode(feats)
      metrics = composite.aggregate(metrics)
      return metrics

    p_decode = jax.pmap(_decode_step, axis_name='batch')
    metrics = p_decode(feats)
    self.assertAllClose(metrics['mean'][0][0], jnp.mean(feats))
    self.assertAllClose(metrics['max'][0][0], jnp.max(feats))

if __name__ == '__main__':
  absltest.main()
