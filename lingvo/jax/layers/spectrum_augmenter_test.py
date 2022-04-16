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
#
# ==============================================================================
"""Tests for spectrum_augmenter."""

from absl.testing import absltest
import jax
from jax import numpy as jnp
from lingvo.jax import base_layer
from lingvo.jax import test_utils
from lingvo.jax.layers import spectrum_augmenter
import numpy as np

to_np = test_utils.to_np


class SpectrumAugmenterTest(test_utils.TestCase):

  def testSpectrumAugmenterWithTimeMask(self):
    batch_size = 5
    inputs = jnp.ones([batch_size, 20, 2], dtype=jnp.float32)
    paddings = []
    for i in range(batch_size):
      paddings.append(
          jnp.concatenate([jnp.zeros([1, i + 12]),
                           jnp.ones([1, 8 - i])],
                          axis=1))
    paddings = jnp.concatenate(paddings, axis=0)

    p = spectrum_augmenter.SpectrumAugmenter.Params()
    p.name = 'specAug_layers'
    p.freq_mask_max_bins = 0
    p.time_mask_max_frames = 5
    p.time_mask_count = 2
    p.time_mask_max_ratio = 1.
    specaug_layer = p.Instantiate()
    expected_output = np.array(
        [[[1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.],
          [1., 1.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [1., 1.], [1., 1.],
          [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.], [1., 1.], [1., 1.], [0., 0.], [1., 1.], [0., 0.],
          [0., 0.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.],
          [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.],
          [0., 0.], [0., 0.], [0., 0.], [1., 1.], [1., 1.], [1., 1.], [1., 1.],
          [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [0., 0.],
          [0., 0.], [1., 1.], [1., 1.], [0., 0.], [0., 0.], [0., 0.], [0., 0.],
          [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.]],
         [[1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.],
          [1., 1.], [1., 1.], [1., 1.], [1., 1.], [0., 0.], [0., 0.], [0., 0.],
          [0., 0.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.]]])
    context_p = base_layer.JaxContext.Params().Set(do_eval=False)
    prng_key = jax.random.PRNGKey(seed=23456)
    theta = specaug_layer.instantiate_variables(prng_key)
    actual_layer_output, _ = test_utils.apply(
        specaug_layer,
        theta,
        specaug_layer.fprop,
        inputs,
        paddings,
        context_p=context_p)
    self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterWithFrequencyMask(self):
    inputs = jnp.ones([3, 5, 10], dtype=jnp.float32)
    paddings = jnp.zeros([3, 5])
    p = spectrum_augmenter.SpectrumAugmenter.Params()
    p.name = 'specAug_layers'
    p.freq_mask_max_bins = 6
    p.freq_mask_count = 2
    p.time_mask_max_frames = 0
    specaug_layer = p.Instantiate()
    # pyformat: disable
    # pylint: disable=bad-whitespace,bad-continuation
    expected_output = np.array(
        [[[1., 1., 1., 1., 1., 0., 0., 0., 0., 1.],
          [1., 1., 1., 1., 1., 0., 0., 0., 0., 1.],
          [1., 1., 1., 1., 1., 0., 0., 0., 0., 1.],
          [1., 1., 1., 1., 1., 0., 0., 0., 0., 1.],
          [1., 1., 1., 1., 1., 0., 0., 0., 0., 1.]],
         [[1., 1., 1., 1., 1., 0., 0., 0., 1., 1.],
          [1., 1., 1., 1., 1., 0., 0., 0., 1., 1.],
          [1., 1., 1., 1., 1., 0., 0., 0., 1., 1.],
          [1., 1., 1., 1., 1., 0., 0., 0., 1., 1.],
          [1., 1., 1., 1., 1., 0., 0., 0., 1., 1.]],
         [[1., 1., 0., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 0., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 0., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 0., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 0., 1., 1., 1., 1., 1., 1., 1.]]])
    # pylint: enable=bad-whitespace,bad-continuation
    # pyformat: enable
    context_p = base_layer.JaxContext.Params().Set(do_eval=False)
    prng_key = jax.random.PRNGKey(seed=34567)
    theta = specaug_layer.instantiate_variables(prng_key)
    actual_layer_output, _ = test_utils.apply(
        specaug_layer,
        theta,
        specaug_layer.fprop,
        inputs,
        paddings,
        context_p=context_p)
    self.assertAllClose(actual_layer_output, expected_output)


if __name__ == '__main__':
  absltest.main()
