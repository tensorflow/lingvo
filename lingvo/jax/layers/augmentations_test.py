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
"""Tests for lingvo Jax augmentation layers."""

from absl import logging
from absl.testing import absltest
import jax
from jax import numpy as jnp
from jax import test_util
from lingvo.jax import base_layer
from lingvo.jax import test_utils
from lingvo.jax.layers import augmentations
import numpy as np

ToNp = test_utils.ToNp


class AugmentationsTest(test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def testMaskedLmDataAugmenterSmall(self):
    p = augmentations.MaskedLmDataAugmenter.Params().Set(
        name='mlm', vocab_size=32000, mask_token_id=0)
    layer = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    with base_layer.JaxContext.NewContext(prng_key=prng_key, global_step=1):
      inputs = jnp.arange(10, dtype=jnp.int32)
      paddings = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0],
                           dtype=jnp.float32)
      augmented_ids, augmented_pos = layer.FProp(None, inputs, paddings)
      logging.info('augmented_ids: %s', augmented_ids)
      logging.info('augmented_pos: %s', augmented_pos)
      expected_ids = np.array([0, 1, 2, 0, 0, 5, 6, 7, 8, 9])
      expected_pos = np.array([0., 0., 0., 1., 1., 0., 0., 0., 0., 0.])
      self.assertAllClose(ToNp(expected_ids), ToNp(augmented_ids))
      self.assertAllClose(ToNp(expected_pos), ToNp(augmented_pos))

  def testMaskedLmDataAugmenterLarge(self):
    p = augmentations.MaskedLmDataAugmenter.Params().Set(
        name='mlm', vocab_size=32000, mask_token_id=0)
    layer = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    with base_layer.JaxContext.NewContext(prng_key=prng_key, global_step=1):
      inputs = jnp.arange(100, dtype=jnp.int32)
      paddings = jnp.zeros_like(inputs).astype(jnp.float32)
      augmented_ids, augmented_pos = layer.FProp(None, inputs, paddings)
      logging.info('augmented_ids: %s', np.array_repr(augmented_ids))
      logging.info('augmented_pos: %s', np.array_repr(augmented_pos))
      expected_ids = np.array([
          0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 15, 16, 17, 0, 0, 20,
          21, 22, 23, 24, 0, 26, 27, 28, 15113, 0, 31, 32, 33, 34, 0, 0, 37, 38,
          0, 40, 41, 27325, 43, 0, 45, 46, 12582, 48, 49, 50, 51, 52, 53, 54,
          55, 56, 57, 58, 59, 60, 61, 0, 63, 64, 0, 66, 0, 68, 19012, 70, 71,
          72, 73, 74, 75, 76, 0, 78, 0, 80, 2952, 82, 83, 84, 85, 86, 0, 0, 89,
          90, 91, 0, 93, 94, 95, 96, 97, 98, 99
      ])
      expected_pos = np.array([
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
          0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0.,
          0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1.,
          0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0.,
          0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.
      ])
      self.assertAllClose(ToNp(expected_ids), ToNp(augmented_ids))
      self.assertAllClose(ToNp(expected_pos), ToNp(augmented_pos))


if __name__ == '__main__':
  absltest.main()
