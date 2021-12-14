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
"""Tests for quantizer."""

from absl.testing import absltest

import jax
from jax import test_util
import jax.numpy as jnp
from lingvo.jax.layers import quantizer
import numpy as np


class SeqVectorQuantizerTest(test_util.JaxTestCase):

  w = np.array([[0.116230249, 0.0104732513, -0.409445882, -0.153374314],
                [-0.0672334433, -0.430877686, -0.280010223, 0.394074917],
                [-0.360892653, -0.153173685, -0.45321393, -0.176380157],
                [0.406187773, 0.304340839, 0.439772606, 0.368542314]])

  def _GetParams(self, num_classes, latent_dim):
    return quantizer.SeqVectorQuantizer.Params().Set(
        name='vq',
        normalize=True,
        num_latent_classes=num_classes,
        latent_dim=latent_dim,
        beta=0.1)

  def testBase(self):
    num_classes = 4
    latent_dim = 4

    b, t = 2, 4
    np.random.seed(2021)
    z = np.random.rand(b, t, latent_dim).astype(np.float32)
    paddings = np.zeros((b, t)).astype(np.float32)

    vq_p = self._GetParams(num_classes, latent_dim)
    vq = vq_p.Instantiate()
    vq_theta = vq.instantiate_variables(jax.random.PRNGKey(1))
    vq_theta.w = jnp.expand_dims(self.w, 1)
    out = vq.fprop(vq_theta, z, paddings)

    with self.subTest('test_shape'):
      self.assertEqual((b, t, latent_dim), out.z_q.shape)
      self.assertEqual((b, t, 1), out.z_codes.shape)
      self.assertEqual((b, t, 1, num_classes), out.z_onehot.shape)
    with self.subTest('test_z_q'):
      self.assertAllClose(15.861525, np.sum(out.z_q))
    with self.subTest('test_z_codes'):
      self.assertEqual(24, np.sum(out.z_codes))
    with self.subTest('test_codebook_coverage'):
      self.assertEqual(0.25, np.sum(out.codebook_coverage))
    with self.subTest('test_pplx'):
      self.assertEqual(1.0, out.pplx)
    with self.subTest('test_entropy'):
      self.assertAllClose(0., out.entropy)

  def testNCodebooks(self):
    num_classes = 4
    latent_dim = 4
    num_groups = 2

    b, t = 2, 4
    np.random.seed(2021)
    z = np.random.rand(b, t, latent_dim).astype(np.float32)
    paddings = np.zeros((b, t)).astype(np.float32)

    vq_p = self._GetParams(num_classes, latent_dim)
    vq_p.Set(num_groups=num_groups)
    vq = vq_p.Instantiate()
    vq_theta = vq.instantiate_variables(jax.random.PRNGKey(1))
    out = vq.fprop(vq_theta, z, paddings)

    with self.subTest('test_shape'):
      self.assertEqual((b, t, latent_dim), out.z_q.shape)
      self.assertEqual((b, t, num_groups), out.z_codes.shape)
      self.assertEqual((b, t, num_groups, num_classes), out.z_onehot.shape)


if __name__ == '__main__':
  absltest.main()
