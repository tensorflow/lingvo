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
"""Tests for Learners."""

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
from jax import test_util
from lingvo.jax import base_layer
from lingvo.jax import learners
from lingvo.jax import optimizers
from lingvo.jax import py_utils
from lingvo.jax import schedules
import numpy as np


class LearnersTest(test_util.JaxTestCase):

  @parameterized.parameters(
      (0.5, 0.5, 1.5, 1., 0.),
      (0., 0., 1.5, 1., 0.),
      (0.5, 0.5, 1.5, 0., 1.),
      (0., 0., 1.5, 0., 1.),
  )
  def test_learner_clip_gradients(self, g1a, g1b, g2, global_clip_norm,
                                  single_clip_norm):
    learner_p = learners.Learner.Params()
    learner_p.name = 'learner'
    learner_p.loss_name = 'loss'
    learner_p.optimizer = optimizers.Sgd.Params()
    learner_p.optimizer.learning_rate = 1.
    learner_p.optimizer.lr_schedule = schedules.Constant.Params()
    if global_clip_norm:
      learner_p.optimizer.clip_gradient_norm_to_value = global_clip_norm
    elif single_clip_norm:
      learner_p.optimizer.clip_gradient_single_norm_to_value = single_clip_norm

    learner_instance = learner_p.Instantiate()

    grads = py_utils.NestedMap(
        grad1=jnp.array([g1a, g1b], dtype=jnp.float32),
        grad2=jnp.array([g2], dtype=jnp.float32))

    with base_layer.JaxContext.new_context():
      transformed_grads = learner_instance.scale_gradients(grads)

    global_norm = np.linalg.norm([g1a, g1b, g2])
    local_norm1 = np.linalg.norm([g1a, g1b])
    local_norm2 = np.linalg.norm([g2])
    if global_clip_norm:
      gn1a = g1a * global_clip_norm / max(global_norm, global_clip_norm)
      gn1b = g1b * global_clip_norm / max(global_norm, global_clip_norm)
      gn2 = g2 * global_clip_norm / max(global_norm, global_clip_norm)
    elif single_clip_norm:
      gn1a = g1a * single_clip_norm / max(local_norm1, single_clip_norm)
      gn1b = g1b * single_clip_norm / max(local_norm1, single_clip_norm)
      gn2 = g2 * single_clip_norm / max(local_norm2, single_clip_norm)
    expected_grad1 = jnp.array([gn1a, gn1b], dtype=jnp.float32)
    expected_grad2 = jnp.array([gn2], dtype=jnp.float32)

    self.assertAllClose(expected_grad1, transformed_grads.grad1)
    self.assertAllClose(expected_grad2, transformed_grads.grad2)


if __name__ == '__main__':
  absltest.main()
