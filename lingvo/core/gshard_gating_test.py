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
r"""Test code for different gating functions."""

from absl import logging
from lingvo import compat as tf
from lingvo.core import gshard_layers
from lingvo.core import test_utils
import numpy as np


class GatingTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  def testExpertChoiceV2(self):
    # pylint: disable=invalid-name
    B, L = 2, 6  # batch, length
    M = 8  # model_dim
    G = 2  # num_groups
    capacity_factor = 1
    S = (capacity_factor * B * L) // G  # group_size
    assert G * S == B * L  # capacity_factor=1
    E = 4
    C = (capacity_factor * G * S) // E
    np.random.seed(123456)
    npy_inputs = np.random.normal(0.0, 1.0,
                                  [B, L, M]).round(decimals=2).astype('float32')

    npy_paddings = np.array([[0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 1]],
                            dtype=np.float32)
    assert npy_paddings.shape == npy_inputs.shape[:-1], (npy_paddings.shape,
                                                         npy_inputs.shape[:-1])

    npy_grouped_inputs = np.reshape(npy_inputs, (G, S, -1))
    npy_grouped_paddings = np.reshape(npy_paddings,
                                      npy_grouped_inputs.shape[:-1])

    w = np.random.normal(0.0, 1.0, [M, E]).round(decimals=2).astype(
        'float32')  # gating weights

    logits = np.einsum('...M,ME->...E', npy_grouped_inputs, w)
    fprop_dtype = tf.float32

    _, combine, dispatch = (
        gshard_layers.TokenShufflingOnlogitsV2(
            logits=tf.convert_to_tensor(logits, dtype=fprop_dtype),
            paddings=tf.convert_to_tensor(
                npy_grouped_paddings, dtype=fprop_dtype),
            num_devices=1,
            experts_dim=E,
            expert_capacity_dim=C,
            fprop_dtype=fprop_dtype,
            use_xla_sharding=False,
            capacity_factor=capacity_factor,
            mask_dtype=np.int32))

    init_op = tf.global_variables_initializer()
    with self.session(use_gpu=False) as sess:
      sess.run(init_op)
      combine = sess.run([combine])
      logging.info(repr(combine))
      logging.info(repr(dispatch))


if __name__ == '__main__':
  test_utils.main()
