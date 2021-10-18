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
"""Tests for lingvo Jax linear layers."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax import test_util
from lingvo.core import layers as lingvo_layers
from lingvo.jax import py_utils
from lingvo.jax import test_utils
from lingvo.jax.layers import linears
import numpy as np
import tensorflow.compat.v2 as tf

ToNp = test_utils.ToNp
ToTfNmap = test_utils.ToTfNmap


class LinearsTest(test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  @parameterized.parameters(('RELU'), ('TANH'), ('RELU6'), ('SIGMOID'),
                            ('NONE'))
  def test_feedforward_layer(self, activation):
    p = linears.FeedForwardLayer.Params().Set(
        name='jax_ffn', input_dims=3, output_dims=20, activation=activation)
    ffn = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = ffn.InstantiateVariables(prng_key)
    npy_input = np.random.normal(1.0, 0.5,
                                 [10, 10, p.input_dims]).astype('float32')
    inputs = jnp.asarray(npy_input)
    outputs = ffn.FProp(initial_vars, inputs)
    logging.info('initial_vars in ffn = %s', initial_vars)
    # Test whether tf projection layer returns same output
    # Modify initial_vars to use TF compatible params
    tf_initial_vars = py_utils.NestedMap()
    tf_initial_vars.w = initial_vars.linear.w
    tf_initial_vars.b = initial_vars.bias.b
    tf_initial_vars = ToTfNmap(tf_initial_vars)
    tf_p = lingvo_layers.ProjectionLayer.Params().Set(
        name='tf_ffn',
        input_dim=p.input_dims,
        output_dim=p.output_dims,
        batch_norm=False,
        has_bias=True,
        activation=activation)
    tf_ffn = tf_p.Instantiate()
    tf_output = tf_ffn.FProp(tf_initial_vars,
                             tf.constant(inputs, dtype=tf.float32))
    np_outputs = ToNp(outputs)
    tf_np_outputs = ToNp(tf_output)
    self.assertAllClose(tf_np_outputs, np_outputs, atol=1e-6)


if __name__ == '__main__':
  absltest.main()
