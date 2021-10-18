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
"""Tests for lingvo Jax embedding and softmax layers."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax import test_util
from lingvo.core import layers as lingvo_layers
from lingvo.jax import py_utils
from lingvo.jax import test_utils
from lingvo.jax.layers import embedding_softmax
import numpy as np
import tensorflow.compat.v2 as tf

ToNp = test_utils.ToNp


class EmbeddingSoftmaxTest(test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  @parameterized.parameters(('index', True), ('index', False), ('matmul', True),
                            ('matmul', False))
  def test_single_sharded_embedding_layer(self, lookup_style, scale_sqrt_depth):
    p = embedding_softmax.SingleShardEmbeddingLayer.Params().Set(
        name='jax_emb_lookup',
        vocab_size=10,
        embedding_dims=40,
        lookup_style=lookup_style,
        scale_sqrt_depth=scale_sqrt_depth)
    emb_layer = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = emb_layer.InstantiateVariables(prng_key)
    npy_input = np.random.randint(0, p.vocab_size, [10, 20]).astype('int32')
    inputs = jnp.asarray(npy_input)
    outputs = emb_layer.FProp(initial_vars, inputs)
    # Test whether tf Embedding layer returns same output
    # Modify initial_vars to use TF compatible params
    tf_initial_vars = initial_vars
    tf_p = lingvo_layers.SingleShardEmbeddingLayer.Params().Set(
        name='tf_emb_lookup',
        vocab_size=p.vocab_size,
        embedding_dim=p.embedding_dims,
        scale_sqrt_depth=scale_sqrt_depth)
    tf_emb_layer = tf_p.Instantiate()
    tf_output = tf_emb_layer.FProp(tf_initial_vars,
                                   tf.constant(inputs, dtype=tf.int32))
    np_outputs = ToNp(outputs)
    tf_np_outputs = ToNp(tf_output)
    self.assertAllClose(tf_np_outputs, np_outputs, atol=1e-6)

  @parameterized.parameters((0., True, False), (0., False, True),
                            (1.0, True, False), (1.0, False, True))
  def test_single_sharded_softmax_layer(self, soft_cap_logits, use_class_ids,
                                        use_class_probabilities):
    if use_class_ids:
      class_ids = np.random.randint(0, 50, [8, 10, 1])
    else:
      class_ids = None
    if use_class_probabilities:
      class_probabilities = np.random.normal(1.5, 2.0, [8, 10, 50])
    else:
      class_probabilities = None
    p = embedding_softmax.SingleShardFullSoftmax.Params().Set(
        name='jax_softmax',
        num_classes=50,
        input_dims=40,
        soft_cap_logits=soft_cap_logits)
    softmax_layer = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=1234)
    initial_vars = softmax_layer.InstantiateVariables(prng_key)
    npy_input = np.random.normal(1.5, 2.0, [8, 10, p.input_dims])
    inputs = jnp.asarray(npy_input)
    class_weights = np.random.normal(1.5, 2.0, [8, 10, 1])
    if class_probabilities is not None:
      class_probabilities /= np.sum(class_probabilities, axis=-1, keepdims=True)
    logits = softmax_layer.GetLogits(initial_vars, inputs)
    outputs = softmax_layer.FProp(
        initial_vars,
        inputs,
        class_weights,
        class_ids=class_ids,
        class_probabilities=class_probabilities)
    # Test whether tf Softmax layer returns same output
    # Modify initial_vars to use TF compatible params
    tf_initial_vars = initial_vars
    tf_initial_vars.linear = py_utils.NestedMap()
    tf_initial_vars.linear.w = initial_vars.logits_ffn.linear.w
    tf_initial_vars.bias = py_utils.NestedMap()
    tf_initial_vars.bias.b = initial_vars.logits_ffn.bias.b
    tf_p = lingvo_layers.SingleShardFullSoftmax.Params().Set(
        name='tf_softmax',
        num_classes=p.num_classes,
        input_dim=p.input_dims,
        logits_soft_max=soft_cap_logits)
    tf_softmax_layer = tf_p.Instantiate()
    tf_logits = tf_softmax_layer.Logits(tf_initial_vars,
                                        tf.constant(inputs, dtype=tf.float32))
    tf_output = tf_softmax_layer.FProp(
        tf_initial_vars,
        tf.constant(inputs, dtype=tf.float32),
        class_weights,
        class_ids=class_ids,
        class_probabilities=class_probabilities)
    # Check all entries in the NestedMap and ensure it matches TF
    np_get_logits = ToNp(logits)
    tf_np_get_logits = ToNp(tf_logits)
    self.assertAllClose(np_get_logits, tf_np_get_logits, atol=1e-6)
    # Note: The argmax-related values are very sensitive to numerical errors.
    for k in outputs.keys():
      self.assertAllClose(ToNp(outputs[k]), ToNp(tf_output[k]), atol=1e-6)

  def test_simple_softmax_layer_class_ids(self):
    batch_size = 8
    num_classes = 50
    class_ids = np.random.randint(0, 50, [8, 1])
    p = embedding_softmax.SingleShardFullSoftmax.Params().Set(
        name='jax_softmax', num_classes=num_classes, input_dims=40)
    softmax_layer = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = softmax_layer.InstantiateVariables(prng_key)
    npy_input = np.random.normal(1.5, 2.0, [batch_size, p.input_dims])
    inputs = jnp.asarray(npy_input)
    class_weights = np.random.normal(1.5, 2.0, [batch_size, 1])
    logits = softmax_layer.GetLogits(initial_vars, inputs)
    outputs = softmax_layer.FProp(
        initial_vars,
        inputs,
        class_weights,
        class_ids=class_ids,
        class_probabilities=None)
    # Test whether tf Softmax layer returns same output.
    # Modify initial_vars to use TF compatible params.
    tf_initial_vars = test_utils.ReplaceJaxSimpleFullSoftmaxVarsToTf(
        initial_vars)
    # Convert all the values to TF tensor.
    tf_initial_vars = tf.nest.map_structure(tf.convert_to_tensor,
                                            tf_initial_vars)

    tf_p = lingvo_layers.SimpleFullSoftmax.Params().Set(
        name='tf_softmax', num_classes=p.num_classes, input_dim=p.input_dims)
    tf_softmax_layer = tf_p.Instantiate()
    tf_logits = tf_softmax_layer.Logits(tf_initial_vars,
                                        tf.constant(inputs, dtype=tf.float32))
    tf_output = tf_softmax_layer.FProp(
        tf_initial_vars,
        tf.constant(inputs, dtype=tf.float32),
        class_weights,
        class_ids=class_ids,
        class_probabilities=None)
    # Check all entries in the NestedMap and ensure it matches TF.
    np_get_logits = ToNp(logits)
    tf_np_get_logits = ToNp(tf_logits)
    self.assertAllClose(np_get_logits, tf_np_get_logits)
    for k in outputs.keys():
      self.assertAllClose(ToNp(outputs[k]), ToNp(tf_output[k]))

  @parameterized.parameters((8, 1001), (16, 1024), (32, 30000))
  def test_simple_softmax_layer_class_probs(self, batch_size, num_classes):
    batch_size = 8
    num_classes = 1001
    class_probabilities = np.random.normal(1.5, 2.0, [batch_size, num_classes])
    # Normalize class probabilities to be a probability distribution.
    class_probabilities /= np.sum(class_probabilities, axis=-1, keepdims=True)
    p = embedding_softmax.SingleShardFullSoftmax.Params().Set(
        name='jax_softmax', num_classes=num_classes, input_dims=40)
    softmax_layer = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = softmax_layer.InstantiateVariables(prng_key)
    npy_input = np.random.normal(1.5, 2.0, [batch_size, p.input_dims])
    inputs = jnp.asarray(npy_input)
    class_weights = np.random.normal(1.5, 2.0, [batch_size, 1])
    logits = softmax_layer.GetLogits(initial_vars, inputs)
    outputs = softmax_layer.FProp(
        initial_vars,
        inputs,
        class_weights,
        class_ids=None,
        class_probabilities=class_probabilities)
    # Test whether tf Softmax layer returns same output.
    # Modify initial_vars to use TF compatible params.
    tf_initial_vars = test_utils.ReplaceJaxSimpleFullSoftmaxVarsToTf(
        initial_vars)
    # Convert all the values to TF tensor.
    tf_initial_vars = tf.nest.map_structure(tf.convert_to_tensor,
                                            tf_initial_vars)

    tf_p = lingvo_layers.SimpleFullSoftmax.Params().Set(
        name='tf_softmax', num_classes=p.num_classes, input_dim=p.input_dims)
    tf_softmax_layer = tf_p.Instantiate()
    tf_logits = tf_softmax_layer.Logits(tf_initial_vars,
                                        tf.constant(inputs, dtype=tf.float32))
    tf_output = tf_softmax_layer.FProp(
        tf_initial_vars,
        tf.constant(inputs, dtype=tf.float32),
        class_weights,
        class_ids=None,
        class_probabilities=class_probabilities)
    # Check all entries in the NestedMap and ensure it matches TF.
    np_get_logits = ToNp(logits)
    tf_np_get_logits = ToNp(tf_logits)
    self.assertAllClose(np_get_logits, tf_np_get_logits)
    for k in outputs.keys():
      self.assertAllClose(ToNp(outputs[k]), ToNp(tf_output[k]))

  def test_simple_softmax_layer_value_error(self):
    batch_size = 8
    num_classes = 50
    class_ids = None
    class_probabilities = None
    p = embedding_softmax.SingleShardFullSoftmax.Params().Set(
        name='jax_softmax', num_classes=num_classes, input_dims=40)
    softmax_layer = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = softmax_layer.InstantiateVariables(prng_key)
    npy_input = np.random.normal(1.5, 2.0, [batch_size, p.input_dims])
    inputs = jnp.asarray(npy_input)
    class_weights = np.random.normal(1.5, 2.0, [batch_size, 1])
    with self.assertRaises(ValueError):
      _ = softmax_layer.FProp(
          initial_vars,
          inputs,
          class_weights,
          class_ids=class_ids,
          class_probabilities=class_probabilities)

  @parameterized.parameters((0., 'index', True), (0., 'matmul', True),
                            (1.0, 'index', False), (1.0, 'matmul', False))
  def test_single_sharded_shared_embedding_softmax_layer(
      self, soft_cap_logits, lookup_style, scale_sqrt_depth):
    class_ids = np.random.randint(1, 50, [8, 10, 1])
    p = embedding_softmax.SingleShardSharedEmbeddingSoftmax.Params().Set(
        name='jax_softmax',
        num_classes=50,
        input_dims=40,
        soft_cap_logits=soft_cap_logits,
        lookup_style=lookup_style,
        scale_sqrt_depth=scale_sqrt_depth)
    softmax_layer = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = softmax_layer.InstantiateVariables(prng_key)
    npy_input = np.random.normal(1.5, 2.0, [8, 10, p.input_dims])
    inputs = jnp.asarray(npy_input)
    class_weights = np.random.normal(1.5, 2.0, [8, 10, 1])
    outputs = softmax_layer.FProp(
        initial_vars, inputs, class_weights, class_ids=class_ids)
    ids = np.squeeze(class_ids, axis=-1)
    emb_lookup_outputs = softmax_layer.EmbLookup(
        initial_vars, ids=jnp.asarray(ids))
    # Test whether tf Softmax layer returns same output
    # Modify initial_vars to use TF compatible params
    tf_initial_vars = initial_vars
    tf_initial_vars.linear = py_utils.NestedMap()
    tf_initial_vars.linear.w = initial_vars.logits_ffn.linear.w
    tf_initial_vars.bias = py_utils.NestedMap()
    tf_initial_vars.bias.b = initial_vars.logits_ffn.bias.b
    tf_p = lingvo_layers.SingleShardSharedEmbeddingSoftmax.Params().Set(
        name='tf_softmax',
        num_classes=p.num_classes,
        input_dim=p.input_dims,
        vocab_size=p.num_classes,
        embedding_dim=p.input_dims,
        logits_soft_max=soft_cap_logits,
        scale_sqrt_depth=scale_sqrt_depth)
    tf_softmax_layer = tf_p.Instantiate()
    tf_output = tf_softmax_layer.FProp(
        tf_initial_vars,
        tf.constant(inputs, dtype=tf.float32),
        class_weights,
        class_ids=class_ids)
    tf_emb_lookup_output = tf_softmax_layer.EmbLookup(
        tf_initial_vars, ids=tf.constant(ids))

    # Check all entries in the NestedMap and ensure it matches TF
    np_logits = ToNp(outputs.logits)
    tf_np_logits = ToNp(tf_output.logits)
    self.assertAllClose(np_logits, tf_np_logits, atol=1e-6)
    for k in outputs.keys():
      self.assertAllClose(ToNp(outputs[k]), ToNp(tf_output[k]), atol=1e-6)
    np_emb_lookup_output = ToNp(emb_lookup_outputs)
    tf_np_emb_lookup_output = ToNp(tf_emb_lookup_output)
    self.assertAllClose(
        tf_np_emb_lookup_output, np_emb_lookup_output, atol=1e-6)

  @parameterized.parameters((1, 10), (1, 1e5), (10, 20), (10, 1e5))
  def test_position_embedding_layer(self, min_timescale, max_timescale):
    p = embedding_softmax.PositionalEmbeddingLayer.Params().Set(
        name='jax_pos',
        embedding_dims=50,
        min_timescale=min_timescale,
        max_timescale=max_timescale)
    pos_layer = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = pos_layer.InstantiateVariables(prng_key)
    seq_length = np.random.randint(100, 1000)
    output = pos_layer.FProp(initial_vars, seq_length)
    output = jnp.squeeze(output, axis=0)
    # Test whether tf PositionalEmbedding layer returns same output
    # Modify initial_vars to use TF compatible params
    tf_initial_vars = initial_vars
    tf_p = lingvo_layers.PositionalEmbeddingLayer.Params().Set(
        name='tf_pos',
        embedding_dim=p.embedding_dims,
        min_timescale=min_timescale,
        max_timescale=max_timescale)
    tf_pos_layer = tf_p.Instantiate()
    tf_output = tf_pos_layer.FProp(tf_initial_vars, seq_length)
    np_pos = ToNp(output)
    tf_np_pos = ToNp(tf_output)
    self.assertAllClose(tf_np_pos, np_pos, atol=1e-3)

  @parameterized.parameters((1, 10), (1, 1e5), (10, 20), (10, 1e5))
  def test_position_embedding_layer_with_position(self, min_timescale,
                                                  max_timescale):
    p = embedding_softmax.PositionalEmbeddingLayer.Params().Set(
        name='jax_pos',
        embedding_dims=50,
        min_timescale=min_timescale,
        max_timescale=max_timescale)
    pos_layer = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = pos_layer.InstantiateVariables(prng_key)
    position = np.array([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                         [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
                         [0, 1, 2, 3, 4, 5, 6, 0, 1, 2],
                         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    output = pos_layer.FProp(initial_vars, position=position)
    # Test whether tf PositionalEmbedding layer returns same output
    # Modify initial_vars to use TF compatible params
    tf_initial_vars = initial_vars
    tf_p = lingvo_layers.PositionalEmbeddingLayer.Params().Set(
        name='tf_pos',
        embedding_dim=p.embedding_dims,
        min_timescale=min_timescale,
        max_timescale=max_timescale)
    tf_pos_layer = tf_p.Instantiate()
    tf_output = tf_pos_layer.FPropWithPosition(tf_initial_vars, position)
    np_pos = ToNp(output)
    tf_np_pos = ToNp(tf_output)
    self.assertAllClose(tf_np_pos, np_pos, atol=1e-3)


if __name__ == '__main__':
  absltest.main()
