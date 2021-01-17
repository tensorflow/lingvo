# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
r"""Test code for Mixture-of-Experts builder."""

from lingvo import compat as tf
from lingvo.core import gshard_builder
from lingvo.core import gshard_layers
from lingvo.core import py_utils
from lingvo.core import test_utils

import numpy as np

tf.flags.DEFINE_integer('num_partitions', 2, 'Number of partitions')

FLAGS = tf.flags.FLAGS


class FakeMoEBuilder(gshard_builder.MoEBuilder):

  def SharedEncBiasWeights(self, name):
    p = self.params
    return self._Var(
        name=name,
        shared_var_collection_suffix='shared_var',
        weights=[('bias',
                  py_utils.WeightParams(
                      shape=[p.model_dim],
                      init=py_utils.WeightInit.Constant(1.0),
                      collections=['_lingvo_enc_bias_gshard_shared_var'],
                      dtype=p.dtype))])

  def FakeLayer(self, name):
    """Returns the Softmax layer with optional label smoothing."""
    return self._Graph(name, ['i'], ['o'],
                       ('->w', self.SharedEncBiasWeights('w')),
                       ('i,w->o', self._Fn('add', lambda x, w: x + w)))


class MoEBuilderTest(test_utils.TestCase):

  def testSharedEncBiasWeights(self):
    model_dim = 4
    key_value_dim = 2
    num_heads = 2
    g = tf.Graph()
    with g.as_default(), self.SetEval(True):
      _ = py_utils.GetOrCreateGlobalStepVar()  # for DeterministicDropout
      builder = FakeMoEBuilder.Params().Set(
          num_devices=FLAGS.num_partitions,
          dropout_rate=0,
          model_dim=model_dim,
          attention_key_value_dim=key_value_dim,
          attention_num_heads=num_heads)
      builder = builder.Instantiate()
      p = builder._Seq('model', builder.FakeLayer('layer0'),
                       builder.FakeLayer('layer1'))
      layer = p.Instantiate()
      all_vars = tf.trainable_variables()
      tf.logging.info(all_vars)
      self.assertEqual(1, len(all_vars))
    with tf.Session(graph=g) as sess, self.SetEval(True):
      x = tf.ones([model_dim])
      y = layer.FPropDefaultTheta(x)
      sess.run(tf.global_variables_initializer())
      y_val = sess.run(y)
      self.assertAllEqual([3.] * model_dim, y_val)

  def _testDecSelfAttentionState(self):
    batch_dim = 2
    length_dim = 4
    model_dim = 8
    key_value_dim = 8
    num_heads = 2

    g = tf.Graph()
    with g.as_default(), self.SetEval(True):
      _ = py_utils.GetOrCreateGlobalStepVar()  # for DeterministicDropout
      builder = gshard_builder.MoEBuilder.Params().Set(
          num_devices=FLAGS.num_partitions,
          dropout_rate=0.1,
          model_dim=model_dim,
          attention_key_value_dim=key_value_dim,
          attention_num_heads=num_heads)
      builder = builder.Instantiate()
      p = builder.DecSelfAttention('dec_self_attention')
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      layer = p.Instantiate()

    with tf.Session(graph=g) as sess, self.SetEval(True):
      inputs = np.random.normal(size=[batch_dim, length_dim, model_dim]).astype(
          np.float32)
      segment_id = np.zeros([batch_dim, length_dim], np.int32)
      segment_pos = np.zeros([batch_dim, length_dim], np.int32)
      segment_id[:, :] = 1
      segment_pos[:] = range(length_dim)
      encoder_output = np.random.normal(
          size=[batch_dim, length_dim, model_dim]).astype(np.float32)
      encoder_segment_id = np.zeros([batch_dim, length_dim], np.int32)
      encoder_segment_pos = np.zeros([batch_dim, length_dim], np.int32)
      encoder_segment_id[:, :] = 1
      encoder_segment_pos[:] = range(length_dim)

      np_fprop_args = (inputs, segment_id, segment_pos, encoder_output,
                       encoder_segment_id, encoder_segment_pos)

      sess.run(tf.global_variables_initializer())

      fprop_args = map(tf.convert_to_tensor, np_fprop_args)

      output1, _ = layer.FPropDefaultTheta(*fprop_args)
      bias1 = layer._fprop._named_tensors['bias']
      k_full1 = layer._fprop._named_tensors['k_full']
      v_full1 = layer._fprop._named_tensors['v_full']

      x1, b1, k1, v1 = sess.run([output1, bias1, k_full1, v_full1])

      # check that output is not random
      x1b = sess.run(output1)
      self.assertAllEqual(x1, x1b)

      # batch-major state
      state = gshard_layers.StateLayer.InitState(layer,
                                                 [batch_dim, 1, length_dim])

      for t in range(length_dim):
        # slice decoder inputs, as if length_dim=1
        np_fprop_args_t = (inputs[:, t:(t + 1)], segment_id[:, t:(t + 1)],
                           segment_pos[:, t:(t + 1)], encoder_output,
                           encoder_segment_id, encoder_segment_pos)
        fprop_args_t = map(tf.convert_to_tensor, np_fprop_args_t)

        # run with sliced input args using theta with state ('incremental mode')
        theta_with_state = gshard_layers.StateLayer.UpdateTheta(
            layer, layer.theta, state, t)
        tgt_mask = np.zeros([batch_dim, 1, length_dim])
        tgt_mask[:, :, :(t + 1)] = 1
        tgt_mask = tf.convert_to_tensor(tgt_mask.astype(np.float32))
        gshard_layers.OverrideLayer.Set('dec_self_attention_bias',
                                        (tgt_mask - 1.0) * 1e9)
        output2, _ = layer.FProp(theta_with_state, *fprop_args_t)
        bias2 = layer._fprop._named_tensors['bias_full']
        k_full2 = layer._fprop._named_tensors['k_full']
        v_full2 = layer._fprop._named_tensors['v_full']
        state = gshard_layers.StateLayer.UpdateState(layer, theta_with_state,
                                                     state)

        x2, b2, k2, v2 = sess.run([output2, bias2, k_full2, v_full2])

        self.assertAllEqual(b1[:, t, :(t + 1)], b2[:, 0, :(t + 1)])
        self.assertAllClose(x1[:, t], x2[:, 0])
        self.assertAllClose(k1[:, :(t + 1)], k2[:, :(t + 1)])
        self.assertAllClose(v1[:, :(t + 1)], v2[:, :(t + 1)])

  def _CreateDynamicShapeInputs(self, batch_dim, length_dim, input_dim):
    inputs = tf.random.normal([batch_dim, length_dim, input_dim], seed=92837472)
    # Create segment_ids with random number of 1s and stack 0s at end.
    num_ones = tf.random.uniform(
        shape=(), minval=1, maxval=length_dim, dtype=tf.int32)
    segment_ids = tf.concat([
        tf.ones([batch_dim, num_ones]),
        tf.zeros([batch_dim, length_dim - num_ones])
    ],
                            axis=1)
    # Remove unpadded positions from the end.
    max_seq_len = tf.cast(
        tf.reduce_max(tf.reduce_sum(segment_ids, -1)), tf.int32)
    inputs = inputs[:, :max_seq_len, :]
    segment_ids = segment_ids[:, :max_seq_len]
    unused_segment_pos = tf.zeros_like(segment_ids)
    return inputs, segment_ids, unused_segment_pos

  def testMoEFPropDynamicShapes(self):
    """Test to verify MoEBuilder.MoE() supports dynamic shapes.

    Test without this change fails.
    """
    batch_dim = 2
    length_dim = 4
    input_dim = 4
    builder = gshard_builder.MoEBuilder.Params().Set(
        model_dim=input_dim, num_devices=2, moe_hidden_dim=16, e_dim=2, c_dim=2)
    p = builder.Instantiate().MoE('moe')
    with self.session(graph=tf.Graph()) as sess:
      tf.random.set_seed(2019)
      # we will reduce the length_dim by 2 dynamically.
      layer = p.Instantiate()
      inputs, segment_ids, segment_pos = self._CreateDynamicShapeInputs(
          batch_dim, length_dim, input_dim)
      # Verify length dimension shape is dynamic(a Tensor).
      self.assertIsInstance(py_utils.GetShape(inputs)[1], tf.Tensor)
      out, aux_loss = layer.FPropDefaultTheta(inputs, segment_ids, segment_pos)
      sess.run(tf.global_variables_initializer())
      _ = sess.run([out, aux_loss])

  def testRepeatMoEFProp(self):
    """Test to verify RecurrentDenseBuilder.DecoderLayerStack().

    Test without this change fails.
    """
    batch_dim = 2
    length_dim = 4
    input_dim = 4
    builder = gshard_builder.RecurrentDenseBuilder.Params().Set(
        model_dim=input_dim,
        num_devices=2,
        moe_hidden_dim=16,
        e_dim=2,
        attention_key_value_dim=input_dim,
        attention_num_heads=1,
        c_dim=2,
        emh_split=[-1, 0, -1, -1],
        ehm_split=[-1, 0, -1, -1])
    b = builder.Instantiate()
    layers = [
        b.DecSelfAttention('dec_self_attention'),
        b.MoE('moe', decoder=True)
    ]
    p = b.DecoderLayerStack('rep', layers, 2)
    with self.session(graph=tf.Graph()) as sess:
      tf.random.set_seed(2019)
      # we will reduce the length_dim by 2 dynamically.
      layer = p.Instantiate()
      inputs = tf.ones([batch_dim, length_dim, input_dim])
      segment_ids = tf.ones([batch_dim, length_dim])
      segment_pos = tf.ones([batch_dim, length_dim])
      outputs = layer.FPropDefaultTheta(inputs, segment_ids, segment_pos,
                                        inputs, segment_ids, segment_pos,
                                        tf.zeros([]))
      sess.run(tf.global_variables_initializer())
      sess.run(outputs)


class UniTransformerTest(test_utils.TestCase):

  def _PreLoadInput(self):
    src = py_utils.NestedMap(
        ids=tf.constant([[2, 3, 1, 2], [4, 1, 4, 3]], dtype=tf.int32),
        segment_ids=tf.constant([[1, 1, 1, 2], [1, 1, 2, 2]], dtype=tf.int32),
        segment_pos=tf.constant([[0, 1, 2, 0], [0, 1, 0, 1]], dtype=tf.int32))
    tgt = py_utils.NestedMap(
        labels=tf.constant([[9, 1, 8, 1], [9, 8, 1, 8]], dtype=tf.int32),
        ids=tf.constant([[0, 9, 1, 8], [0, 9, 8, 1]], dtype=tf.int32),
        segment_ids=tf.constant([[1, 1, 2, 2], [1, 1, 1, 2]], dtype=tf.int32),
        segment_pos=tf.constant([[0, 1, 0, 1], [0, 1, 2, 0]], dtype=tf.int32))
    return py_utils.NestedMap(src=src, tgt=tgt)

  def testUniTransformerFProp(self):
    length_dim = 4
    graph = tf.Graph()
    params = gshard_builder.UniTransformer.Params().Set(
        gated_gelu=False,
        positional_embedding=False,
        dtype=tf.float32,
        name='transformer',
        builder=gshard_builder.DenseBuilder.Params().Set(
            device_mesh_shape=[1, 1],
            device_mesh=None,
            relative_attention_num_buckets=32,
            relative_attention_type='bias',
            relative_attention_max_distance=128,
            dtype=tf.float32,
            num_devices=1,  # we call .Split num_devices on axis 0 (batch)
            relative_attention_use_universal_1d_position=True,
            model_dim=32,
            attention_num_heads=8,
            ff_dim=128,
            attention_key_value_dim=8,
            attention_combine_dims=True),
        batch_size=32,
        sequence_length=length_dim,
        num_transformer_layers=2,
        aux_loss_coef=0.0,
        loss_denominator=None,
        label_smoothing=0,
        vocab_size=128,
        max_length=length_dim)
    with graph.as_default():
      py_utils.GetOrCreateGlobalStepVar()
      params.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      tf.random.set_seed(24332)
      model = params.Instantiate()

    with tf.Session(graph=graph) as sess:
      input_batch = self._PreLoadInput()
      loss = model.FPropDefaultTheta(input_batch)[0]['loss'][0]
      sess.run(tf.global_variables_initializer())
      loss_eval = sess.run(loss)
      test_utils.CompareToGoldenSingleFloat(self, 5.635831, loss_eval)


if __name__ == '__main__':
  tf.test.main()
