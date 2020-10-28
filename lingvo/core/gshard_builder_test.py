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
from lingvo.core import moe_layers
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
      state = moe_layers.StateLayer.InitState(layer, [batch_dim, 1, length_dim])

      for t in range(length_dim):
        # slice decoder inputs, as if length_dim=1
        np_fprop_args_t = (inputs[:, t:(t + 1)], segment_id[:, t:(t + 1)],
                           segment_pos[:, t:(t + 1)], encoder_output,
                           encoder_segment_id, encoder_segment_pos)
        fprop_args_t = map(tf.convert_to_tensor, np_fprop_args_t)

        # run with sliced input args using theta with state ('incremental mode')
        theta_with_state = moe_layers.StateLayer.UpdateTheta(
            layer, layer.theta, state, t)
        tgt_mask = np.zeros([batch_dim, 1, length_dim])
        tgt_mask[:, :, :(t + 1)] = 1
        tgt_mask = tf.convert_to_tensor(tgt_mask.astype(np.float32))
        moe_layers.OverrideLayer.Set('dec_self_attention_bias',
                                     (tgt_mask - 1.0) * 1e9)
        output2, _ = layer.FProp(theta_with_state, *fprop_args_t)
        bias2 = layer._fprop._named_tensors['bias_full']
        k_full2 = layer._fprop._named_tensors['k_full']
        v_full2 = layer._fprop._named_tensors['v_full']
        state = moe_layers.StateLayer.UpdateState(layer, theta_with_state,
                                                  state)

        x2, b2, k2, v2 = sess.run([output2, bias2, k_full2, v_full2])

        self.assertAllEqual(b1[:, t, :(t + 1)], b2[:, 0, :(t + 1)])
        self.assertAllClose(x1[:, t], x2[:, 0])
        self.assertAllClose(k1[:, :(t + 1)], k2[:, :(t + 1)])
        self.assertAllClose(v1[:, :(t + 1)], v2[:, :(t + 1)])


if __name__ == '__main__':
  tf.test.main()
