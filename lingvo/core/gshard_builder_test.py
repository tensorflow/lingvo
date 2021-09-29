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
from lingvo.core import cluster_factory
from lingvo.core import gshard_builder
from lingvo.core import gshard_layers
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.core import tpu_summary
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


class _MoEBuilder(gshard_builder.MoEBuilder):

  @classmethod
  def Params(cls):
    p = super().Params()
    p.use_xla_dynamic_update_slice = False
    return p


class MoEBuilderTest(test_utils.TestCase):

  def testSetFPropDtypeBfloat16(self):
    builder_p = _MoEBuilder.Params()
    builder_p.cls.SetFPropDtype(builder_p, tf.bfloat16)
    self.assertEqual(tf.bfloat16, builder_p.fprop_dtype)
    self.assertEqual(tf.float32, builder_p.attention_logits_dtype)

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

  def testRepeatLayerUnrolledState(self):
    batch_dim = 2
    length_dim = 4
    model_dim = 8
    key_value_dim = 8
    num_heads = 2
    g = tf.Graph()
    with g.as_default(), self.SetEval(True):
      tf.random.set_seed(398847392)
      _ = py_utils.GetOrCreateGlobalStepVar()  # for DeterministicDropout
      builder = _MoEBuilder.Params().Set(
          deterministic_dropout=True,
          num_devices=FLAGS.num_partitions,
          dropout_rate=0.0,
          model_dim=model_dim,
          attention_key_value_dim=key_value_dim,
          attention_num_heads=num_heads)
      builder = builder.Instantiate()
      # Build a stack of two attention layers.
      p = builder.DecoderLayerStack(
          'decoder',
          sub_layers=[builder.DecSelfAttention('atten')],
          num=4,
          use_repeat_layer=True)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      layer = p.Instantiate()

    with tf.Session(graph=g) as sess, self.SetEval(True):
      np.random.seed(12345)
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

      sess.run(tf.global_variables_initializer())

      fprop_nmap = py_utils.NestedMap(
          vec=tf.convert_to_tensor(inputs),
          segment_id=tf.convert_to_tensor(segment_id),
          segment_pos=tf.convert_to_tensor(segment_pos),
          encoder_output=tf.convert_to_tensor(encoder_output),
          encoder_segment_id=tf.convert_to_tensor(encoder_segment_id),
          encoder_segment_pos=tf.convert_to_tensor(encoder_segment_pos),
          aux_loss=tf.constant(0.0))
      output1 = layer.FPropDefaultTheta(fprop_nmap).vec
      # Get the bias, k, v states from those two attention layers.
      layer_0 = layer.blocks.body_iter_00000.layer_000.atten
      bias1 = layer_0._fprop._named_tensors['qq_bias']
      k_full1 = layer_0._fprop._named_tensors['k_full']
      v_full1 = layer_0._fprop._named_tensors['v_full']

      x1, b1, k1, v1 = sess.run([output1, bias1, k_full1, v_full1])

      # check that output is not random
      x1b = sess.run(output1)
      self.assertAllEqual(x1, x1b)

      # batch-major state
      state = gshard_layers.StateLayer.InitState(layer,
                                                 [batch_dim, 1, length_dim])

      # Increment the state one by one.
      for t in range(length_dim):
        # run with sliced input args using theta with state ('incremental mode')
        theta_with_state = gshard_layers.StateLayer.UpdateTheta(
            layer, layer.theta, state, t)
        tgt_mask = np.zeros([batch_dim, 1, length_dim])
        tgt_mask[:, :, :(t + 1)] = 1
        tgt_mask = tf.convert_to_tensor(tgt_mask.astype(np.float32))
        gshard_layers.OverrideLayer.Set('dec_self_attention_bias',
                                        (tgt_mask - 1.0) * 1e9)
        # slice decoder inputs, as if length_dim=1
        fprop_nmap_t = py_utils.NestedMap(
            vec=tf.convert_to_tensor(inputs[:, t:(t + 1)]),
            segment_id=tf.convert_to_tensor(segment_id[:, t:(t + 1)]),
            segment_pos=tf.convert_to_tensor(segment_pos[:, t:(t + 1)]),
            encoder_output=tf.convert_to_tensor(encoder_output),
            encoder_segment_id=tf.convert_to_tensor(encoder_segment_id),
            encoder_segment_pos=tf.convert_to_tensor(encoder_segment_pos),
            aux_loss=tf.constant(0.0))
        output2 = layer.FProp(theta_with_state, fprop_nmap_t).vec
        # Get the state slices.
        bias2 = layer_0._fprop._named_tensors['bias_full']
        k_full2 = layer_0._fprop._named_tensors['k_full']
        v_full2 = layer_0._fprop._named_tensors['v_full']

        state = gshard_layers.StateLayer.UpdateState(layer, theta_with_state,
                                                     state)

        x2, b2, k2, v2 = sess.run([output2, bias2, k_full2, v_full2])

        self.assertAllClose(x1[:, t], x2[:, 0])
        self.assertAllEqual(b1[:, t, :(t + 1)], b2[:, 0, :(t + 1)])
        self.assertAllClose(k1[:, :(t + 1)], k2[:, :(t + 1)])
        self.assertAllClose(v1[:, :(t + 1)], v2[:, :(t + 1)])
      # Clear the OverrideLayer.
      gshard_layers.OverrideLayer.Clear()

  def _GetInputs(self, reshape_m=False):
    x = tf.constant([[[.1, .2, .3, .4], [.3, .4, .5, .6], [.5, .6, .1, .2]],
                     [[.7, .8, .4, .5], [.9, .1, .2, .3], [.0, .9, .3, .7]]],
                    dtype=tf.float32)
    seg_id = tf.constant([[1, 1, 1], [1, 1, 1]], dtype=tf.int32)
    pos_id = tf.constant([[0, 1, 2], [0, 1, 2]], dtype=tf.int32)
    if reshape_m:
      # Reshape with model_dim_reshape_segments = 2
      x = tf.reshape(x, [2, 3, 2, 2])
    return x, seg_id, pos_id

  def testMoEModelDimReshapeFProp(self):
    """Test to verify MoEBuilder.MoE() supports dynamic shapes.

    Test without this change fails.
    """
    builder = gshard_builder.DenseBuilder.Params().Set(
        e_dim=2,
        c_dim=2,
        deterministic_dropout=True,
        dtype=tf.float32,
        relative_attention_type='bias',
        model_dim=4,
        attention_num_heads=2,
        attention_combine_dims=True,
        attention_num_memory_heads=1,
        model_dim_reshape_segments=2,
        ff_dim=8,
        attention_key_value_dim=2,
        moe_hidden_dim=8).Instantiate()
    p = builder.DecoderLayerStack(
        'decoder',
        sub_layers=[
            builder.DecSelfAttentionRelativeBias('dec_self_attention'),
            builder.MoE('moe', decoder=True)
        ],
        num=2,
        use_repeat_layer=True)

    with self.session(graph=tf.Graph()) as sess:
      tf.random.set_seed(2019)
      # we will reduce the length_dim by 2 dynamically.
      layer = p.Instantiate()
      inputs, segment_ids, segment_pos = self._GetInputs(reshape_m=True)
      dec_inputs = py_utils.NestedMap(
          vec=inputs,
          segment_id=segment_ids,
          segment_pos=segment_pos,
          encoder_output=inputs,
          encoder_segment_id=tf.zeros_like(segment_ids),
          encoder_segment_pos=tf.zeros_like(segment_pos),
          aux_loss=tf.constant(0.0))
      # Verify length dimension shape is dynamic(a Tensor).
      out = layer.FPropDefaultTheta(dec_inputs).vec
      sess.run(tf.global_variables_initializer())
      sess.run([out])

  def testEncNotVisible(self):

    def _Notvisible(x):
      a, b = tf.expand_dims(x, -1), tf.expand_dims(x, -2)
      return tf.cast(
          tf.math.logical_or(
              tf.not_equal(a, b),
              # also ignoring segment_id=0
              tf.math.logical_not(
                  tf.math.logical_or(tf.cast(a, tf.bool), tf.cast(b,
                                                                  tf.bool)))),
          tf.float32)

    builder = gshard_builder.DenseBuilder.Params().Set(
        dtype=tf.float32).Instantiate()
    graph = tf.Graph()
    with graph.as_default():
      segment_ids = tf.convert_to_tensor([[1, 1, 1, 1]], dtype=tf.int32)
      y = builder._EncNotVisible(segment_ids, segment_ids)
      y2 = _Notvisible(segment_ids)
    with self.session(graph=graph) as sess:
      sess.run(tf.global_variables_initializer())
      y_val, y2_val = sess.run([y, y2])
      self.assertAllEqual(y_val, y2_val)

  def testDecNotVisible(self):

    def _Notvisible(seg_id, seg_pos):
      a, b = tf.expand_dims(seg_id, -1), tf.expand_dims(seg_id, -2)
      return tf.cast(
          tf.math.logical_or(
              tf.less(tf.expand_dims(seg_pos, -1), tf.expand_dims(seg_pos, -2)),
              tf.math.logical_or(
                  tf.not_equal(a, b),
                  tf.math.logical_not(
                      tf.math.logical_or(
                          tf.cast(a, tf.bool), tf.cast(b, tf.bool))))),
          tf.float32)

    builder = gshard_builder.DenseBuilder.Params().Set(
        dtype=tf.float32).Instantiate()
    graph = tf.Graph()
    with graph.as_default():
      segment_ids = tf.convert_to_tensor([[1, 1, 1, 1]], dtype=tf.int32)
      segment_pos = tf.convert_to_tensor([[1, 2, 3, 4]], dtype=tf.int32)
      y = builder._DecNotVisible(segment_ids, segment_pos)
      y2 = _Notvisible(segment_ids, segment_pos)
    with self.session(graph=graph) as sess:
      sess.run(tf.global_variables_initializer())
      y_val, y2_val = sess.run([y, y2])
      self.assertAllEqual(y_val, y2_val)

  def testLayerStack(self):
    model_dim = 4
    num_heads = 2
    d_kv = 2
    d_ff = 8
    builder = gshard_builder.DenseBuilder.Params().Set(
        deterministic_dropout=True,
        dtype=tf.float32,
        relative_attention_type='bias',
        model_dim=model_dim,
        attention_num_heads=num_heads,
        attention_combine_dims=True,
        attention_num_memory_heads=1,
        model_dim_reshape_segments=2,
        ff_dim=d_ff,
        attention_key_value_dim=d_kv).Instantiate()

    def _GetOutputs(enc, dec):
      x, seg_id, pos_id = self._GetInputs(reshape_m=True)
      enc_inputs = py_utils.NestedMap(
          vec=x,
          segment_id=seg_id,
          segment_pos=pos_id,
          aux_loss=tf.constant(0.0))
      enc_outs = enc.FPropDefaultTheta(enc_inputs)
      dec_inputs = py_utils.NestedMap(
          vec=x,
          segment_id=seg_id,
          segment_pos=pos_id,
          encoder_output=enc_outs.vec,
          encoder_segment_id=tf.zeros_like(seg_id),
          encoder_segment_pos=tf.zeros_like(pos_id),
          aux_loss=enc_outs.aux_loss)
      return dec.FPropDefaultTheta(dec_inputs).vec

    # Build a graph with RepeatLayer.
    g = tf.Graph()
    with g.as_default():
      tf.random.set_seed(None)
      enc = builder.EncoderLayerStack(
          'encoder',
          sub_layers=[builder.DenseReluDense('ffw')],
          num=2,
          use_repeat_layer=True).Instantiate()
      dec = builder.DecoderLayerStack(
          'decoder',
          sub_layers=[builder.DenseReluDense('ffw', decoder=True)],
          num=2,
          use_repeat_layer=True).Instantiate()
      rep_out = _GetOutputs(enc, dec)

    tf.Session.reset(target='')
    with tf.Session(graph=g) as sess:
      sess.run(tf.global_variables_initializer())
      rep_out = rep_out.eval(session=sess)
      var_values = sess.run(tf.trainable_variables())

    # Build a graph without RepeatLayer.
    g = tf.Graph()
    with g.as_default():
      tf.random.set_seed(None)
      enc = builder.EncoderLayerStack(
          'encoder', sub_layers=[builder.DenseReluDense('ffw')],
          num=2).Instantiate()
      dec = builder.DecoderLayerStack(
          'decoder',
          sub_layers=[builder.DenseReluDense('ffw', decoder=True)],
          num=2).Instantiate()
      dec_out = _GetOutputs(enc, dec)

    tf.Session.reset(target='')
    with tf.Session(graph=g) as sess:
      tf_vars = [
          enc.vars.layer_000.ln.w.scale, enc.vars.layer_000.ffw.w.wi,
          enc.vars.layer_000.ffw.w.wo, enc.vars.layer_001.ln.w.scale,
          enc.vars.layer_001.ffw.w.wi, enc.vars.layer_001.ffw.w.wo,
          enc.vars.final_layer_norm.w.scale, dec.vars.layer_000.ln.w.scale,
          dec.vars.layer_000.ffw.w.wi, dec.vars.layer_000.ffw.w.wo,
          dec.vars.layer_001.ln.w.scale, dec.vars.layer_001.ffw.w.wi,
          dec.vars.layer_001.ffw.w.wo, dec.vars.final_layer_norm.w.scale
      ]
      for val, var in zip(var_values, tf_vars):
        sess.run(tf.assign(var, val))
      dec_out = dec_out.eval(session=sess)
      self.assertAllClose(dec_out, rep_out)

  def testLayerStackSummary(self):
    # In this test we very that summaries created inside stack layers
    # are processed properly with and without RepeatedLayer
    model_dim = 4
    num_heads = 2
    d_kv = 2
    d_ff = 8
    num_experts = 2
    builder = gshard_builder.DenseBuilder.Params().Set(
        deterministic_dropout=True,
        dtype=tf.float32,
        relative_attention_type='bias',
        model_dim=model_dim,
        attention_num_heads=num_heads,
        attention_combine_dims=True,
        attention_num_memory_heads=1,
        model_dim_reshape_segments=None,
        ff_dim=d_ff,
        moe_hidden_dim=d_ff,
        e_dim=num_experts,
        c_dim=1,
        num_groups=num_experts,
        num_devices=num_experts,
        attention_key_value_dim=d_kv).Instantiate()

    def _GetOutputs(enc, dec):
      x, seg_id, pos_id = self._GetInputs()
      enc_inputs = py_utils.NestedMap(
          vec=x,
          segment_id=seg_id,
          segment_pos=pos_id,
          aux_loss=tf.constant(0.0))
      enc_outs = enc.FPropDefaultTheta(enc_inputs)
      dec_inputs = py_utils.NestedMap(
          vec=x,
          segment_id=seg_id,
          segment_pos=pos_id,
          encoder_output=enc_outs.vec,
          encoder_segment_id=tf.zeros_like(seg_id),
          encoder_segment_pos=tf.zeros_like(pos_id),
          aux_loss=enc_outs.aux_loss)
      return dec.FPropDefaultTheta(dec_inputs).vec

    # Build a graph with RepeatLayer unrolled.
    g = tf.Graph()
    with g.as_default(), tpu_summary.context(), cluster_factory.SetEval(
        mode=True):
      tf.random.set_seed(None)
      enc = builder.EncoderLayerStack(
          'encoder',
          sub_layers=[builder.DenseReluDense('ffw')],
          num=2,
          use_repeat_layer=True).Instantiate()
      dec = builder.DecoderLayerStack(
          'decoder',
          sub_layers=[builder.MoE('moe', decoder=True)],
          num=2,
          use_repeat_layer=True).Instantiate()
      rep_unroll_out = _GetOutputs(enc, dec)
      rep_unroll_summary = tpu_summary.merge_all()

    expected_rep_unroll_summary = [
        'index_1/decoder_1/blocks/blocks_body/layer_000/moe/ffw/compute_gating',
        'index_1/decoder_1/blocks/blocks_body_1/layer_000/moe/ffw/compute_gating',
        'over_capacity_1_ratio/decoder_1/blocks/blocks_body/layer_000/moe/ffw/compute_gating/over_capacity',
        'over_capacity_1_ratio/decoder_1/blocks/blocks_body_1/layer_000/moe/ffw/compute_gating/over_capacity',
        'over_capacity_2_ratio/decoder_1/blocks/blocks_body/layer_000/moe/ffw/compute_gating/over_capacity_1',
        'over_capacity_2_ratio/decoder_1/blocks/blocks_body_1/layer_000/moe/ffw/compute_gating/over_capacity_1',
        'top1_expert/decoder_1/blocks/blocks_body/layer_000/moe/ffw/compute_gating',
        'top1_expert/decoder_1/blocks/blocks_body_1/layer_000/moe/ffw/compute_gating'
    ]
    self.assertCountEqual(expected_rep_unroll_summary, rep_unroll_summary)

    tf.Session.reset(target='')
    with tf.Session(graph=g) as sess:
      sess.run(tf.global_variables_initializer())
      rep_unroll_out, rep_unroll_summary = sess.run(
          [rep_unroll_out, rep_unroll_summary])
      var_values = sess.run(tf.trainable_variables())
    # Build a graph without RepeatLayer.
    g = tf.Graph()
    with g.as_default(), tpu_summary.context():
      tf.random.set_seed(None)
      enc = builder.EncoderLayerStack(
          'encoder', sub_layers=[builder.DenseReluDense('ffw')],
          num=2).Instantiate()
      dec = builder.DecoderLayerStack(
          'decoder', sub_layers=[builder.MoE('moe', decoder=True)],
          num=2).Instantiate()
      dec_out = _GetOutputs(enc, dec)
      dec_summary = tpu_summary.merge_all()

    expected_dec_summary = [
        'index_1/decoder_1/layer_000/moe/ffw/compute_gating',
        'index_1/decoder_1/layer_001/moe/ffw/compute_gating',
        'over_capacity_1_ratio/decoder_1/layer_000/moe/ffw/compute_gating/over_capacity',
        'over_capacity_1_ratio/decoder_1/layer_001/moe/ffw/compute_gating/over_capacity',
        'over_capacity_2_ratio/decoder_1/layer_000/moe/ffw/compute_gating/over_capacity_1',
        'over_capacity_2_ratio/decoder_1/layer_001/moe/ffw/compute_gating/over_capacity_1',
        'top1_expert/decoder_1/layer_000/moe/ffw/compute_gating',
        'top1_expert/decoder_1/layer_001/moe/ffw/compute_gating'
    ]
    self.assertCountEqual(expected_dec_summary, dec_summary)

    tf.Session.reset(target='')
    with tf.Session(graph=g) as sess:
      tf_vars = [
          enc.vars.layer_000.ln.w.scale, enc.vars.layer_000.ffw.w.wi,
          enc.vars.layer_000.ffw.w.wo, enc.vars.layer_001.ln.w.scale,
          enc.vars.layer_001.ffw.w.wi, enc.vars.layer_001.ffw.w.wo,
          enc.vars.final_layer_norm.w.scale, dec.vars.layer_000.ln.w.scale,
          dec.vars.layer_000.moe.moe.wi, dec.vars.layer_000.moe.moe.wo,
          dec.vars.layer_000.moe.ffw.top_2_gating.w,
          dec.vars.layer_001.ln.w.scale, dec.vars.layer_001.moe.moe.wi,
          dec.vars.layer_001.moe.moe.wo,
          dec.vars.layer_001.moe.ffw.top_2_gating.w,
          dec.vars.final_layer_norm.w.scale
      ]
      for val, var in zip(var_values, tf_vars):
        sess.run(tf.assign(var, val))
      dec_out, dec_summary = sess.run([dec_out, dec_summary])
      self.assertAllClose(dec_out, rep_unroll_out)

      for name, alt_name in zip(expected_dec_summary,
                                expected_rep_unroll_summary):
        self.assertAllClose(dec_summary[name], rep_unroll_summary[alt_name])

  def testParallelDecSelfAttentionRelativeBiasFFN(self):
    model_dim = 4
    num_heads = 2
    d_kv = 2
    d_ff = 8
    builder = gshard_builder.DenseBuilder.Params().Set(
        dtype=tf.float32,
        relative_attention_type='bias',
        model_dim=model_dim,
        attention_num_heads=num_heads,
        attention_combine_dims=True,
        attention_num_memory_heads=1,
        model_dim_reshape_segments=2,
        ff_dim=d_ff,
        attention_key_value_dim=d_kv).Instantiate()

    # Build a graph with separate attention and ffn layers.
    # Naively compute the output by adding the outputs of the two directly.
    g = tf.Graph()
    with g.as_default():
      tf.random.set_seed(None)
      x, seg_id, pos_id = self._GetInputs(reshape_m=True)
      atten = builder.DecSelfAttentionRelativeBias('atten').Instantiate()
      ffn = builder.DenseReluDenseGated('ffn', tf.nn.relu, True).Instantiate()
      y_atten, _ = atten.FPropDefaultTheta(x, seg_id, pos_id, tf.constant(0),
                                           tf.constant(0), tf.constant(0))
      y_ffn, _ = ffn.FPropDefaultTheta(x, seg_id, pos_id, tf.constant(0),
                                       tf.constant(0), tf.constant(0))
      y_exp = (y_atten + y_ffn) * (2.0**-0.5)
    tf.Session.reset(target='')
    with tf.Session(graph=g) as sess:
      sess.run(tf.global_variables_initializer())
      y_exp = y_exp.eval(session=sess)
      var_values = sess.run(tf.trainable_variables())

    # Build a graph with dedeciated parallel layer and load the variable values.
    # Expect output the same as the previous naive implementation.
    g = tf.Graph()
    with g.as_default():
      x, seg_id, pos_id = self._GetInputs(reshape_m=True)
      parallel = builder.ParallelDecSelfAttentionRelativeBiasFFN(
          'parallel', tf.nn.relu, hidden_dim_reshape_segments=2).Instantiate()
      y_parallel, _ = parallel.FPropDefaultTheta(x, seg_id, pos_id,
                                                 tf.constant(0), tf.constant(0),
                                                 tf.constant(0))
    tf.Session.reset(target='')
    with tf.Session(graph=g) as sess:
      tf_vars = [
          parallel.vars.w_atten.wq, parallel.vars.w_atten.wk,
          parallel.vars.w_atten.wv, parallel.vars.w_atten.wo,
          parallel.vars.wrb.wrb, parallel.vars.w_fflayer.wi_0,
          parallel.vars.w_fflayer.wi_1, parallel.vars.w_fflayer.wo
      ]
      for val, var in zip(var_values, tf_vars):
        sess.run(tf.assign(var, val))
      y_parallel = y_parallel.eval(session=sess)
      self.assertAllClose(y_exp, y_parallel)

  def testEmbedding(self):
    builder = gshard_builder.DenseBuilder.Params().Set(
        model_dim=4, model_dim_reshape_segments=2).Instantiate()
    ids = [[1, 2, 3], [3, 2, 1]]
    graph = tf.Graph()
    with graph.as_default():
      tf.random.set_seed(24332)
      py_utils.GetOrCreateGlobalStepVar()
      emb_layer_p = builder.Embedding('emb', vocab_dim=4)
      emb_layer = emb_layer_p.Instantiate()
      enc_out = emb_layer.FPropDefaultTheta(
          tf.convert_to_tensor(ids, dtype=tf.int32))

    expected_val = [[[[-0.67452705, -2.6386688], [1.1666715, 0.04592554]],
                     [[-1.0561675, -0.48270327], [0.7765603, 0.6768117]],
                     [[0.8349989, 0.67100984], [-0.15557083, 1.275625]]],
                    [[[0.8349989, 0.67100984], [-0.15557083, 1.275625]],
                     [[-1.0561675, -0.48270327], [0.7765603, 0.6768117]],
                     [[-0.67452705, -2.6386688], [1.1666715, 0.04592554]]]]
    with self.session(graph=graph) as sess:
      sess.run(tf.global_variables_initializer())
      enc_out_vals = sess.run(enc_out)
      self.assertAllClose(expected_val, enc_out_vals)


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

  def _testUniTransformerFProp(self, use_moe=False):
    length_dim = 4
    graph = tf.Graph()
    params = gshard_builder.UniTransformer.Params().Set(
        gated_gelu=False,
        moe=use_moe,
        moe_gated_gelu=use_moe,
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
            e_dim=2 if use_moe else None,
            num_groups=1 if use_moe else None,
            c_dim=2 if use_moe else None,
            model_dim=32,
            attention_num_heads=8,
            moe_hidden_dim=128,
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
      golden_float = 5.761248 if use_moe else 5.635831
      test_utils.CompareToGoldenSingleFloat(self, golden_float, loss_eval)

  def testUniTransformerFProp(self):
    self._testUniTransformerFProp(use_moe=False)

  def testUniTransformerMoEGluFProp(self):
    self._testUniTransformerFProp(use_moe=True)

  def testUniTransformerParallelFProp(self):
    length_dim = 4
    graph = tf.Graph()
    params = gshard_builder.UniTransformer.Params().Set(
        gated_gelu=False,
        gated_ffn_activation=tf.nn.relu,
        positional_embedding=False,
        dtype=tf.float32,
        name='transformer',
        parallel_ffn=True,
        hidden_dim_reshape_segments=2,
        conv_kernel_size=2,
        builder=gshard_builder.RecurrentDenseBuilderParallelDecode.Params().Set(
            device_mesh_shape=[1, 1],
            device_mesh=None,
            relative_attention_num_buckets=32,
            relative_attention_type='bias',
            relative_attention_max_distance=128,
            dtype=tf.float32,
            num_devices=1,  # we call .Split num_devices on axis 0 (batch)
            relative_attention_use_universal_1d_position=True,
            model_dim=32,
            model_dim_reshape_segments=2,
            attention_num_memory_heads=1,
            proj_weight_hdim=2,
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
      test_utils.CompareToGoldenSingleFloat(self, 5.832047, loss_eval)


class BertModelTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    tf.random.set_seed(12356)

  def _PreLoadInput(self):
    src = py_utils.NestedMap(
        ids=tf.constant([[2, 3, 1, 2, 4, 1, 4, 3]], dtype=tf.int32),
        segment_ids=tf.constant([[1, 1, 1, 1, 1, 1, 1, 0]], dtype=tf.int32),
        segment_pos=tf.constant([[0, 1, 2, 3, 4, 5, 6, 0]], dtype=tf.int32))
    return src

  def _BertTransformerParams(self):
    length_dim = 8
    use_moe = False  # Make it work for moe case later.
    p = gshard_builder.BertTransformer.Params().Set(
        name='transformer',
        gated_ffn_activation=None,
        moe=use_moe,
        moe_gated_gelu=use_moe,
        positional_embedding=False,
        dtype=tf.float32,
        builder=gshard_builder.DenseBuilder.Params().Set(
            device_mesh_shape=[1, 1],
            device_mesh=None,
            relative_attention_num_buckets=32,
            relative_attention_type='bias',
            relative_attention_max_distance=128,
            dtype=tf.float32,
            num_devices=1,  # we call .Split num_devices on axis 0 (batch)
            relative_attention_use_universal_1d_position=True,
            e_dim=2 if use_moe else None,
            num_groups=1 if use_moe else None,
            c_dim=2 if use_moe else None,
            model_dim=32,
            attention_num_heads=8,
            moe_hidden_dim=128,
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
    return p

  def testBertTransformerFProp(self):
    graph = tf.Graph()
    params = self._BertTransformerParams()
    with graph.as_default():
      py_utils.GetOrCreateGlobalStepVar()
      params.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      tf.random.set_seed(24332)
      bert_model = params.Instantiate()

    with tf.Session(graph=graph) as sess:
      input_batch = self._PreLoadInput()
      metrics_t = bert_model.FPropDefaultTheta(input_batch)[0]
      loss = bert_model.loss
      sess.run(tf.global_variables_initializer())
      num_total_tokens = sess.run(metrics_t['num_total_tokens'][0])
      test_utils.CompareToGoldenSingleFloat(self, 7, num_total_tokens)

      num_masked_tokens = sess.run(metrics_t['num_masked_tokens'][0])
      test_utils.CompareToGoldenSingleFloat(self, 3, num_masked_tokens)

      loss_eval = sess.run(loss)
      golden_float = 5.97803
      test_utils.CompareToGoldenSingleFloat(self, golden_float, loss_eval)


if __name__ == '__main__':
  tf.test.main()
