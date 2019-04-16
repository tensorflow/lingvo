# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for layers_with_gpipe."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.core.layers_with_gpipe import GPipeEvolvedTransformerDecoderLayer
from lingvo.core.layers_with_gpipe import GPipeEvolvedTransformerEncoderLayer
from lingvo.core.layers_with_gpipe import GPipeTransformerLayer
from lingvo.core.layers_with_gpipe import GPipeTransformerStack


class GPipeTransformerLayersTest(test_utils.TestCase):

  def _testInputs(self, depth=3, dtype=tf.float32):
    np.random.seed(505837249)
    source_vecs = tf.stack(
        [tf.constant(np.random.rand(2, depth), dtype=dtype) for _ in range(5)])
    source_padding = tf.constant([[0, 0, 1, 1, 0], [1, 0, 0, 0, 1]],
                                 dtype=dtype)
    aux_source_vecs = tf.stack(
        [tf.constant(np.random.rand(2, depth), dtype=dtype) for _ in range(7)])
    aux_source_paddings = tf.constant(
        [[0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1]], dtype=dtype)
    source_padding = tf.transpose(source_padding)
    aux_source_paddings = tf.transpose(aux_source_paddings)
    return (source_vecs, source_padding, aux_source_vecs, aux_source_paddings)

  def testTransformerLayerExtendStep(self):
    with self.session(use_gpu=True) as sess:
      depth = 4
      np.random.seed(6348575)
      p = GPipeTransformerLayer.Params()
      p.name = 'transformer'
      p.source_dim = depth
      p.has_aux_atten = True
      p.mask_self_atten = True
      p.tr_fflayer_tpl.hidden_dim = 7
      p.tr_atten_tpl.num_attention_heads = 2
      transformer = GPipeTransformerLayer(p)

      (source_vecs, _, aux_vecs, aux_paddings) = self._testInputs(depth=depth)
      source_padding = tf.zeros([5, 2])

      output1 = transformer.FPropDefaultTheta(
          aux_vecs, aux_paddings, source_vecs, source_padding, None, None)
      h1 = output1[2]

      h2 = []
      cached_source_vecs = tf.zeros([0, 2, 4])
      cached_source_contexts = tf.zeros([0, 2, 4])
      prefix_states = py_utils.NestedMap(
          key=cached_source_vecs, value=cached_source_contexts)
      for i in range(5):
        h, _, prefix_states = transformer.ExtendStep(
            transformer.theta, source_vecs[i, :, :], prefix_states, aux_vecs,
            aux_paddings)
        h2.append(h)

      h2 = tf.stack(h2)

      tf.global_variables_initializer().run()
      h1_v, h2_v = sess.run([h1, h2])
      self.assertAllClose(h1_v, h2_v)
      self.assertAllClose(h1_v[2][1],
                          [1.10429943, -1.64884555, 0.15726769, -0.00250494])

  def testEvolvedTransformerEncoderLayerConstruction(self):
    p = GPipeEvolvedTransformerEncoderLayer.Params()
    p.name = 'gpipe_evolved_transformer_encoder'
    p.source_dim = 4
    p.transformer_tpl.tr_fflayer_tpl.hidden_dim = 7
    p.transformer_tpl.tr_atten_tpl.num_attention_heads = 2
    _ = GPipeEvolvedTransformerEncoderLayer(p)

  def testEvolvedTransformerEncoderLayerFProp(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(6348575)
      depth = 4
      p = GPipeEvolvedTransformerEncoderLayer.Params()
      p.name = 'gpipe_evolved_transformer_encoder'
      p.source_dim = depth
      p.transformer_tpl.tr_fflayer_tpl.hidden_dim = 7
      p.transformer_tpl.tr_atten_tpl.num_attention_heads = 2
      transformer = GPipeEvolvedTransformerEncoderLayer(p)

      (source_vecs, source_padding, _, _) = self._testInputs(depth=depth)

      h = transformer.FPropDefaultTheta(source_vecs, source_padding, None)[0]

      tf.global_variables_initializer().run()
      actual_layer_output = sess.run([h])[0]
      tf.logging.info(np.array_repr(actual_layer_output))
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_layer_output = [
          [[-2.03854632, -1.07184005, -0.28417355,  0.17936069],
           [-0.74067241, -1.48318326,  0.26369774,  0.62173623]],
          [[-2.12831736, -0.86353737, -0.54453588,  0.13070297],
           [-0.76326936, -0.04828247, -0.49510449,  1.20852029]],
          [[ 0.85539216, -1.21577334, -1.28910851, -0.15619087],
           [-1.45574117, -1.11208296,  0.71455258,  0.91494167]],
          [[-1.21304905, -1.37239563,  0.7022025 ,  0.16537377],
           [ 3.07106829,  1.35782909, -0.9944036 , -2.28987551]],
          [[-0.13129801, -1.70681071, -0.42324018,  1.32114363],
           [-1.53065133,  0.18422687, -0.93387115,  1.37142754]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_layer_output, actual_layer_output)

  def testEvolvedTransformerDecoderLayerConstruction(self):
    p = GPipeEvolvedTransformerDecoderLayer.Params()
    p.name = 'gpipe_evolved_transformer_decoder'
    p.source_dim = 16
    p.transformer_tpl.tr_atten_tpl.num_attention_heads = 2
    p.has_aux_atten = True
    p.mask_self_atten = True
    _ = GPipeEvolvedTransformerDecoderLayer(p)

  def testEvolvedTransformerDecoderLayerFProp(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(6348575)
      depth = 4
      p = GPipeEvolvedTransformerDecoderLayer.Params()
      p.name = 'gpipe_evolved_transformer_decoder'
      p.source_dim = depth
      p.has_aux_atten = True
      p.mask_self_atten = True
      p.tr_double_heads_atten_tpl.num_attention_heads = 2
      p.tr_atten_tpl.num_attention_heads = 2
      p.transformer_tpl.tr_atten_tpl.num_attention_heads = 2
      transformer = GPipeEvolvedTransformerDecoderLayer(p)

      (source_vecs, source_padding, aux_vecs,
       aux_paddings) = self._testInputs(depth=depth)

      h = transformer.FPropDefaultTheta(aux_vecs, aux_paddings, source_vecs,
                                        source_padding, None, None)[0]

      tf.global_variables_initializer().run()
      actual_layer_output = sess.run([h])[0]
      tf.logging.info(np.array_repr(actual_layer_output))
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_layer_output = [
          [[ 0.5904724 ,  0.05267439,  0.89581013,  0.63010913],
           [ 0.79584485,  0.07670615,  0.40381077,  0.26504567]],
          [[ 0.35448784,  0.28477612,  0.05394353,  0.06531866],
           [ 0.44413447,  0.81940264,  0.98786688,  0.35846332]],
          [[ 0.66811442,  0.07942203,  0.56781054,  0.83598584],
           [ 0.45858502,  0.44949403,  0.06522893,  0.10947803]],
          [[ 0.58166796,  0.94657594,  0.17643142,  0.02062288],
           [ 0.40596515,  0.01996579,  0.93727112,  0.97478259]],
          [[ 0.34873158,  0.0095871 ,  0.34063059,  0.64620447],
           [ 0.70584863,  0.69263214,  0.38247514,  0.28985959]],
          [[ 0.66496903,  0.20383522,  0.35497066,  0.66646087],
           [ 0.0787568 ,  0.26172587,  0.23034802,  0.88751978]],
          [[ 0.68153989,  0.81061888,  0.90142977,  0.87612331],
           [ 0.15129775,  0.56084079,  0.87029755,  0.37908044]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_layer_output, actual_layer_output)

  def testEvolvedTransformerDecoderLayerExtendStep(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(6348575)
      depth = 4
      p = GPipeEvolvedTransformerDecoderLayer.Params()
      p.name = 'gpipe_evolved_transformer_decoder'
      p.source_dim = depth
      p.has_aux_atten = True
      p.mask_self_atten = True
      p.tr_double_heads_atten_tpl.num_attention_heads = 2
      p.tr_atten_tpl.num_attention_heads = 2
      p.transformer_tpl.tr_atten_tpl.num_attention_heads = 2
      et_decoder = GPipeEvolvedTransformerDecoderLayer(p)

      (source_vecs, _, aux_vecs, aux_paddings) = self._testInputs(depth=depth)
      source_padding = tf.zeros([5, 2])

      h1 = et_decoder.FPropDefaultTheta(aux_vecs,
                                        aux_paddings,
                                        source_vecs, source_padding,
                                        None, None)[2]
      h2 = []

      double_head_attention_states = py_utils.NestedMap(
          key=tf.zeros([0, 2, 4]), value=tf.zeros([0, 2, 4]))
      transformer_layer_states = py_utils.NestedMap(
          key=tf.zeros([0, 2, 4]), value=tf.zeros([0, 2, 4]))
      branched_convs_input = tf.zeros([0, 2, 4])

      prefix_states = py_utils.NestedMap(
          double_head_attention_states=double_head_attention_states,
          transformer_layer_states=transformer_layer_states,
          branched_convs_input=branched_convs_input)

      for i in range(5):
        h, _, prefix_states = et_decoder.ExtendStep(
            et_decoder.theta, source_vecs[i, :, :], prefix_states, aux_vecs,
            aux_paddings)
        h2.append(h)

      h2 = tf.stack(h2)

      tf.global_variables_initializer().run()
      h1_v, h2_v = sess.run([h1, h2])
      self.assertAllClose(h1_v, h2_v)


class GPipeTransformerStackTest(test_utils.TestCase, parameterized.TestCase):
  """Tests for GPipeTransformerStack layer."""

  def _TransformerParams(self,
                         num_decoder_layers=0,
                         num_encoder_layers=4,
                         splits=1,
                         num_micro_batches=1):
    model_dim = 2
    params = GPipeTransformerStack.Params()
    params.name = 'transformer'
    params.model_dim = model_dim
    params.num_decoder_layers = num_decoder_layers
    params.decoder_tpl.tr_atten_tpl.num_attention_heads = 1
    params.decoder_tpl.tr_fflayer_tpl.hidden_dim = model_dim
    params.num_encoder_layers = num_encoder_layers
    params.encoder_tpl.tr_atten_tpl.num_attention_heads = 1
    params.encoder_tpl.tr_fflayer_tpl.hidden_dim = model_dim
    params.num_micro_batches = num_micro_batches
    params.splits = splits
    params.random_seed = 0
    params.state_dtype = tf.float32
    return params

  def _TransformerParamsWithEmbeddings(self,
                                       num_decoder_layers=0,
                                       num_encoder_layers=4,
                                       splits=1,
                                       num_micro_batches=1):
    model_dim = 4
    params = GPipeTransformerStack.Params()
    params.name = 'transformer'
    params.model_dim = model_dim
    params.num_decoder_layers = num_decoder_layers
    params.decoder_tpl.tr_atten_tpl.num_attention_heads = 1
    params.decoder_tpl.tr_fflayer_tpl.hidden_dim = model_dim
    params.num_encoder_layers = num_encoder_layers
    params.encoder_tpl.tr_atten_tpl.num_attention_heads = 1
    params.encoder_tpl.tr_fflayer_tpl.hidden_dim = model_dim
    params.num_micro_batches = num_micro_batches
    params.use_pipelined_embeddings = True
    params.state_dtype = tf.float32

    emb_params = params.emb_tpl
    # Default config for the token embedding.
    emb_params.token_emb.use_matmul = True
    emb_params.token_emb.use_3d_weight_tensor = False
    emb_params.token_emb.vocab_size = 10
    emb_params.token_emb.embedding_dim = model_dim

    # Default config for the position embedding.
    emb_params.position_emb.embedding_dim = model_dim
    emb_params.position_emb.trainable_scaling = False
    params.splits = splits
    params.random_seed = 0
    return params

  def _random_inputs(self, batch):
    input_arr = np.array([
        [[0, 1]] * batch,
        [[1, -1]] * batch,
    ])
    paddings_arr = np.array([[0] * batch] * 2)
    tgt_input_arr = np.array([
        [[1, 2]] * batch,
        [[1, -1]] * batch,
        [[2, 1]] * batch,
    ])
    tgt_paddings_arr = np.array([[0] * batch] * 3)
    inputs = tf.constant(input_arr.tolist(), dtype=tf.float32)
    paddings = tf.constant(paddings_arr.tolist(), dtype=tf.float32)
    tgt_inputs = tf.constant(tgt_input_arr.tolist(), dtype=tf.float32)
    tgt_paddings = tf.constant(tgt_paddings_arr.tolist(), dtype=tf.float32)
    return inputs, paddings, tgt_inputs, tgt_paddings

  def _random_inputs_vecs(self, batch):
    input_arr = np.array([[[0, 1, 1, -1]] * batch, [[0, 2, 7, 1]] * batch])
    paddings_arr = np.array([[0] * batch] * 2)
    tgt_input_arr = np.array([[[1, 2, 0, 1]] * batch, [[1, -1, 1, 0]] * batch,
                              [[2, 1, 2, 1]] * batch])
    tgt_paddings_arr = np.array([[0] * batch] * 3)
    inputs = tf.constant(input_arr.tolist(), dtype=tf.float32)
    paddings = tf.constant(paddings_arr.tolist(), dtype=tf.float32)
    tgt_inputs = tf.constant(tgt_input_arr.tolist(), dtype=tf.float32)
    tgt_paddings = tf.constant(tgt_paddings_arr.tolist(), dtype=tf.float32)
    return inputs, paddings, tgt_inputs, tgt_paddings

  def _random_inputs_ids(self, batch):
    input_arr = np.array([[6] * batch, [4] * batch])
    paddings_arr = np.array([[0] * batch] * 2)
    tgt_input_arr = np.array([[3] * batch, [7] * batch, [9] * batch])
    tgt_paddings_arr = np.array([[0] * batch] * 3)
    inputs = tf.constant(input_arr.tolist(), dtype=tf.int32)
    paddings = tf.constant(paddings_arr.tolist(), dtype=tf.float32)
    tgt_inputs = tf.constant(tgt_input_arr.tolist(), dtype=tf.int32)
    tgt_paddings = tf.constant(tgt_paddings_arr.tolist(), dtype=tf.float32)
    return inputs, paddings, tgt_inputs, tgt_paddings

  @parameterized.named_parameters({
      'testcase_name': '_one_split',
      'splits': 1,
      'num_micro_batches': 1
  }, {
      'testcase_name': '_two_splits',
      'splits': 2,
      'num_micro_batches': 2
  }, {
      'testcase_name': '_manual_splits',
      'splits': [4, 8],
      'num_micro_batches': 1
  })
  def testGPipeTransformerEncoderFPropDefaultTheta(self,
                                                   splits=1,
                                                   num_micro_batches=1):
    batch = 4
    tf.flags.FLAGS.tpu_compatible = True
    with self.session() as sess:
      params = self._TransformerParams(
          num_decoder_layers=4,
          num_encoder_layers=4,
          splits=splits,
          num_micro_batches=num_micro_batches)
      params.dtype = tf.float32
      params.fprop_dtype = tf.float32
      xformer = GPipeTransformerStack(params)

      inputs, paddings, _, _ = self._random_inputs(batch)

      output = xformer.EncoderFPropDefaultTheta(inputs, paddings)

      tf.global_variables_initializer().run()
      output = sess.run(output)

      self.assertAllCloseAccordingToType([[[0.21085747, 0.60925347]] * batch,
                                          [[0.21085747, 0.60925347]] * batch],
                                         output)

  @parameterized.named_parameters({
      'testcase_name': '_split1',
      'splits': 1,
      'num_micro_batches': 1
  }, {
      'testcase_name': '_split1_nmb2',
      'splits': 1,
      'num_micro_batches': 2
  }, {
      'testcase_name': '_split1_nmb4',
      'splits': 1,
      'num_micro_batches': 4
  }, {
      'testcase_name': '_split2_nmb1',
      'splits': 2,
      'num_micro_batches': 1
  }, {
      'testcase_name': '_split2_nmb2',
      'splits': 2,
      'num_micro_batches': 2
  }, {
      'testcase_name': '_split2_nmb4',
      'splits': 2,
      'num_micro_batches': 4
  }, {
      'testcase_name': '_split4_nmb1',
      'splits': 4,
      'num_micro_batches': 1
  }, {
      'testcase_name': '_split4_nmb2',
      'splits': 4,
      'num_micro_batches': 2
  }, {
      'testcase_name': '_split4_nmb4',
      'splits': 4,
      'num_micro_batches': 4
  })
  def testGPipeTransformerStackFProp(self, splits=1, num_micro_batches=1):
    batch = 4
    tf.flags.FLAGS.tpu_compatible = True
    with self.session() as sess:
      params = self._TransformerParams(
          splits=splits, num_micro_batches=num_micro_batches)
      params.dtype = tf.float32
      params.fprop_dtype = tf.float32
      xformer = GPipeTransformerStack(params)

      inputs, paddings, _, _ = self._random_inputs(batch)

      output = xformer.FProp(xformer.theta, inputs, paddings)

      tf.global_variables_initializer().run()
      output = sess.run(output)

      self.assertAllCloseAccordingToType([[[0.21085747, 0.60925347]] * batch,
                                          [[0.21085747, 0.60925347]] * batch],
                                         output)

  @parameterized.named_parameters({
      'testcase_name': '_split1',
      'splits': 1,
      'num_micro_batches': 1
  }, {
      'testcase_name': '_split1_nmb2',
      'splits': 1,
      'num_micro_batches': 2
  }, {
      'testcase_name': '_split2_nmb1',
      'splits': 2,
      'num_micro_batches': 1
  }, {
      'testcase_name': '_split2_nmb2',
      'splits': 2,
      'num_micro_batches': 2
  })
  def testGPipeTransformerStackFPropWithEmbeddings(self,
                                                   splits=1,
                                                   num_micro_batches=1):
    batch = 4
    tf.flags.FLAGS.tpu_compatible = True
    with self.session() as sess:
      params = self._TransformerParamsWithEmbeddings(
          splits=splits, num_micro_batches=num_micro_batches)
      params.dtype = tf.float32
      params.fprop_dtype = tf.float32
      xformer = GPipeTransformerStack(params)

      inputs, paddings, _, _ = self._random_inputs_ids(batch)

      output = xformer.FProp(xformer.theta, inputs, paddings)

      tf.global_variables_initializer().run()
      output = sess.run(output)

      self.assertAllCloseAccordingToType(
          [[[-1.67121327, -1.24759686, 1.41572773, 2.42515182]] * batch,
           [[-1.71240354, -1.1253252, 0.23407015, 3.40547156]] * batch], output)

  @parameterized.named_parameters({
      'testcase_name': '_one_split',
      'splits': 1
  }, {
      'testcase_name': '_two_splits',
      'splits': 2
  })
  def testGPipeTransformerFPropPackedInput(self, splits=1):
    batch = 4
    tf.flags.FLAGS.tpu_compatible = True
    with self.session() as sess:
      with tf.variable_scope('transformer_test', reuse=tf.AUTO_REUSE):
        params = self._TransformerParams(splits=splits)
        params.dtype = tf.float32
        params.fprop_dtype = tf.float32
        packed_params = params.Copy()
        packed_params.packed_input = True
        xformer = GPipeTransformerStack(params)
        packed_xformer = GPipeTransformerStack(packed_params)
        # Prepare inputs
        inputs, paddings, tgt_inputs, tgt_paddings = self._random_inputs(batch)
        packed_inputs = tf.reshape(inputs, [-1, 1, 2])
        packed_tgt_inputs = tf.reshape(tgt_inputs, [-1, 1, 2])
        packed_paddings = tf.reshape(paddings, [-1, 1])
        packed_tg_paddings = tf.reshape(tgt_paddings, [-1, 1])
        segment_ids = tf.transpose(
            tf.constant([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=tf.float32))
        tgt_segment_id = tf.transpose(
            tf.constant([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]],
                        dtype=tf.float32))

        output = xformer.FProp(xformer.theta, inputs, paddings, tgt_inputs,
                               tgt_paddings)
        packed_output = packed_xformer.FProp(
            packed_xformer.theta, packed_inputs, packed_paddings,
            packed_tgt_inputs, packed_tg_paddings, segment_ids, tgt_segment_id)
        packed_output = tf.reshape(packed_output, output.shape)

        tf.global_variables_initializer().run()
        output, packed_output = sess.run([output, packed_output])
        self.assertAllClose(output, packed_output)

  @parameterized.named_parameters({
      'testcase_name': '_one_split',
      'splits': 1
  }, {
      'testcase_name': '_two_splits',
      'splits': 2
  })
  def testGPipeTransformerFPropPackedInputWithEmbeddings(self, splits=1):
    batch = 4
    tf.flags.FLAGS.tpu_compatible = True
    with self.session() as sess:
      with tf.variable_scope('transformer_test', reuse=tf.AUTO_REUSE):
        params = self._TransformerParamsWithEmbeddings(
            splits=splits, num_decoder_layers=2)
        params.dtype = tf.float32
        params.fprop_dtype = tf.float32
        packed_params = params.Copy()
        packed_params.packed_input = True
        xformer = GPipeTransformerStack(params)
        packed_xformer = GPipeTransformerStack(packed_params)
        # Prepare inputs
        inputs, paddings, tgt_inputs, tgt_paddings = self._random_inputs_ids(
            batch)
        packed_inputs = tf.reshape(inputs, [-1, 1])
        packed_tgt_inputs = tf.reshape(tgt_inputs, [-1, 1])
        packed_paddings = tf.reshape(paddings, [-1, 1])
        packed_tg_paddings = tf.reshape(tgt_paddings, [-1, 1])
        segment_ids = tf.transpose(
            tf.constant([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=tf.float32))
        tgt_segment_id = tf.transpose(
            tf.constant([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]],
                        dtype=tf.float32))
        segment_pos_id = tf.transpose(
            tf.constant([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=tf.int32))
        tgt_segment_pos_id = tf.transpose(
            tf.constant([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]], dtype=tf.int32))

        output = xformer.FProp(xformer.theta, inputs, paddings, tgt_inputs,
                               tgt_paddings)
        packed_output = packed_xformer.FProp(packed_xformer.theta,
                                             packed_inputs, packed_paddings,
                                             packed_tgt_inputs,
                                             packed_tg_paddings, segment_ids,
                                             tgt_segment_id, segment_pos_id,
                                             tgt_segment_pos_id)
        packed_output = tf.reshape(packed_output, output.shape)

        tf.global_variables_initializer().run()
        output, packed_output = sess.run([output, packed_output])
        self.assertAllClose(output, packed_output, rtol=1e-05, atol=1e-05)

  @parameterized.named_parameters({
      'testcase_name': '_split1',
      'splits': 1,
      'num_micro_batches': 1
  }, {
      'testcase_name': '_split1_nmb2',
      'splits': 1,
      'num_micro_batches': 2
  }, {
      'testcase_name': '_split1_nmb4',
      'splits': 1,
      'num_micro_batches': 4
  }, {
      'testcase_name': '_split2_nmb2',
      'splits': 2,
      'num_micro_batches': 2
  }, {
      'testcase_name': '_split_manual_nmb2',
      'splits': [3, 4],
      'num_micro_batches': 2
  }, {
      'testcase_name': '_split4_nmb2',
      'splits': 4,
      'num_micro_batches': 2
  }, {
      'testcase_name': '_split4_nmb4',
      'splits': 4,
      'num_micro_batches': 4
  })
  def testGPipeTransformerStackTrainTransparentFProp(self,
                                                     splits=1,
                                                     num_micro_batches=1):
    # time = 2,
    batch = 4
    with self.session() as sess:
      params = self._TransformerParams(
          splits=splits,
          num_micro_batches=num_micro_batches,
          num_decoder_layers=3,
          num_encoder_layers=1)
      params.is_transparent = True
      params.num_transparent_outputs = 3
      params.transparent_merger_dropout_prob = 0.0
      xformer = GPipeTransformerStack(params)

      inputs, paddings, tgt_inputs, tgt_paddings = self._random_inputs(
          batch=batch)
      py_utils.GetGlobalStep()
      tf.set_random_seed(1234)
      tf.global_variables_initializer().run()
      enc_outputs = xformer.EncoderFPropDefaultTheta(inputs, paddings)
      dec_output = xformer.FProp(xformer.theta, inputs, paddings, tgt_inputs,
                                 tgt_paddings)
      enc_out_1, enc_out_2, enc_out_3 = sess.run(enc_outputs)
      dec_out = sess.run(dec_output)
      self.assertAllClose(enc_out_1, enc_out_2)
      self.assertAllClose(enc_out_2, enc_out_3)
      self.assertAllClose(enc_out_1, [[[-0.27896273, 1.46589136]] * batch,
                                      [[1.03141928, -0.847896]] * batch])
      self.assertAllClose(
          dec_out,
          [[[2.926736, -4.090812]] * batch, [[-1.69508219, 1.75891459]] * batch,
           [[-1.6950829, 1.75891507]] * batch])

  @parameterized.named_parameters({
      'testcase_name': '_split1',
      'splits': 1,
      'num_micro_batches': 1
  }, {
      'testcase_name': '_split1_nmb2',
      'splits': 1,
      'num_micro_batches': 2
  }, {
      'testcase_name': '_split2_nmb2',
      'splits': 2,
      'num_micro_batches': 2
  }, {
      'testcase_name': '_split_manual_nmb2',
      'splits': [3, 4],
      'num_micro_batches': 2
  })
  def testGPipeTransformerStackTrainTransparentFPropWithEmbeddings(
      self, splits=1, num_micro_batches=1):
    # time = 2,
    batch = 4
    with self.session() as sess:
      params = self._TransformerParamsWithEmbeddings(
          splits=splits,
          num_micro_batches=num_micro_batches,
          num_decoder_layers=3,
          num_encoder_layers=1)
      params.is_transparent = True
      params.num_transparent_outputs = 3
      params.transparent_merger_dropout_prob = 0.0
      xformer = GPipeTransformerStack(params)

      input_ids, id_paddings, tgt_inputs, tgt_paddings = self._random_inputs_ids(
          batch=batch)
      inputs, paddings, _, _ = self._random_inputs_vecs(batch=batch)
      py_utils.GetGlobalStep()
      tf.set_random_seed(1234)
      tf.global_variables_initializer().run()
      enc_outputs = xformer.EncoderFPropDefaultTheta(inputs, paddings)
      dec_output = xformer.FProp(xformer.theta, input_ids, id_paddings,
                                 tgt_inputs, tgt_paddings)
      enc_out_1, enc_out_2, enc_out_3 = sess.run(enc_outputs)
      dec_out = sess.run(dec_output)
      self.assertAllClose(enc_out_1, enc_out_2)
      self.assertAllClose(enc_out_2, enc_out_3)
      self.assertAllClose(
          [[[0.68660116, 0.947429, 0.78953624, -1.20142817]] * batch,
           [[0.57919669, 1.12979364, 4.29336643, 0.45106331]] * batch],
          enc_out_1)
      self.assertAllClose(
          [[[-0.46651918, -1.62957835, 1.15657926, 1.08397353]] * batch,
           [[-0.34674695, -1.65999401, 1.08431196, 1.07384491]] * batch,
           [[-0.41073492, -1.60431314, 1.04607999, 1.08858371]] * batch],
          dec_out)

  @parameterized.named_parameters({
      'testcase_name': '_split1',
      'splits': 1,
      'num_micro_batches': 1
  }, {
      'testcase_name': '_split1_nmb2',
      'splits': 1,
      'num_micro_batches': 2
  }, {
      'testcase_name': '_split1_nmb4',
      'splits': 1,
      'num_micro_batches': 4
  }, {
      'testcase_name': '_split2_nmb2',
      'splits': 2,
      'num_micro_batches': 2
  }, {
      'testcase_name': '_split_manual_nmb2',
      'splits': [1, 4],
      'num_micro_batches': 2
  }, {
      'testcase_name': '_split2_nmb4',
      'splits': 2,
      'num_micro_batches': 4
  }, {
      'testcase_name': '_split4_nmb4',
      'splits': 2,
      'num_micro_batches': 4
  })
  def testGPipeTransformerStackTrainEncoderTransparentFProp(
      self, splits=1, num_micro_batches=1):
    # time = 2,
    batch = 4
    with self.session() as sess:
      params = self._TransformerParams(
          splits=splits,
          num_micro_batches=num_micro_batches,
          num_decoder_layers=2,
          num_encoder_layers=2)
      params.is_transparent = True
      params.num_transparent_outputs = 1
      params.transparent_merger_dropout_prob = 0.0
      xformer = GPipeTransformerStack(params)

      inputs, paddings, tgt_inputs, tgt_paddings = self._random_inputs(
          batch=batch)
      py_utils.GetGlobalStep()
      tf.set_random_seed(1234)
      tf.global_variables_initializer().run()
      enc_output = xformer.EncoderFPropDefaultTheta(inputs, paddings)
      dec_output = xformer.FProp(xformer.theta, inputs, paddings, tgt_inputs,
                                 tgt_paddings)
      enc_out = sess.run(enc_output)
      dec_out = sess.run(dec_output)
      self.assertAllClose(
          enc_out,
          [[[-0.118476, 1.031626]] * batch, [[0.643884, -1.02581167]] * batch])
      self.assertAllClose(
          dec_out,
          [[[-2.8764534, 1.00808454]] * batch, [[1.02129495, -0.78406084]] *
           batch, [[1.02129495, -0.78406084]] * batch])

  def testGPipeTransformerStackTrainTransparentFPropEval(self):
    # time = 2,
    batch = 4
    with self.session() as sess:
      params = self._TransformerParams(
          num_decoder_layers=3, num_encoder_layers=1)
      params.is_transparent = True
      params.num_transparent_outputs = 3
      params.is_eval = True

      xformer = GPipeTransformerStack(params)

      inputs, paddings, _, _ = self._random_inputs(batch=batch)

      tf.global_variables_initializer().run()
      enc_outputs = xformer.EncoderFPropDefaultTheta(inputs, paddings)
      enc_out = sess.run(enc_outputs)
      self.assertAllClose(enc_out,
                          [[[[-0.27896273] * 3, [1.46589136] * 3]] * batch,
                           [[[1.03141928] * 3, [-0.847896] * 3]] * batch])

  def testGPipeTransformerStackTrainEncoderTransparentFPropEval(self):
    # time = 2,
    batch = 4
    with self.session() as sess:
      params = self._TransformerParams(
          num_decoder_layers=3, num_encoder_layers=3)
      params.is_transparent = True
      params.num_transparent_outputs = 1
      params.is_eval = True

      xformer = GPipeTransformerStack(params)

      inputs, paddings, _, _ = self._random_inputs(batch=batch)

      tf.global_variables_initializer().run()
      enc_outputs = xformer.EncoderFPropDefaultTheta(inputs, paddings)
      enc_out = sess.run(enc_outputs)
      self.assertAllClose(enc_out, [[[0.18823329, 0.71548849]] * batch,
                                    [[0.76032472, -0.82791042]] * batch])

  @parameterized.named_parameters({
      'testcase_name': '_split1',
      'splits': 1,
      'num_micro_batches': 1
  }, {
      'testcase_name': '_split1_nmb2',
      'splits': 1,
      'num_micro_batches': 2
  }, {
      'testcase_name': '_split1_nmb4',
      'splits': 1,
      'num_micro_batches': 4
  }, {
      'testcase_name': '_split2_nmb1',
      'splits': 2,
      'num_micro_batches': 1
  }, {
      'testcase_name': '_split2_nmb2',
      'splits': 2,
      'num_micro_batches': 2
  }, {
      'testcase_name': '_split2_nmb4',
      'splits': 2,
      'num_micro_batches': 4
  }, {
      'testcase_name': '_split4_nmb1',
      'splits': 4,
      'num_micro_batches': 1
  }, {
      'testcase_name': '_split4_nmb2',
      'splits': 4,
      'num_micro_batches': 2
  }, {
      'testcase_name': '_split4_nmb4',
      'splits': 4,
      'num_micro_batches': 4
  })
  def testGPipeTransformerDecoderStackFProp(self, splits=1,
                                            num_micro_batches=1):
    batch = 4
    tf.flags.FLAGS.tpu_compatible = True
    with self.session() as sess:
      params = self._TransformerParams(
          num_decoder_layers=4,
          num_encoder_layers=0,
          splits=splits,
          num_micro_batches=num_micro_batches)
      params.dtype = tf.float32
      params.fprop_dtype = tf.float32
      xformer = GPipeTransformerStack(params)

      inputs, paddings, tgt_inputs, tgt_paddings = self._random_inputs(batch)

      output = xformer.FProp(xformer.theta, inputs, paddings, tgt_inputs,
                             tgt_paddings)

      tf.global_variables_initializer().run()
      output_val = sess.run(output)
      self.assertAllCloseAccordingToType([[[1.03550637, -1.3199079]] * batch,
                                          [[-3.36382699, -0.74492991]] * batch,
                                          [[-3.36382723, -0.74492997]] * batch],
                                         output_val)

  @parameterized.named_parameters({
      'testcase_name': '_split1',
      'splits': 1,
      'num_micro_batches': 1
  }, {
      'testcase_name': '_split1_nmb2',
      'splits': 1,
      'num_micro_batches': 2
  }, {
      'testcase_name': '_split2_nmb1',
      'splits': 2,
      'num_micro_batches': 1
  }, {
      'testcase_name': '_split2_nmb2',
      'splits': 2,
      'num_micro_batches': 2
  })
  def testGPipeTransformerDecoderStackFPropWithEmbeddings(
      self, splits=1, num_micro_batches=1):
    batch = 4
    tf.flags.FLAGS.tpu_compatible = True
    with self.session() as sess:
      params = self._TransformerParamsWithEmbeddings(
          num_decoder_layers=4,
          num_encoder_layers=0,
          splits=splits,
          num_micro_batches=num_micro_batches)
      params.dtype = tf.float32
      xformer = GPipeTransformerStack(params)

      inputs, paddings, tgt_inputs, tgt_paddings = self._random_inputs_ids(
          batch)

      output = xformer.FProp(xformer.theta, inputs, paddings, tgt_inputs,
                             tgt_paddings)

      tf.global_variables_initializer().run()
      output_val = sess.run(output)
      self.assertAllCloseAccordingToType(
          [[[-2.29650807, 0.25992393, 1.81951356, 1.52897644]] * batch,
           [[-2.14101386, 0.32607365, 1.73413348, 1.51806736]] * batch,
           [[-2.18863297, 0.34420109, 1.65913653, 1.58703828]] * batch],
          output_val)


if __name__ == '__main__':
  tf.test.main()
