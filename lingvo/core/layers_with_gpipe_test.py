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
import lingvo.compat as tf
from lingvo.core import layers_with_gpipe
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.core.layers_with_gpipe import GPipeEvolvedTransformerDecoderLayer
from lingvo.core.layers_with_gpipe import GPipeEvolvedTransformerEncoderLayer
from lingvo.core.layers_with_gpipe import GPipeEvolvedTransformerStack
from lingvo.core.layers_with_gpipe import GPipeTransformerLayer
from lingvo.core.layers_with_gpipe import GPipeTransformerStack
import numpy as np
from six.moves import range


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

  def testGPipeSoftmaxLayerInputfromEncoder(self):
    with self.session(use_gpu=True):
      depth = 4
      np.random.seed(6348575)
      p = GPipeTransformerLayer.Params()
      p.name = 'transformer'
      p.source_dim = depth
      p.has_aux_atten = False
      p.tr_fflayer_tpl.hidden_dim = 7
      p.tr_atten_tpl.num_attention_heads = 2
      transformer = p.Instantiate()
      softmax = layers_with_gpipe.GPipeTransformerSoftmaxLayer.Params()
      softmax.name = 'softmax'
      softmax.inputs_from_decoder = False
      softmax.softmax.num_classes = 2
      softmax.softmax.input_dim = depth
      softmax = softmax.Instantiate()
      (source_vecs, _, _, _) = self._testInputs(depth=depth)
      source_padding = tf.zeros([5, 2])
      labels = tf.ones([5, 2], dtype=tf.int32)
      label_weights = tf.ones([5, 2])
      softmax_inputs = transformer.FPropDefaultTheta(source_vecs,
                                                     source_padding, None, None,
                                                     None, None, labels,
                                                     label_weights, None, None)
      softmax_outputs = softmax.FPropDefaultTheta(*softmax_inputs)
      self.assertEqual([5, 2], softmax_outputs[0].shape)
      self.assertEqual([5, 2, 2], softmax_outputs[1].shape)

  def testGPipeSoftmaxLayerInputfromDecoder(self):
    with self.session(use_gpu=True):
      depth = 4
      np.random.seed(6348575)
      p = GPipeTransformerLayer.Params()
      p.name = 'transformer'
      p.source_dim = depth
      p.has_aux_atten = True
      p.mask_self_atten = True
      p.tr_fflayer_tpl.hidden_dim = 7
      p.tr_atten_tpl.num_attention_heads = 2
      transformer = p.Instantiate()
      softmax = layers_with_gpipe.GPipeTransformerSoftmaxLayer.Params()
      softmax.name = 'softmax'
      softmax.inputs_from_decoder = True
      softmax.softmax.num_classes = 2
      softmax.softmax.input_dim = depth
      softmax = softmax.Instantiate()
      (source_vecs, _, aux_vecs, aux_paddings) = self._testInputs(depth=depth)
      source_padding = tf.zeros([5, 2])
      labels = tf.ones([5, 2], dtype=tf.int32)
      label_weights = tf.ones([5, 2])
      softmax_inputs = transformer.FPropDefaultTheta(aux_vecs, aux_paddings,
                                                     source_vecs,
                                                     source_padding, None, None,
                                                     labels, label_weights,
                                                     None, None)
      softmax_outputs = softmax.FPropDefaultTheta(*softmax_inputs)
      self.assertEqual([5, 2], softmax_outputs[0].shape)
      self.assertEqual([5, 2, 2], softmax_outputs[1].shape)

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

      output1 = transformer.FPropDefaultTheta(aux_vecs, aux_paddings,
                                              source_vecs, source_padding, None,
                                              None, None, None, None, None)
      h1 = output1[2]

      h2 = []
      cached_source_vecs = tf.zeros([0, 2, 4])
      cached_source_contexts = tf.zeros([0, 2, 4])
      prefix_states = py_utils.NestedMap(
          key=cached_source_vecs, value=cached_source_contexts)
      for i in range(5):
        h, _, prefix_states = transformer.ExtendStep(transformer.theta,
                                                     source_vecs[i, :, :],
                                                     prefix_states, aux_vecs,
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

      h = transformer.FPropDefaultTheta(source_vecs, source_padding, None, None,
                                        None, None, None, None, None, None)[0]

      tf.global_variables_initializer().run()
      actual_layer_output = sess.run([h])[0]
      tf.logging.info(np.array_repr(actual_layer_output))
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_layer_output = [
          [[-2.05546284, -1.15067506, -0.00898855, 0.26481438],
           [-0.8181392, -1.40835416, 0.47274107, 0.86176264]],
          [[-1.43251371,  0.69686228, -0.70026731, -0.47239268],
           [-0.39946821, -0.27037358, -0.22701442, 1.33816898]],
          [[ 0.89412129, -1.07294774, -0.86541933, -0.21121001],
           [-1.55683649, -1.14919782, 0.95287859, 0.11334917]],
          [[-1.11910486, -1.01226425, 0.68622279, 0.00536875],
           [ 2.33264184,  1.45991778, -0.71802276, -1.77120328]],
          [[ 0.52163047, -1.90511549, -0.56069887, 1.1521647 ],
           [-1.49890876,  0.20399603, -0.78263998, 1.30187178]]]

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
                                        source_padding, None, None, None, None,
                                        None, None)[0]

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

      h1 = et_decoder.FPropDefaultTheta(aux_vecs, aux_paddings, source_vecs,
                                        source_padding, None, None, None, None,
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
        h, _, prefix_states = et_decoder.ExtendStep(et_decoder.theta,
                                                    source_vecs[i, :, :],
                                                    prefix_states, aux_vecs,
                                                    aux_paddings)
        h2.append(h)

      h2 = tf.stack(h2)

      tf.global_variables_initializer().run()
      h1_v, h2_v = sess.run([h1, h2])
      self.assertAllClose(h1_v, h2_v)


def _AddClassesToTestParams(base_parameters_set, class_parameters_set):
  output_parameters = []
  for class_parameters in class_parameters_set:
    for base_parameters in base_parameters_set:
      testcase_name = (
          base_parameters['testcase_name'] + class_parameters['testcase_name'])
      new_parameters = base_parameters.copy()
      new_parameters.update(class_parameters)
      new_parameters['testcase_name'] = testcase_name
      output_parameters.append(new_parameters)
  return output_parameters


def _TransformerParamsWithEmbeddings(num_decoder_layers=0,
                                     num_encoder_layers=4,
                                     splits=1,
                                     num_micro_batches=1,
                                     has_softmax=False):
  model_dim = 4
  params = GPipeTransformerStack.Params()
  params.name = 'transformer'
  params.model_dim = model_dim
  params.num_decoder_layers = num_decoder_layers
  params.decoder_tpl.source_dim = model_dim
  params.decoder_tpl.tr_atten_tpl.num_attention_heads = 1
  params.decoder_tpl.tr_fflayer_tpl.hidden_dim = model_dim
  params.num_encoder_layers = num_encoder_layers
  params.encoder_tpl.source_dim = model_dim
  params.encoder_tpl.tr_atten_tpl.num_attention_heads = 1
  params.encoder_tpl.tr_fflayer_tpl.hidden_dim = model_dim
  params.num_micro_batches = num_micro_batches
  params.state_dtype = tf.float32
  if has_softmax:
    params.softmax_tpl.softmax.input_dim = model_dim
    params.softmax_tpl.softmax.num_classes = 2
  else:
    params.softmax_tpl = None

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


def _EvolvedTransformerParamsWithEmbeddings(num_decoder_layers=0,
                                            num_encoder_layers=4,
                                            splits=1,
                                            num_micro_batches=1,
                                            has_softmax=False):
  model_dim = 4
  params = GPipeEvolvedTransformerStack.Params()
  params.name = 'evolved_transformer'
  params.model_dim = model_dim
  params.num_decoder_layers = num_decoder_layers
  params.decoder_tpl.source_dim = model_dim
  params.decoder_tpl.tr_atten_tpl.num_attention_heads = 1
  params.decoder_tpl.tr_double_heads_atten_tpl.num_attention_heads = 1
  params.decoder_tpl.transformer_tpl.tr_atten_tpl.num_attention_heads = 1
  params.num_encoder_layers = num_encoder_layers
  params.encoder_tpl.source_dim = model_dim
  params.encoder_tpl.transformer_tpl.tr_atten_tpl.num_attention_heads = 1
  params.num_micro_batches = num_micro_batches
  params.state_dtype = tf.float32
  if has_softmax:
    params.softmax_tpl.softmax.input_dim = model_dim
    params.softmax_tpl.softmax.num_classes = 2
  else:
    params.softmax_tpl = None

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


def _TransformerRandomInputs(batch):
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


def _TransformerRandomInputsIds(batch):
  input_arr = np.array([[6] * batch, [4] * batch])
  paddings_arr = np.array([[0] * batch] * 2)
  tgt_input_arr = np.array([[3] * batch, [7] * batch, [9] * batch])
  tgt_paddings_arr = np.array([[0] * batch] * 3)
  inputs = tf.constant(input_arr.tolist(), dtype=tf.int32)
  paddings = tf.constant(paddings_arr.tolist(), dtype=tf.float32)
  tgt_inputs = tf.constant(tgt_input_arr.tolist(), dtype=tf.int32)
  tgt_paddings = tf.constant(tgt_paddings_arr.tolist(), dtype=tf.float32)
  return inputs, paddings, tgt_inputs, tgt_paddings


def _TransformerRandomInputsVecs(batch):
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


def _EvolvedTransformerRandomInputs(batch):
  padding_length = 10
  input_padding = [[[0, 0]] * batch] * padding_length
  input_arr = np.array([[[0, 1]] * batch] + input_padding + [[[1, -1]] * batch])
  padding_indexes = [[1] * batch] * padding_length
  paddings_arr = np.array([[0] * batch] + padding_indexes + [[0] * batch])
  tgt_input_arr = np.array([[[1, 2]] * batch] + input_padding +
                           [[[1, -1]] * batch] + input_padding +
                           [[[2, 1]] * batch])
  tgt_paddings_arr = np.array([[0] * batch] + padding_indexes + [[0] * batch] +
                              padding_indexes + [[0] * batch])
  inputs = tf.constant(input_arr.tolist(), dtype=tf.float32)
  paddings = tf.constant(paddings_arr.tolist(), dtype=tf.float32)
  tgt_inputs = tf.constant(tgt_input_arr.tolist(), dtype=tf.float32)
  tgt_paddings = tf.constant(tgt_paddings_arr.tolist(), dtype=tf.float32)
  return inputs, paddings, tgt_inputs, tgt_paddings


def _EvolvedTransformerRandomInputsIds(batch):
  padding_length = 10
  input_padding = [[0] * batch] * padding_length
  input_arr = np.array([[6] * batch] + input_padding + [[4] * batch])
  padding_indexes = [[1] * batch] * padding_length
  paddings_arr = np.array([[0] * batch] + padding_indexes + [[0] * batch])
  tgt_input_arr = np.array([[3] * batch] + input_padding + [[7] * batch] +
                           input_padding + [[9] * batch])
  tgt_paddings_arr = np.array([[0] * batch] + padding_indexes + [[0] * batch] +
                              padding_indexes + [[0] * batch])
  inputs = tf.constant(input_arr.tolist(), dtype=tf.int32)
  paddings = tf.constant(paddings_arr.tolist(), dtype=tf.float32)
  tgt_inputs = tf.constant(tgt_input_arr.tolist(), dtype=tf.int32)
  tgt_paddings = tf.constant(tgt_paddings_arr.tolist(), dtype=tf.float32)
  return inputs, paddings, tgt_inputs, tgt_paddings


class GPipeTransformerStackTest(test_utils.TestCase,
                                parameterized.TestCase):  # was tf.test.TestCase
  """Tests for GPipeTransformerStack layer."""

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
        params = _TransformerParamsWithEmbeddings(
            splits=splits, num_decoder_layers=2)
        params.dtype = tf.float32
        params.fprop_dtype = tf.float32
        packed_params = params.Copy()
        packed_params.packed_input = True
        xformer = GPipeTransformerStack(params)
        packed_xformer = GPipeTransformerStack(packed_params)
        # Prepare inputs
        inputs, paddings, tgt_inputs, tgt_paddings = _TransformerRandomInputsIds(
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
                               tgt_paddings)[2]
        packed_output = packed_xformer.FProp(
            packed_xformer.theta, packed_inputs, packed_paddings,
            packed_tgt_inputs, packed_tg_paddings, segment_ids, tgt_segment_id,
            None, None, segment_pos_id, tgt_segment_pos_id)[2]
        packed_output = tf.reshape(packed_output, output.shape)

        tf.global_variables_initializer().run()
        output, packed_output = sess.run([output, packed_output])
        self.assertAllClose(output, packed_output, rtol=1e-05, atol=1e-05)

  @parameterized.named_parameters(
      {
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
      params = _TransformerParamsWithEmbeddings(
          splits=splits,
          num_micro_batches=num_micro_batches,
          num_decoder_layers=3,
          num_encoder_layers=1)
      params.is_transparent = True
      params.transparent_merger_dropout_prob = 0.0
      xformer = GPipeTransformerStack(params)

      input_ids, id_paddings, tgt_inputs, tgt_paddings = _TransformerRandomInputsIds(
          batch=batch)
      inputs, paddings, _, _ = _TransformerRandomInputsVecs(batch=batch)
      tf.set_random_seed(1234)
      tf.global_variables_initializer().run()
      enc_outputs = xformer.EncoderFPropDefaultTheta(inputs, paddings)
      dec_output = xformer.FProp(xformer.theta, input_ids, id_paddings,
                                 tgt_inputs, tgt_paddings)[2]
      enc_out_1 = sess.run(enc_outputs)
      dec_out = sess.run(dec_output)
      self.assertAllClose(
          [[[0.68660116, 0.947429, 0.78953624, -1.20142817]] * batch,
           [[0.57919669, 1.12979364, 4.29336643, 0.45106331]] * batch],
          enc_out_1)
      self.assertAllClose(
          [[[-0.46651918, -1.62957835, 1.15657926, 1.08397353]] * batch,
           [[-0.34674695, -1.65999401, 1.08431196, 1.07384491]] * batch,
           [[-0.41073492, -1.60431314, 1.04607999, 1.08858371]] * batch],
          dec_out)

  # pylint: disable=bad-continuation
  # pyformat: disable
  @parameterized.named_parameters(
      _AddClassesToTestParams(({
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
      }), ({
          'testcase_name':
              '_transformer',
          'params_fn':
              _TransformerParamsWithEmbeddings,
          'stack_cls':
              GPipeTransformerStack,
          'inputs_fn':
              _TransformerRandomInputsIds,
          'expected_output':
              [[[-2.29650807, 0.25992393, 1.81951356, 1.52897644]] * 4,
               [[-2.14101386, 0.32607365, 1.73413348, 1.51806736]] * 4,
               [[-2.18863297, 0.34420109, 1.65913653, 1.58703828]] * 4]
      }, {
          'testcase_name':
              '_evolved_transformer',
          'params_fn':
              _EvolvedTransformerParamsWithEmbeddings,
          'stack_cls':
              GPipeEvolvedTransformerStack,
          'inputs_fn':
              _EvolvedTransformerRandomInputsIds,
          'expected_output': [
       [[-1.20986187e+00, -8.79107893e-01, 2.33660197e+00, 1.29931402e+00]] * 4,
       [[8.22729468e-02, -1.69113564e+00, 1.46728563e+00, 1.97036579e-01]] * 4,
       [[4.03284192e-01, -1.43809485e+00, 1.43909585e+00, -3.48087758e-01]] * 4,
       [[1.11850870e+00, 1.40267342e-01, -8.61150026e-02, -1.17340684e+00]] * 4,
       [[-8.50459397e-01, 1.13398373e-01, -8.99486303e-01, 1.55763149e+00]] * 4,
       [[-1.34836328e+00, -8.96196485e-01, 1.64263427e+00, 5.80114782e-01]] * 4,
       [[-6.36429131e-01, -1.42686498e+00, 1.74190414e+00, 3.46417874e-01]] * 4,
       [[-3.47153991e-02, -1.68883836e+00, 1.50680923e+00, 2.68793613e-01]] * 4,
       [[3.18001330e-01, -1.56888866e+00, 1.43814754e+00, -1.29013240e-01]] * 4,
       [[7.52930105e-01, -4.46001530e-01, 9.77058828e-01, -1.25989068e+00]] * 4,
       [[1.19872391e-04, 2.86186844e-01, -1.45033073e+00, 1.09993601e+00]] * 4,
       [[-1.00774717e+00, -1.06434298e+00, 2.23334217e+00, 1.35035193e+00]] * 4,
       [[-4.18500155e-01, -1.56903136e+00, 1.65510190e+00, 3.69135976e-01]] * 4,
       [[3.35794777e-01, -1.86454248e+00, 1.07441854e+00, 5.19350410e-01]] * 4,
       [[7.03283787e-01, -1.83285737e+00, 7.31202900e-01, 4.69641596e-01]] * 4,
       [[1.05139160e+00, -1.67215705e+00, 3.49240720e-01, 3.43249470e-01]] * 4,
       [[1.17681658e+00, -6.31463170e-01, -1.17845654e+00, 6.49677992e-01]] * 4,
       [[-1.67068458e+00, -5.03913283e-01, 1.19847691e+00, 9.21518922e-01]] * 4,
       [[-6.94836795e-01, -1.37394738e+00, 1.76720309e+00, 3.22906941e-01]] * 4,
       [[1.74539179e-01, -1.83659112e+00, 1.22484326e+00, 4.97922838e-01]] * 4,
       [[6.15836561e-01, -1.85283577e+00, 8.11907887e-01, 4.95319605e-01]] * 4,
       [[9.34679568e-01, -1.74228907e+00, 4.96106327e-01, 3.83897305e-01]] * 4,
       [[2.47265399e-01, -1.82084727e+00, 9.77287710e-01, 1.51184368e+00]] * 4
      ]})))
  # pyformat: enable
  # pylint: enable=bad-continuation
  def testGPipeTransformerDecoderStackFPropWithEmbeddings(
      self,
      params_fn,
      expected_output,
      inputs_fn,
      stack_cls,
      splits=1,
      num_micro_batches=1):
    batch = 4
    tf.flags.FLAGS.tpu_compatible = True
    with self.session() as sess:
      params = params_fn(
          num_decoder_layers=4,
          num_encoder_layers=0,
          splits=splits,
          num_micro_batches=num_micro_batches)
      params.dtype = tf.float32
      xformer = stack_cls(params)

      inputs, paddings, tgt_inputs, tgt_paddings = inputs_fn(batch)

      output = xformer.FProp(xformer.theta, inputs, paddings, tgt_inputs,
                             tgt_paddings)[2]

      tf.global_variables_initializer().run()
      output_val = sess.run(output)
      self.assertAllCloseAccordingToType(
          expected_output, output_val, rtol=1e-05, atol=1e-05)

  @parameterized.named_parameters(
      _AddClassesToTestParams(({
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
      }), ({
          'testcase_name': '_transformer',
          'params_fn': _TransformerParamsWithEmbeddings,
          'stack_cls': GPipeTransformerStack,
          'inputs_fn': _TransformerRandomInputsIds
      }, {
          'testcase_name': '_evolved_transformer',
          'params_fn': _EvolvedTransformerParamsWithEmbeddings,
          'stack_cls': GPipeEvolvedTransformerStack,
          'inputs_fn': _EvolvedTransformerRandomInputsIds
      })))
  def testGPipeTransformerLmModel(self,
                                  params_fn,
                                  stack_cls,
                                  inputs_fn,
                                  splits=1,
                                  num_micro_batches=1):
    batch = 4
    tf.flags.FLAGS.tpu_compatible = True
    with self.session() as sess:
      with tf.variable_scope('transformer_test', reuse=tf.AUTO_REUSE):
        params = params_fn(
            splits=splits,
            num_micro_batches=num_micro_batches,
            num_decoder_layers=0,
            has_softmax=True)
        params.state_dtype = tf.float32
      xformer = stack_cls(params)

      input_ids, id_paddings, _, _ = inputs_fn(batch=batch)
      labels = tf.ones([input_ids.shape.as_list()[0], batch])
      label_weights = tf.ones([input_ids.shape.as_list()[0], batch])
      tf.set_random_seed(1234)
      tf.global_variables_initializer().run()
      xent, logits = xformer.FProp(xformer.theta, input_ids, id_paddings, None,
                                   None, None, None, labels, label_weights)
      xent_out, logits_out = sess.run([xent, logits])
      print('xent_out={}'.format(xent_out))
      print('logits_out={}'.format(logits_out))

  @parameterized.named_parameters(
      _AddClassesToTestParams(({
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
      }), ({
          'testcase_name': '_transformer',
          'params_fn': _TransformerParamsWithEmbeddings,
          'stack_cls': GPipeTransformerStack,
          'inputs_fn': _TransformerRandomInputsIds
      }, {
          'testcase_name': '_evolved_transformer',
          'params_fn': _EvolvedTransformerParamsWithEmbeddings,
          'stack_cls': GPipeEvolvedTransformerStack,
          'inputs_fn': _EvolvedTransformerRandomInputsIds
      })))
  def testGPipeTransformerMtModel(self,
                                  params_fn,
                                  stack_cls,
                                  inputs_fn,
                                  splits=1,
                                  num_micro_batches=1):
    batch = 4
    tf.flags.FLAGS.tpu_compatible = True
    with self.session() as sess:
      with tf.variable_scope('transformer_test', reuse=tf.AUTO_REUSE):
        params = params_fn(
            splits=splits,
            num_micro_batches=num_micro_batches,
            num_decoder_layers=2,
            has_softmax=True)
        params.state_dtype = tf.float32
      xformer = stack_cls(params)

      input_ids, id_paddings, tgt_inputs, tgt_paddings = (
          inputs_fn(batch=batch))
      labels = tf.ones([tgt_inputs.shape.as_list()[0], batch])
      label_weights = tf.ones([tgt_inputs.shape.as_list()[0], batch])
      tf.set_random_seed(1234)
      tf.global_variables_initializer().run()
      xent, logits = xformer.FProp(xformer.theta, input_ids, id_paddings,
                                   tgt_inputs, tgt_paddings, None, None, labels,
                                   label_weights)
      xent_out, logits_out = sess.run([xent, logits])
      print('xent_out={}'.format(xent_out))
      print('logits_out={}'.format(logits_out))


if __name__ == '__main__':
  tf.test.main()
