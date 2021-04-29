# Lint as: python3
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

    input_task_arr = np.array([[0] * depth, [0] * depth])
    tgt_task_arr = np.array([[0] * depth] * 3)
    input_tasks = tf.constant(input_task_arr.tolist(), dtype=tf.int32)
    tgt_tasks = tf.constant(tgt_task_arr.tolist(), dtype=tf.int32)
    return (source_vecs, source_padding, aux_source_vecs, aux_source_paddings,
            input_tasks, tgt_tasks)

  def testGPipeSoftmaxLayerInputfromEncoder(self, use_task_ids=False):
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
      softmax.num_classes = 2
      softmax.input_dim = depth
      softmax = softmax.Instantiate()
      (source_vecs, _, _, _, _, _) = self._testInputs(depth=depth)
      source_padding = tf.zeros([5, 2])
      softmax_inputs = transformer.FPropDefaultTheta(source_vecs,
                                                     source_padding, None, None,
                                                     None, None, None, None)
      softmax_outputs = softmax.FPropDefaultTheta(*softmax_inputs)
      self.assertEqual([5, 2, 2], softmax_outputs.shape)

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
      softmax.num_classes = 2
      softmax.input_dim = depth
      softmax = softmax.Instantiate()
      (source_vecs, _, aux_vecs, aux_paddings, _,
       _) = self._testInputs(depth=depth)
      source_padding = tf.zeros([5, 2])
      softmax_inputs = transformer.FPropDefaultTheta(aux_vecs, aux_paddings,
                                                     source_vecs,
                                                     source_padding, None, None,
                                                     None, None)
      softmax_outputs = softmax.FPropDefaultTheta(*softmax_inputs)
      self.assertEqual([5, 2, 2], softmax_outputs.shape)

  def testTransformerLayerExtendStep(self):
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
      transformer = GPipeTransformerLayer(p)

      (source_vecs, _, aux_vecs, aux_paddings, input_tasks,
       tgt_tasks) = self._testInputs(depth=depth)
      source_padding = tf.zeros([5, 2])

      output1 = transformer.FPropDefaultTheta(aux_vecs, aux_paddings,
                                              source_vecs, source_padding, None,
                                              None, input_tasks, tgt_tasks)
      h1 = output1[2]
      out_src_task, out_tgt_task = output1[-2], output1[-1]

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

      self.evaluate(tf.global_variables_initializer())
      h1_v, h2_v = self.evaluate([h1, h2])
      self.assertAllClose(h1_v, h2_v)
      self.assertAllClose(out_src_task, input_tasks)
      self.assertAllClose(out_tgt_task, tgt_tasks)
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
    with self.session(use_gpu=True):
      np.random.seed(6348575)
      depth = 4
      p = GPipeEvolvedTransformerEncoderLayer.Params()
      p.name = 'gpipe_evolved_transformer_encoder'
      p.source_dim = depth
      p.transformer_tpl.tr_fflayer_tpl.hidden_dim = 7
      p.transformer_tpl.tr_atten_tpl.num_attention_heads = 2
      transformer = GPipeEvolvedTransformerEncoderLayer(p)

      (source_vecs, source_padding, _, _, _, _) = self._testInputs(depth=depth)

      output = transformer.FPropDefaultTheta(source_vecs, source_padding, None,
                                             None, None, None, None, None, None,
                                             None)
      h = output[0]

      self.evaluate(tf.global_variables_initializer())
      actual_layer_output = self.evaluate([h])[0]
      tf.logging.info(np.array_repr(actual_layer_output))
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_layer_output = [
          [[-2.059799  , -1.2909635 ,  0.25885093,  0.14688382],
           [-0.8974035 , -1.5618052 ,  0.4086342 ,  0.8884169 ]],
          [[-1.8822882 , -0.0494532 , -0.54840994, -0.27622807],
           [-0.9241364 , -0.16045655, -0.32357514,  1.485477  ]],
          [[ 0.8964625 , -1.1524196 , -1.0661578 , -0.22737658],
           [-1.6575948 , -1.1755587 ,  0.93549323,  0.8066237 ]],
          [[-1.1761202 , -1.3182621 ,  0.79037327, -0.03893514],
           [ 2.1135292 ,  1.3683249 , -0.24891634, -2.525507  ]],
          [[ 0.24391544, -1.6017915 , -0.7599152 ,  1.1556216 ],
           [-1.6751958 ,  0.04410481, -0.8073819 ,  1.6209301 ]]]

      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(
          expected_layer_output, actual_layer_output, rtol=1e-05, atol=1e-05)

  def testEvolvedTransformerDecoderLayerConstruction(self):
    p = GPipeEvolvedTransformerDecoderLayer.Params()
    p.name = 'gpipe_evolved_transformer_decoder'
    p.source_dim = 16
    p.transformer_tpl.tr_atten_tpl.num_attention_heads = 2
    p.has_aux_atten = True
    p.mask_self_atten = True
    _ = GPipeEvolvedTransformerDecoderLayer(p)

  def testEvolvedTransformerDecoderLayerFProp(self):
    with self.session(use_gpu=True):
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

      (source_vecs, source_padding, aux_vecs, aux_paddings, _,
       _) = self._testInputs(depth=depth)

      output = transformer.FPropDefaultTheta(aux_vecs, aux_paddings,
                                             source_vecs, source_padding, None,
                                             None, None, None, None, None)
      h = output[0]

      self.evaluate(tf.global_variables_initializer())
      actual_layer_output = self.evaluate([h])[0]
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
    with self.session(use_gpu=True):
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

      (source_vecs, _, aux_vecs, aux_paddings, input_tasks,
       tgt_tasks) = self._testInputs(depth=depth)
      source_padding = tf.zeros([5, 2])

      output1 = et_decoder.FPropDefaultTheta(aux_vecs, aux_paddings,
                                             source_vecs, source_padding, None,
                                             None, input_tasks, tgt_tasks)
      h1 = output1[2]
      out_src_task, out_tgt_task = output1[-2], output1[-1]
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

      self.evaluate(tf.global_variables_initializer())
      h1_v, h2_v = self.evaluate([h1, h2])
      self.assertAllClose(h1_v, h2_v)
      self.assertAllClose(out_src_task, input_tasks)
      self.assertAllClose(out_tgt_task, tgt_tasks)


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
                                     has_softmax=False,
                                     use_task_ids=False):
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
    params.softmax_tpl.input_dim = model_dim
    params.softmax_tpl.num_classes = 2
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

  # Task embeddings.
  if use_task_ids:
    emb_params.enc_task_emb = emb_params.token_emb.Copy()
    emb_params.dec_task_emb = emb_params.token_emb.Copy()
  params.splits = splits
  params.random_seed = 0
  return params


def _EvolvedTransformerParamsWithEmbeddings(num_decoder_layers=0,
                                            num_encoder_layers=4,
                                            splits=1,
                                            num_micro_batches=1,
                                            has_softmax=False,
                                            use_task_ids=False):
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
    params.softmax_tpl.input_dim = model_dim
    params.softmax_tpl.num_classes = 2
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

  # Task embeddings.
  if use_task_ids:
    emb_params.enc_task_emb = emb_params.token_emb.Copy()
    emb_params.dec_task_emb = emb_params.token_emb.Copy()
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
  input_task_arr = np.array([[0] * batch, [0] * batch])
  tgt_input_arr = np.array([[3] * batch, [7] * batch, [9] * batch])
  tgt_paddings_arr = np.array([[0] * batch] * 3)
  tgt_task_arr = np.array([[0] * batch] * 3)
  inputs = tf.constant(input_arr.tolist(), dtype=tf.int32)
  paddings = tf.constant(paddings_arr.tolist(), dtype=tf.float32)
  input_tasks = tf.constant(input_task_arr.tolist(), dtype=tf.int32)
  tgt_inputs = tf.constant(tgt_input_arr.tolist(), dtype=tf.int32)
  tgt_paddings = tf.constant(tgt_paddings_arr.tolist(), dtype=tf.float32)
  tgt_tasks = tf.constant(tgt_task_arr.tolist(), dtype=tf.int32)
  return inputs, paddings, tgt_inputs, tgt_paddings, input_tasks, tgt_tasks


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
  input_task_arr = np.array([[0] * batch] + input_padding + [[0] * batch])
  padding_indexes = [[1] * batch] * padding_length
  paddings_arr = np.array([[0] * batch] + padding_indexes + [[0] * batch])
  tgt_input_arr = np.array([[3] * batch] + input_padding + [[7] * batch] +
                           input_padding + [[9] * batch])
  tgt_task_arr = np.array([[0] * batch] + input_padding + [[0] * batch] +
                          input_padding + [[0] * batch])
  tgt_paddings_arr = np.array([[0] * batch] + padding_indexes + [[0] * batch] +
                              padding_indexes + [[0] * batch])
  inputs = tf.constant(input_arr.tolist(), dtype=tf.int32)
  paddings = tf.constant(paddings_arr.tolist(), dtype=tf.float32)
  input_tasks = tf.constant(input_task_arr.tolist(), dtype=tf.int32)
  tgt_inputs = tf.constant(tgt_input_arr.tolist(), dtype=tf.int32)
  tgt_paddings = tf.constant(tgt_paddings_arr.tolist(), dtype=tf.float32)
  tgt_tasks = tf.constant(tgt_task_arr.tolist(), dtype=tf.int32)
  return inputs, paddings, tgt_inputs, tgt_paddings, input_tasks, tgt_tasks


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
  def testGPipeTransformerBatchMajorConstruction(self, splits=1):
    batch = 4
    tf.flags.FLAGS.tpu_compatible = True
    with self.session():
      with tf.variable_scope('transformer_test', reuse=tf.AUTO_REUSE):
        params = _TransformerParamsWithEmbeddings(
            splits=splits, num_decoder_layers=4, has_softmax=True)
        params.batch_dim = 0
        params.emb_tpl.batch_dim = 0
        xformer = params.Instantiate()
        input_ids, id_paddings, tgt_inputs, tgt_paddings, _, _ = (
            _TransformerRandomInputsIds(batch=batch))
        input_ids = tf.transpose(input_ids)
        id_paddings = tf.transpose(id_paddings)
        tgt_inputs = tf.transpose(tgt_inputs)
        tgt_paddings = tf.transpose(tgt_paddings)
        labels = tf.ones([batch, tgt_inputs.shape.as_list()[1]], dtype=tf.int32)
        label_weights = tf.ones([batch, tgt_inputs.shape.as_list()[1]])
        tf.random.set_seed(1234)
        self.evaluate(tf.global_variables_initializer())
        xent, logits = xformer.FProp(xformer.theta, input_ids, id_paddings,
                                     tgt_inputs, tgt_paddings, None, None,
                                     labels, label_weights, None, None, None,
                                     None)
        xent_out, logits_out = self.evaluate([xent, logits])
        print('xent_out={}'.format(xent_out))
        print('logits_out={}'.format(logits_out))

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
    with self.session():
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
        inputs, paddings, tgt_inputs, tgt_paddings, _, _ = _TransformerRandomInputsIds(
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
        segment_segment_pos = tf.transpose(
            tf.constant([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=tf.int32))
        tgt_segment_segment_pos = tf.transpose(
            tf.constant([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]], dtype=tf.int32))

        output = xformer.FProp(xformer.theta, inputs, paddings, tgt_inputs,
                               tgt_paddings)[2]
        packed_output = packed_xformer.FProp(
            packed_xformer.theta, packed_inputs, packed_paddings,
            packed_tgt_inputs, packed_tg_paddings, segment_ids, tgt_segment_id,
            None, None, segment_segment_pos, tgt_segment_segment_pos)[2]
        packed_output = tf.reshape(packed_output, output.shape)

        self.evaluate(tf.global_variables_initializer())
        output, packed_output = self.evaluate([output, packed_output])
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
    with self.session():
      params = _TransformerParamsWithEmbeddings(
          splits=splits,
          num_micro_batches=num_micro_batches,
          num_decoder_layers=3,
          num_encoder_layers=1)
      params.is_transparent = True
      params.transparent_merger_dropout_prob = 0.0
      xformer = GPipeTransformerStack(params)

      input_ids, id_paddings, tgt_inputs, tgt_paddings, _, _ = _TransformerRandomInputsIds(
          batch=batch)
      inputs, paddings, _, _ = _TransformerRandomInputsVecs(batch=batch)
      tf.random.set_seed(1234)
      self.evaluate(tf.global_variables_initializer())
      enc_outputs = xformer.EncoderFPropDefaultTheta(inputs, paddings)
      dec_output = xformer.FProp(xformer.theta, input_ids, id_paddings,
                                 tgt_inputs, tgt_paddings)[2]
      enc_out_1 = self.evaluate(enc_outputs)
      dec_out = self.evaluate(dec_output)
      self.assertAllClose(
          [[[0.017581, 0.802863, 0.975554, -1.164572]] * batch,
           [[-0.549953, 1.196884, 4.910457, -0.102137]] * batch], enc_out_1)
      self.assertAllClose(
          [[[-1.122128, 1.111972, 4.642949, -2.14831]] * batch,
           [[-1.336919, 1.182709, 4.785938, -2.039246]] * batch,
           [[-1.335168, 1.297679, 4.720459, -2.111006]] * batch], dec_out)

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
              [[[-2.306972, 2.07353, 5.542715, -2.421023]] * 4,
               [[-2.366269, 2.0730767, 5.573641, -2.371158]] * 4,
               [[-2.379758, 2.092285, 5.570339, -2.370932]] * 4]
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
       [[-0.563482, 0.873324, 4.807824, -2.21867]] * 4,
       [[-0.69335, 0.275365, 3.85347, -1.25291]] * 4,
       [[-0.691535, 0.513666, 3.818152, -1.448952]] * 4,
       [[-0.906161, 0.207199, 3.865673, -1.015405]] * 4,
       [[-0.779552, 0.199017, 3.864474, -1.115817]] * 4,
       [[-0.733586, 0.229975, 3.860421, -1.181348]] * 4,
       [[-0.712061, 0.24913, 3.857376, -1.215363]] * 4,
       [[-0.697319, 0.267139, 3.854558, -1.242636]] * 4,
       [[-0.667346, 0.455872, 3.825749, -1.421671]] * 4,
       [[-0.933211, 0.304563, 3.863014, -1.08345]] * 4,
       [[-0.804276, 0.187855, 3.865573, -1.084767]] * 4,
       [[-0.481617, 0.657398, 4.772114, -2.071473]] * 4,
       [[-0.658759, 0.130042, 3.853722, -1.142575]] * 4,
       [[-0.641755, 0.148885, 3.851666, -1.173375]] * 4,
       [[-0.623731, 0.205668, 3.847772, -1.23985]] * 4,
       [[-0.827914, 0.091931, 3.861872, -0.967819]] * 4,
       [[-0.74599, 0.066758, 3.858107, -1.010512]] * 4,
       [[-0.69355, 0.099376, 3.856509, -1.085727]] * 4,
       [[-0.664674, 0.124304, 3.854321, -1.132531]] * 4,
       [[-0.646149, 0.143505, 3.852249, -1.164976]] * 4,
       [[-0.630899, 0.169327, 3.849945, -1.200752]] * 4,
       [[-0.83285, 0.199918, 3.866308, -1.072462]] * 4,
       [[-0.440338, 0.562651, 4.749378, -2.003749]] * 4
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
    with self.session():
      params = params_fn(
          num_decoder_layers=4,
          num_encoder_layers=0,
          splits=splits,
          num_micro_batches=num_micro_batches)
      params.dtype = tf.float32
      xformer = stack_cls(params)

      inputs, paddings, tgt_inputs, tgt_paddings, _, _ = inputs_fn(batch)

      output = xformer.FProp(xformer.theta, inputs, paddings, tgt_inputs,
                             tgt_paddings)[2]

      self.evaluate(tf.global_variables_initializer())
      output_val = self.evaluate(output)
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
    with self.session():
      with tf.variable_scope('transformer_test', reuse=tf.AUTO_REUSE):
        params = params_fn(
            splits=splits,
            num_micro_batches=num_micro_batches,
            num_decoder_layers=0,
            has_softmax=True)
        params.state_dtype = tf.float32
      xformer = stack_cls(params)

      input_ids, id_paddings, _, _, _, _ = inputs_fn(batch=batch)
      labels = tf.ones([input_ids.shape.as_list()[0], batch], dtype=tf.int32)
      label_weights = tf.ones([input_ids.shape.as_list()[0], batch])
      tf.random.set_seed(1234)
      self.evaluate(tf.global_variables_initializer())
      xent, logits = xformer.FProp(xformer.theta, input_ids, id_paddings, None,
                                   None, None, None, labels, label_weights)
      xent_out, logits_out = self.evaluate([xent, logits])
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
      }, {
          'testcase_name': '_split1_task_embs',
          'use_task_embs': True
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
                                  num_micro_batches=1,
                                  use_task_embs=False):
    batch = 4
    tf.flags.FLAGS.tpu_compatible = True
    with self.session():
      with tf.variable_scope('transformer_test', reuse=tf.AUTO_REUSE):
        params = params_fn(
            splits=splits,
            num_micro_batches=num_micro_batches,
            num_decoder_layers=2,
            has_softmax=True,
            use_task_ids=use_task_embs)
        params.state_dtype = tf.float32
      xformer = stack_cls(params)

      input_ids, id_paddings, tgt_inputs, tgt_paddings, input_task_ids, tgt_task_ids = (
          inputs_fn(batch=batch))
      labels = tf.ones([tgt_inputs.shape.as_list()[0], batch], dtype=tf.int32)
      label_weights = tf.ones([tgt_inputs.shape.as_list()[0], batch])
      tf.random.set_seed(1234)
      self.evaluate(tf.global_variables_initializer())
      xent, logits = xformer.FProp(xformer.theta, input_ids, id_paddings,
                                   tgt_inputs, tgt_paddings, None, None, labels,
                                   label_weights, None, None, input_task_ids,
                                   tgt_task_ids)
      xent_out, logits_out = self.evaluate([xent, logits])
      print('xent_out={}'.format(xent_out))
      print('logits_out={}'.format(logits_out))


def _BatchMajorTransformerParams(splits=1,
                                 num_micro_batches=1,
                                 packed_input=True):
  model_dim = 4
  params = layers_with_gpipe.GPipeBatchMajorTransformerStack.Params()
  params.name = 'transformer'
  params.model_dim = model_dim
  params.packed_input = packed_input
  params.num_decoder_layers = 4
  params.decoder_tpl.input_dim = model_dim
  params.decoder_tpl.tr_atten_tpl.num_heads = 1
  params.decoder_tpl.tr_fflayer_tpl.hidden_dim = model_dim
  params.decoder_tpl.mask_self_atten = True
  params.decoder_tpl.has_aux_atten = True
  params.num_encoder_layers = 2
  params.encoder_tpl.input_dim = model_dim
  params.encoder_tpl.tr_atten_tpl.num_heads = 1
  params.encoder_tpl.tr_fflayer_tpl.hidden_dim = model_dim
  params.state_dtype = tf.float32
  params.softmax_tpl.input_dim = model_dim
  params.softmax_tpl.num_classes = 2

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
  params.num_micro_batches = num_micro_batches
  params.random_seed = 0
  return params


def _BatchMajorTransformerRandomInputs(batch):
  input_arr = np.array([[6, 4]] * batch)
  paddings_arr = np.array([[0] * 2] * batch)
  input_seg_arr = np.array([[0, 1]] * batch)
  input_pos_arr = np.array([[0, 0]] * batch)
  tgt_input_arr = np.array([[3, 7, 9]] * batch)
  tgt_paddings_arr = np.array([[0] * 3] * batch)
  tgt_seg_arr = np.array([[0, 1, 1]] * batch)
  tgt_pos_arr = np.array([[0, 0, 1]] * batch)
  inputs = tf.constant(input_arr.tolist(), dtype=tf.int32)
  paddings = tf.constant(paddings_arr.tolist(), dtype=tf.float32)
  input_seg = tf.constant(input_seg_arr.tolist(), dtype=tf.int32)
  input_pos = tf.constant(input_pos_arr.tolist(), dtype=tf.int32)
  tgt_inputs = tf.constant(tgt_input_arr.tolist(), dtype=tf.int32)
  tgt_paddings = tf.constant(tgt_paddings_arr.tolist(), dtype=tf.float32)
  tgt_seg = tf.constant(tgt_seg_arr.tolist(), dtype=tf.int32)
  tgt_pos = tf.constant(tgt_pos_arr.tolist(), dtype=tf.int32)
  return (inputs, paddings, tgt_inputs, tgt_paddings, input_seg, tgt_seg,
          input_pos, tgt_pos)


class GPipeBatchMajorTransformerStackTest(test_utils.TestCase,
                                          parameterized.TestCase
                                         ):  # was tf.test.TestCase
  """Tests for GPipeBatchMajorTransformerStack layer."""

  @parameterized.named_parameters(
      {
          'testcase_name': '_one_split_one_mb_packed',
          'splits': 1,
          'num_micro_batches': 1,
          'packed_input': True
      }, {
          'testcase_name': '_two_splits_one_mb_packed',
          'splits': 2,
          'num_micro_batches': 1,
          'packed_input': True
      }, {
          'testcase_name': '_two_splits_two_mb_packed',
          'splits': 2,
          'num_micro_batches': 2,
          'packed_input': True
      }, {
          'testcase_name': '_one_split_two_mb_packed',
          'splits': 1,
          'num_micro_batches': 2,
          'packed_input': True
      }, {
          'testcase_name': '_one_split_one_mb_unpacked',
          'splits': 1,
          'num_micro_batches': 1,
          'packed_input': False
      }, {
          'testcase_name': '_two_splits_one_mb_unpacked',
          'splits': 2,
          'num_micro_batches': 1,
          'packed_input': False
      }, {
          'testcase_name': '_two_splits_two_mb_unpacked',
          'splits': 2,
          'num_micro_batches': 2,
          'packed_input': False
      }, {
          'testcase_name': '_one_split_two_mb_unpacked',
          'splits': 1,
          'num_micro_batches': 2,
          'packed_input': False
      })
  def testGPipeBatchMajorTransformerFProp(self,
                                          splits=1,
                                          num_micro_batches=1,
                                          packed_input=True):
    batch = 4
    tf.flags.FLAGS.tpu_compatible = True
    with self.session():
      with tf.variable_scope('transformer_test', reuse=tf.AUTO_REUSE):
        params = _BatchMajorTransformerParams(splits, num_micro_batches,
                                              packed_input)
        xformer = params.Instantiate()
        (input_ids, id_paddings, tgt_inputs, tgt_paddings, input_seg, tgt_seg,
         input_pos, tgt_pos) = (
             _BatchMajorTransformerRandomInputs(batch=batch))
        labels = tf.ones([batch, tgt_inputs.shape.as_list()[1]], dtype=tf.int32)
        label_weights = tf.ones([batch, tgt_inputs.shape.as_list()[1]])
        tf.random.set_seed(1234)
        self.evaluate(tf.global_variables_initializer())
        xent, logits = xformer.FProp(xformer.theta, input_ids, id_paddings,
                                     tgt_inputs, tgt_paddings, input_seg,
                                     tgt_seg, labels, label_weights, input_pos,
                                     tgt_pos)
        xent_out, logits_out = self.evaluate([xent, logits])
        print('xent_out={}'.format(xent_out))
        print('logits_out={}'.format(logits_out))


if __name__ == '__main__':
  tf.test.main()
