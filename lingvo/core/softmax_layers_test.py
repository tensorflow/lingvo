# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Test suite covering lingvo softmax layers.

Separated out from the monolithic layers_test.py to reduce flakiness likely
caused by parallel tests setting competing numpy seed values.
"""
import contextlib

from absl.testing import parameterized
import lingvo.compat as tf
from lingvo.core import layers
from lingvo.core import layers_test
from lingvo.core import py_utils
from lingvo.core import quant_utils
from lingvo.core import symbolic
from lingvo.core import test_utils
import numpy as np


class SoftmaxLayerTest(test_utils.TestCase):

  def _RunSimpleFullSoftmax(
      self,
      num_shards=1,
      chunk_size=0,
      inputs=None,
      class_ids=None,
      class_weights=None,
      class_probabilities=None,
      num_samples=0,
      default_qdomain=None,
      logits_qdomain=None,
      training_step=-1,
      seed=None,
      dtype=tf.float32,
      fprop_dtype=None,
      apply_pruning=False,
      use_bias=True,
  ):
    if fprop_dtype is None:
      fprop_dtype = dtype
    with contextlib.ExitStack() as context_stack:
      g = layers_test._ResetTfStatus(self, context_stack)
    with self.session(use_gpu=True, graph=g):
      if seed is not None:
        tf.random.set_seed(seed)
      if class_ids is None:
        class_ids = tf.constant([[1], [5], [10]], dtype=tf.int32)
      else:
        class_ids = tf.constant(class_ids)
      if class_weights is None:
        class_weights = tf.constant([1.0, 0.4, 0.8], dtype=fprop_dtype)
      else:
        class_weights = tf.constant(class_weights)
      np.random.seed(12345)
      if inputs is None:
        inputs = [tf.constant(np.random.rand(3, 10), dtype=fprop_dtype)]
      else:
        inputs = [tf.constant(inputs, dtype=fprop_dtype)]

      params = layers.SimpleFullSoftmax.Params()
      params.dtype = dtype
      params.fprop_dtype = fprop_dtype
      params.name = 'softmax'
      params.input_dim = 10
      params.num_classes = 32
      params.num_shards = num_shards
      params.chunk_size = chunk_size
      params.apply_pruning = apply_pruning
      params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)
      params.random_seed = 12345678
      params.use_bias = use_bias

      if default_qdomain is not None:
        params.qdomain.default = default_qdomain
      if logits_qdomain is not None:
        params.qdomain.logits = logits_qdomain

      if num_samples > 0:
        # Turn on sampled soft-max; the asserts need to hold for it to be used.
        params.num_sampled = num_samples
        assert class_probabilities is None
        assert chunk_size == 0

      params.vn.global_vn = False
      softmax = layers.SimpleFullSoftmax(params)
      self.evaluate(tf.global_variables_initializer())
      if training_step >= 0:
        self.evaluate(
            tf.assign(py_utils.GetOrCreateGlobalStepVar(), training_step)
        )
      xent_loss = softmax.FProp(
          softmax.theta,
          inputs,
          class_weights=class_weights,
          class_ids=class_ids,
          class_probabilities=class_probabilities,
      )

      all_vars = tf.get_collection('SimpleFullSoftmax_vars')
      expected_var_names = []
      for i in range(num_shards):
        expected_var_names.append('softmax/weight_%d/var:0' % i)
        if use_bias:
          expected_var_names.append('softmax/bias_%d/var:0' % i)

      all_var_names = [v.name for v in all_vars]
      self.assertCountEqual(expected_var_names, all_var_names)

      return self.evaluate(xent_loss)

  def testSimpleFullSoftmaxMasked(self):
    num_shards = 2
    apply_pruning = True
    params = layers.SimpleFullSoftmax.Params()
    params.name = 'softmax'
    params.dtype = tf.float32
    params.input_dim = 10
    params.num_classes = 32
    params.fprop_dtype = tf.float32
    params.num_shards = num_shards
    params.apply_pruning = apply_pruning
    params.random_seed = 12345678
    softmax_layer = layers.SimpleFullSoftmax(params)

    self.assertIn('weight_0', softmax_layer.vars.weight_0.name)
    self.assertIn('weight_1', softmax_layer.vars.weight_1.name)
    self.assertIn('mask_0', softmax_layer.vars.mask_0.name)
    self.assertIn('mask_1', softmax_layer.vars.mask_1.name)
    self.assertIn('threshold_0', softmax_layer.vars.threshold_0.name)
    self.assertIn('threshold_1', softmax_layer.vars.threshold_1.name)

    self.assertEqual(
        softmax_layer.theta.weight_0.get_shape(), tf.TensorShape([10, 16])
    )
    self.assertEqual(
        softmax_layer.theta.weight_1.get_shape(), tf.TensorShape([10, 16])
    )
    self.assertEqual(
        softmax_layer.theta.mask_0.get_shape(), tf.TensorShape([10, 16])
    )
    self.assertEqual(
        softmax_layer.theta.mask_1.get_shape(), tf.TensorShape([10, 16])
    )
    self.assertEqual(
        softmax_layer.theta.threshold_0.get_shape(), tf.TensorShape([])
    )
    self.assertEqual(
        softmax_layer.theta.threshold_0.get_shape(), tf.TensorShape([])
    )

    softmax_var_count = 4  # 2 each for weights and biases (we have 2 shards)
    wts = tf.get_collection('SimpleFullSoftmax_vars')
    self.assertEqual(softmax_var_count, len(wts))

    softmax_mask_count = 2
    masks = tf.get_collection('masks')
    self.assertEqual(softmax_mask_count, len(masks))

    softmax_threshold_count = 2
    threshold = tf.get_collection('thresholds')
    self.assertEqual(softmax_threshold_count, len(threshold))

    # Sampled and Masked
    xent_loss = self._RunSimpleFullSoftmax(
        num_samples=32, seed=12345, apply_pruning=True
    )
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    self.assertNear(loss, 8.681571, 1e-5)
    self.assertNear(log_perplexity, 3.946169, 1e-5)

    # Sharded and Masked
    xent_loss = self._RunSimpleFullSoftmax(num_shards=2, apply_pruning=True)
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    self.assertNear(loss, 6.14888, 1e-5)
    self.assertNear(log_perplexity, 2.79495, 1e-5)

    # Non_2D and Masked
    xent_loss = self._RunSimpleFullSoftmax(
        inputs=np.random.rand(4, 3, 10),
        class_weights=np.ones((4, 3)),
        class_ids=np.random.randint(32, size=(4, 3)),
        apply_pruning=True,
    )
    self.assertEqual(xent_loss.logits.shape, (4, 3, 32))
    self.assertEqual(xent_loss.per_example_xent.shape, (4, 3))
    self.assertEqual(xent_loss.per_example_weight.shape, (4, 3))

    xent_loss = self._RunSimpleFullSoftmax(
        inputs=np.random.rand(4, 3, 10),
        class_weights=np.ones((4, 3)),
        class_probabilities=np.random.uniform(size=(4, 3, 32)),
        apply_pruning=True,
    )
    self.assertEqual(xent_loss.logits.shape, (4, 3, 32))
    self.assertEqual(xent_loss.per_example_xent.shape, (4, 3))
    self.assertEqual(xent_loss.per_example_weight.shape, (4, 3))

    # Chunked and Masked
    for chunk_size in (0, 1, 2, 3, 4, 5):
      print('chunk_size = ', chunk_size)
      xent_output = self._RunSimpleFullSoftmax(
          chunk_size=chunk_size, apply_pruning=True
      )
      loss = xent_output.total_xent
      log_perplexity = xent_output.avg_xent
      print('xent_output ', xent_output)
      print(
          'xent_output.per_example_argmax.dtype ',
          xent_output.per_example_argmax.dtype,
      )
      self.assertAllClose(loss, 6.22425)
      self.assertAllClose(log_perplexity, 2.82920)
      self.assertAllEqual(
          xent_output.per_example_argmax, np.argmax(xent_output.logits, axis=1)
      )

  def testSimpleFullSoftmax_Sampled(self):
    xent_loss = self._RunSimpleFullSoftmax(num_samples=32, seed=12345)
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    self.assertNear(loss, 8.681571, 1e-5)
    self.assertNear(log_perplexity, 3.946169, 1e-5)

  def testSimpleFullSoftmax_NoBias(self):
    xent_loss = self._RunSimpleFullSoftmax(seed=12345, use_bias=False)
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    err = 1e-5
    self.assertNear(loss, 12.476410, err=err)
    self.assertNear(log_perplexity, 5.671095, err=err)

  def testSimpleFullSoftmax_SampledAndSharded(self):
    xent_loss = self._RunSimpleFullSoftmax(
        num_shards=4, num_samples=32, seed=12345
    )
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    self.assertNear(loss, 8.510439, 1e-5)
    self.assertNear(log_perplexity, 3.868381, 1e-5)

  def testSimpleFullSoftmax_Non2D(self):
    xent_loss = self._RunSimpleFullSoftmax(
        inputs=np.random.rand(4, 3, 10),
        class_weights=np.ones((4, 3)),
        class_ids=np.random.randint(32, size=(4, 3)),
    )
    self.assertEqual(xent_loss.logits.shape, (4, 3, 32))
    self.assertEqual(xent_loss.per_example_xent.shape, (4, 3))
    self.assertEqual(xent_loss.per_example_weight.shape, (4, 3))

    xent_loss = self._RunSimpleFullSoftmax(
        inputs=np.random.rand(4, 3, 10),
        class_weights=np.ones((4, 3)),
        class_probabilities=np.random.uniform(size=(4, 3, 32)),
    )
    self.assertEqual(xent_loss.logits.shape, (4, 3, 32))
    self.assertEqual(xent_loss.per_example_xent.shape, (4, 3))
    self.assertEqual(xent_loss.per_example_weight.shape, (4, 3))

  def _testSimpleFullSoftmax_Basic_Helper(self, dtype, fprop_dtype):
    xent_loss = self._RunSimpleFullSoftmax(dtype=dtype, fprop_dtype=fprop_dtype)
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    print(['loss', loss])
    print(['log_perplexity', log_perplexity])
    err = 1e-5
    if fprop_dtype == tf.float16 or fprop_dtype == tf.bfloat16:
      err = 1e-2
    self.assertNear(loss, 6.22425, err=err)
    self.assertNear(log_perplexity, 2.8292, err=err)
    self.assertAllEqual(
        xent_loss.per_example_argmax, np.argmax(xent_loss.logits, axis=1)
    )

  def testSimpleFullSoftmax_Basic_Float32(self):
    self._testSimpleFullSoftmax_Basic_Helper(
        dtype=tf.float32, fprop_dtype=tf.float32
    )

  def testSimpleFullSoftmax_Basic_Float32Float16(self):
    self._testSimpleFullSoftmax_Basic_Helper(
        dtype=tf.float32, fprop_dtype=tf.float16
    )

  def testSimpleFullSoftmax_Sharded(self):
    xent_loss = self._RunSimpleFullSoftmax(2)
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    print(['loss', loss])
    print(['log_perplexity', log_perplexity])
    self.assertNear(loss, 6.14888, 1e-5)
    self.assertNear(log_perplexity, 2.79495, 1e-5)

  def testSimpleFullSoftmax_Chunked(self):
    for chunk_size in (0, 1, 2, 3, 4, 5):
      print('chunk_size = ', chunk_size)
      xent_output = self._RunSimpleFullSoftmax(chunk_size=chunk_size)
      loss = xent_output.total_xent
      log_perplexity = xent_output.avg_xent
      print('xent_output ', xent_output)
      print(
          'xent_output.per_example_argmax.dtype ',
          xent_output.per_example_argmax.dtype,
      )
      self.assertAllClose(loss, 6.22425)
      self.assertAllClose(log_perplexity, 2.82920)
      self.assertAllEqual(
          xent_output.per_example_argmax, np.argmax(xent_output.logits, axis=1)
      )

  def testSimpleFullSoftmax_Basic_Distributions(self):
    with self.session(use_gpu=False):
      class_ids = tf.constant([1, 5, 10], dtype=tf.int32)
      class_weights = tf.constant([1.0, 0.4, 0.8], dtype=tf.float32)
      np.random.seed(12345)
      inputs = [tf.constant(np.random.rand(3, 10), dtype=tf.float32)]

      params = layers.SimpleFullSoftmax.Params()
      params.name = 'softmax'
      params.input_dim = 10
      params.num_classes = 32
      params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)
      params.vn.global_vn = False
      softmax = layers.SimpleFullSoftmax(params)
      xent_loss = softmax.XentLoss(
          inputs,
          class_weights=class_weights,
          class_probabilities=tf.one_hot(class_ids, params.num_classes),
      )
      self.evaluate(tf.global_variables_initializer())
      loss = self.evaluate(xent_loss.total_xent)
      log_perplexity = self.evaluate(xent_loss.avg_xent)
      print(['loss', loss])
      print(['log_perplexity', log_perplexity])
      self.assertNear(loss, 6.22425, 1e-5)
      self.assertNear(log_perplexity, 2.8292, 1e-5)

  def testSimpleFullSoftmax_GlobalVN(self):
    with self.session(use_gpu=False):
      class_ids = tf.constant([1, 5, 10], dtype=tf.int32)
      class_weights = tf.constant([1.0, 0.4, 0.8], dtype=tf.float32)
      np.random.seed(12345)
      inputs = [tf.constant(np.random.rand(3, 10), dtype=tf.float32)]

      params = layers.SimpleFullSoftmax.Params()
      params.name = 'softmax'
      params.input_dim = 10
      params.num_classes = 32
      params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)
      params.vn.global_vn = True
      params.vn.seed = 23456
      params.vn.scale = 1.0
      softmax = layers.SimpleFullSoftmax(params)
      xent_loss = softmax.XentLoss(
          inputs, class_weights=class_weights, class_ids=class_ids
      )
      self.evaluate(tf.global_variables_initializer())
      loss = self.evaluate(xent_loss.total_xent)
      log_perplexity = self.evaluate(xent_loss.avg_xent)
      print(['testSimpleFullSoftmax_GlobalVN loss', loss])
      print(['testSimpleFullSoftmax_GlobalVN log_perplexity', log_perplexity])
      self.assertNear(loss, 16.186937, 1e-4)
      self.assertNear(log_perplexity, 7.35769, 1e-4)

  @test_utils.SkipIfEager
  def testSimpleFullSoftmax_PerStepVN(self):
    with self.session(use_gpu=False):
      class_ids = tf.constant([1, 5, 10], dtype=tf.int32)
      class_weights = tf.constant([1.0, 0.4, 0.8], dtype=tf.float32)
      np.random.seed(12345)
      inputs = [tf.constant(np.random.rand(3, 10), dtype=tf.float32)]

      params = layers.SimpleFullSoftmax.Params()
      params.name = 'softmax'
      params.input_dim = 10
      params.num_classes = 32
      params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)
      params.vn.global_vn = False
      params.vn.per_step_vn = True
      params.vn.seed = 23456
      params.vn.scale = 1.0
      softmax = layers.SimpleFullSoftmax(params)
      xent_loss = softmax.XentLoss(
          inputs, class_weights=class_weights, class_ids=class_ids
      )
      self.evaluate(tf.global_variables_initializer())
      loss = self.evaluate(xent_loss.total_xent)
      log_perplexity = self.evaluate(xent_loss.avg_xent)
      print(['testShardedFullSoftmax_PerStepVN loss', loss])
      print(['testShardedFullSoftmax_PerStepVN log_perplexity', log_perplexity])
      self.assertNear(loss, 8.315969, 1e-4)
      self.assertNear(log_perplexity, 3.779986, 1e-4)

  def testSimpleFullSoftmax_FakeQuantized(self):
    default_qdomain = quant_utils.SymmetricScheduledClipQDomain.Params()
    default_qdomain.cc_schedule = (
        quant_utils.FakeQuantizationSchedule.Params().Set(
            clip_start_step=0, clip_end_step=2, quant_start_step=2
        )
    )
    logits_qdomain = default_qdomain.Copy()
    xent_loss = self._RunSimpleFullSoftmax(
        default_qdomain=default_qdomain,
        logits_qdomain=logits_qdomain,
        training_step=5,
    )
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    print(['loss', loss])
    print(['log_perplexity', log_perplexity])
    self.assertNear(loss, 6.285590, 1e-5)
    self.assertNear(log_perplexity, 2.857086, 1e-5)

  def _RunSimpleFullSoftmaxGradientChecker(
      self, batch_size, num_classes, chunk_size, num_shards
  ):
    for dtype, use_gpu, tolerance in [
        (tf.float32, True, 1e-2),
        (tf.float64, False, 1e-4),
    ]:
      tf.logging.info('dtype %s tolerance %g', dtype, tolerance)
      with contextlib.ExitStack() as context_stack:
        g = layers_test._ResetTfStatus(self, context_stack)
      with self.session(use_gpu=use_gpu, graph=g) as sess:
        input_dim = 10
        np.random.seed(12345)
        class_ids = tf.constant(
            np.random.randint(num_classes, size=(batch_size, 1)), dtype=tf.int32
        )
        class_weights = tf.constant(np.random.rand(batch_size), dtype=dtype)
        inputs = [
            tf.constant(np.random.rand(batch_size, input_dim), dtype=dtype)
        ]

        params = layers.SimpleFullSoftmax.Params()
        params.name = 'softmax'
        params.dtype = dtype
        params.input_dim = input_dim
        params.num_classes = num_classes
        params.num_shards = num_shards
        params.chunk_size = chunk_size
        params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)
        params.vn.global_vn = False
        softmax = layers.SimpleFullSoftmax(params)
        softmax_vars = softmax.vars.Flatten()
        self.evaluate(tf.global_variables_initializer())

        # pylint: disable=cell-var-from-loop
        @test_utils.DefineAndTrace()
        def _Grad():
          xent_loss = softmax.XentLoss(
              inputs, class_weights=class_weights, class_ids=class_ids
          )
          # softmax_vars = softmax.vars.Flatten()
          # Now add the backward graph.
          grads = tf.gradients(xent_loss.total_xent, softmax_vars)
          assert len(softmax_vars) == len(grads)
          return grads

        # pylint: enable=cell-var-from-loop

        grads = self.evaluate(_Grad)

        if tf.executing_eagerly():
          for var, grad_x in zip(softmax_vars, grads):
            x = tf.TensorSpec(shape=var.shape, dtype=var.dtype)

            # pylint: disable=cell-var-from-loop
            @test_utils.DefineAndTrace(x)
            def _TotalXent(x):
              var.assign(x)
              xent_loss = softmax.XentLoss(
                  inputs, class_weights=class_weights, class_ids=class_ids
              )
              return xent_loss.total_xent

            # pylint: enable=cell-var-from-loop

            grad_symbolic = grad_x
            grad_numeric = test_utils.ComputeNumericGradientEager(
                _TotalXent, var
            )
            self.assertAllClose(
                grad_symbolic, grad_numeric, atol=tolerance, rtol=tolerance
            )
        else:
          xent_loss = softmax.XentLoss(
              inputs, class_weights=class_weights, class_ids=class_ids
          )
          for x, grad_symbolic in zip(softmax_vars, grads):
            grad_numeric = test_utils.ComputeNumericGradient(
                sess, xent_loss.total_xent, x
            )
            self.assertAllClose(
                grad_symbolic, grad_numeric, atol=tolerance, rtol=tolerance
            )

  def testSimpleFullSoftmaxGradientChecker(self):
    self._RunSimpleFullSoftmaxGradientChecker(3, 4, 0, 1)
    self._RunSimpleFullSoftmaxGradientChecker(3, 4, 0, 2)
    self._RunSimpleFullSoftmaxGradientChecker(3, 4, 2, 2)
    self._RunSimpleFullSoftmaxGradientChecker(3, 4, 5, 2)

  def testSimpleFullSoftmax_SymbolicShape(self):
    with self.session(use_gpu=False):
      class_ids = tf.constant([1, 5, 10], dtype=tf.int32)
      class_weights = tf.constant([1.0, 0.4, 0.8], dtype=tf.float32)
      np.random.seed(12345)
      inputs = [tf.constant(np.random.rand(3, 10), dtype=tf.float32)]

      # Use a symbol to represent the input dim.
      input_dim = symbolic.Symbol('input_dim')
      params = layers.SimpleFullSoftmax.Params()
      params.name = 'softmax'
      params.input_dim = input_dim
      params.num_classes = 32
      with symbolic.SymbolToValueMap(symbolic.STATIC_VALUES, {input_dim: 10}):
        softmax = layers.SimpleFullSoftmax(params)
        xent_loss = softmax.XentLoss(
            inputs, class_weights=class_weights, class_ids=class_ids
        )
        self.evaluate(tf.global_variables_initializer())
        self.evaluate(xent_loss.total_xent)


class SharedSoftmaxLayerTest(test_utils.TestCase):

  def _testSharedSoftmaxLayerEmbLookup(self, scale_sqrt_depth=False):
    with contextlib.ExitStack() as context_stack:
      g = layers_test._ResetTfStatus(self, context_stack)
      tf.random.set_seed(398847392)
      params = layers.SharedSoftmaxLayer.Params().Set(
          softmax=layers.SimpleFullSoftmax.Params().Set(
              num_shards=1, chunk_size=0, apply_pruning=False
          ),
          dtype=tf.float32,
          fprop_dtype=None,
          name='shared_layer',
          input_dim=128,
          num_classes=8000,
          params_init=py_utils.WeightInit.Gaussian(0.5, 123456),
          scale_sqrt_depth=scale_sqrt_depth,
          random_seed=12345678,
      )

      emb_layer = layers.SharedSoftmaxLayer(params)

      emb_matrix = tf.einsum(
          'ji', emb_layer.softmax.DenseWeights(emb_layer.softmax.theta).wm
      )
      ids = tf.constant([[89], [100]])
      outputs = emb_layer.EmbLookup(emb_layer.theta, ids)

    with self.session(use_gpu=True, graph=g):
      self.evaluate(tf.global_variables_initializer())
      emb_matrix_val, ids_val, outputs_val = self.evaluate(
          [emb_matrix, ids, outputs]
      )
      self.assertEqual(emb_matrix_val.shape, (8000, 128))
      self.assertEqual(ids_val.shape, (2, 1))
      self.assertEqual(outputs_val.shape, (2, 1, 128))
      if scale_sqrt_depth:
        emb_matrix_val *= params.input_dim**0.5
      self.assertAllClose(emb_matrix_val[89, :], outputs_val[0, 0, :])
      self.assertAllClose(emb_matrix_val[100, :], outputs_val[1, 0, :])

  def testSharedSoftmaxLayerEmbLookup(self):
    self._testSharedSoftmaxLayerEmbLookup()

  def testSharedSoftmaxLayerEmbLookupScaling(self):
    self._testSharedSoftmaxLayerEmbLookup(True)


class EinsumSoftmaxLayerTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': 'no_label_smoothing', 'expected_loss': 27.981373},
      {
          'testcase_name': 'bfloat16_input',
          'expected_loss': 27.980932,
          'input_dtype': tf.bfloat16,
      },
      {
          'testcase_name': 'with_label_smoothing',
          'expected_loss': 28.038475,
          'label_smoothing': True,
      },
      {
          'testcase_name': 'focal_loss',
          'expected_loss': 27.539188,
          'focal_loss_gamma': 0.5,
      },
  )
  def testEinsumSoftmax(
      self,
      expected_loss,
      label_smoothing=False,
      input_dtype=tf.float32,
      focal_loss_gamma=None,
  ):
    with self.session(use_gpu=False) as sess:
      tf.random.set_seed(123)
      input_dim = 10
      num_classes = 32
      params = layers.EinsumSoftmax.Params().Set(
          name='softmax',
          input_dim=input_dim,
          num_classes=num_classes,
          focal_loss_gamma=focal_loss_gamma,
      )
      params.random_seed = 12345678
      softmax = params.Instantiate()
      sess.run(tf.global_variables_initializer())
      np.random.seed(12345)
      inputs = tf.constant(np.random.rand(2, 4, 10), dtype=input_dtype)
      logits = softmax.Logits(softmax.theta, inputs)
      self.assertAllEqual([2, 4, num_classes], py_utils.GetShape(logits))
      class_ids = tf.constant([[3, 4, 5, 2], [4, 5, 6, 2]], dtype=tf.int32)
      class_weights = tf.ones_like(class_ids, dtype=tf.float32)
      if label_smoothing:
        class_onehots = tf.one_hot(
            class_ids, depth=num_classes, dtype=tf.float32
        )
        class_probabilities = (
            0.1 / (num_classes - 1) * (1.0 - class_onehots)
            + 0.9 * class_onehots
        )
      else:
        class_probabilities = None
      per_example_loss, per_example_argmax = softmax.XentLossFromLogits(
          softmax.theta, logits, class_weights, class_ids, class_probabilities
      )
      self.assertAllEqual([2, 4], py_utils.GetShape(per_example_loss))
      self.assertAllClose(
          expected_loss, sess.run(tf.reduce_sum(per_example_loss))
      )
      self.assertAllEqual([2, 4], py_utils.GetShape(per_example_argmax))


class FocalFullSoftmaxLayerTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('no_smooth_no_focal', False, None, 27.981373),
      ('no_smooth_focal', False, 0.5, 27.539188),
      ('smooth_no_focal', True, None, 28.038475),
      ('smooth_focal', True, 0.5, 27.5981063),
  )
  def testFocalFullSoftmax(self, label_smoothing, gamma, expected_loss):
    with self.session(use_gpu=False) as sess:
      tf.random.set_seed(123)
      input_dim = 10
      num_classes = 32
      params = layers.FocalFullSoftmax.Params().Set(
          name='softmax',
          input_dim=input_dim,
          num_classes=num_classes,
          focal_loss_gamma=gamma,
      )
      params.random_seed = 12345678
      softmax = params.Instantiate()
      sess.run(tf.global_variables_initializer())
      np.random.seed(12345)
      inputs = tf.constant(np.random.rand(8, 10), dtype=tf.float32)
      logits = softmax.Logits(softmax.theta, inputs)
      self.assertAllEqual([8, num_classes], py_utils.GetShape(logits))
      class_ids = tf.constant([3, 4, 5, 2, 4, 5, 6, 2], dtype=tf.int32)
      class_weights = tf.ones_like(class_ids, dtype=tf.float32)
      if label_smoothing:
        class_onehots = tf.one_hot(
            class_ids, depth=num_classes, dtype=tf.float32
        )
        class_probabilities = (
            0.1 / (num_classes - 1) * (1.0 - class_onehots)
            + 0.9 * class_onehots
        )
      else:
        class_probabilities = None
      per_example_loss, per_example_argmax = softmax.XentLossFromLogits(
          softmax.theta, logits, class_weights, class_ids, class_probabilities
      )
      self.assertAllEqual([8], py_utils.GetShape(per_example_loss))
      self.assertAllClose(
          expected_loss, sess.run(tf.reduce_sum(per_example_loss))
      )
      self.assertAllEqual([8], py_utils.GetShape(per_example_argmax))


class SingleShardSoftmaxLayerTest(test_utils.TestCase):

  def _RunSimpleFullSoftmax(self,
                            inputs=None,
                            class_ids=None,
                            class_weights=None,
                            class_probabilities=None,
                            chunk_size=0,
                            dtype=tf.float32,
                            fprop_dtype=None):
    if fprop_dtype is None:
      fprop_dtype = dtype
    with self.session(use_gpu=True, graph=tf.Graph()):
      inputs = tf.constant(inputs, dtype=fprop_dtype)
      if class_ids is not None:
        class_ids = tf.constant(class_ids, dtype=tf.int32)
      if class_weights is not None:
        class_weights = tf.constant(class_weights, dtype=dtype)
      if class_probabilities is not None:
        class_probabilities = tf.constant(class_probabilities, dtype=dtype)

      params = layers.SingleShardFullSoftmax.Params()
      params.dtype = dtype
      params.fprop_dtype = fprop_dtype
      params.name = 'softmax'
      params.input_dim = 10
      params.num_classes = 32
      params.chunk_size = chunk_size
      params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)
      params.random_seed = 12345678

      params.vn.global_vn = False
      softmax = params.Instantiate()
      xent_loss = softmax.FProp(
          softmax.theta,
          inputs,
          class_weights=class_weights,
          class_ids=class_ids,
          class_probabilities=class_probabilities)

      self.evaluate(tf.global_variables_initializer())
      return self.evaluate(xent_loss)

  def testSoftmaxCapping(self):
    with self.session(use_gpu=True, graph=tf.Graph()):
      inputs = tf.constant(np.random.rand(4, 3, 10), dtype=tf.float32)
      class_weights = tf.constant(np.ones((4, 3, 1)), dtype=tf.float32)
      class_ids = tf.constant(
          np.random.randint(32, size=(4, 3, 1)), dtype=tf.int32)

      params = layers.SingleShardFullSoftmax.Params()
      params.name = 'softmax'
      params.input_dim = 10
      params.num_classes = 32
      params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)
      params.logits_soft_max = 1.0
      params.random_seed = 12345678

      params.vn.global_vn = False
      softmax = params.Instantiate()
      xent_loss = softmax.FProp(
          softmax.theta,
          inputs,
          class_weights=class_weights,
          class_ids=class_ids)

      self.evaluate(tf.global_variables_initializer())
      return self.evaluate(xent_loss)

  def testSimpleFullSoftmax_Non2D_ClassId(self):
    np.random.seed(1234578)
    xent_loss = self._RunSimpleFullSoftmax(
        inputs=np.random.rand(4, 3, 10),
        class_weights=np.ones((4, 3, 1)),
        class_ids=np.random.randint(32, size=(4, 3, 1)),
        chunk_size=2)
    self.assertEqual(xent_loss.per_example_xent.shape, (4, 3))
    self.assertEqual(xent_loss.per_example_weight.shape, (4, 3))

  def testSimpleFullSoftmax_Non2D_ClassProb(self):
    np.random.seed(12345)
    xent_loss = self._RunSimpleFullSoftmax(
        inputs=np.random.rand(4, 3, 10),
        class_weights=np.ones((4, 3, 1)),
        class_probabilities=np.random.randint(32, size=(4, 3, 32)),
        chunk_size=1)
    self.assertEqual(xent_loss.per_example_xent.shape, (4, 3))
    self.assertEqual(xent_loss.per_example_weight.shape, (4, 3))

  def _testSimpleFullSoftmax_Basic_Helper(self, dtype, fprop_dtype):
    np.random.seed(12345)
    class_ids = [[1], [5], [10]]
    class_weights = [[1.0], [0.4], [0.8]]
    inputs = np.random.rand(3, 10)
    xent_loss = self._RunSimpleFullSoftmax(
        inputs=inputs,
        class_weights=class_weights,
        class_ids=class_ids,
        dtype=dtype,
        fprop_dtype=fprop_dtype)
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    print(['loss', loss])
    print(['log_perplexity', log_perplexity])
    err = 1e-5
    if fprop_dtype == tf.float16 or fprop_dtype == tf.bfloat16:
      err = 1e-2
    self.assertNear(loss, 6.22425, err=err)
    self.assertNear(log_perplexity, 2.8292, err=err)
    self.assertAllEqual(xent_loss.per_example_argmax,
                        np.argmax(xent_loss.logits, axis=1))

  def testSimpleFullSoftmax_Basic_Float32(self):
    self._testSimpleFullSoftmax_Basic_Helper(
        dtype=tf.float32, fprop_dtype=tf.float32)

  def testSimpleFullSoftmax_Basic_Float32Float16(self):
    self._testSimpleFullSoftmax_Basic_Helper(
        dtype=tf.float32, fprop_dtype=tf.float16)

  def testSimpleFullSoftmax_Chunked(self):
    np.random.seed(12345)
    class_ids = [[1], [5], [10]]
    class_weights = [[1.0], [0.4], [0.8]]
    inputs = np.random.rand(3, 10)
    per_example_xent = None
    per_example_argmax = None
    for chunk_size in (0, 1, 3):
      xent_output = self._RunSimpleFullSoftmax(
          inputs=inputs,
          class_weights=class_weights,
          class_ids=class_ids,
          chunk_size=chunk_size)
      loss = xent_output.total_xent
      log_perplexity = xent_output.avg_xent
      print('xent_output ', xent_output)
      print('xent_output.per_example_argmax.dtype ',
            xent_output.per_example_argmax.dtype)
      self.assertAllClose(loss, 6.22425)
      self.assertAllClose(log_perplexity, 2.82920)
      if per_example_xent is None:
        per_example_xent = xent_output.per_example_xent
        per_example_argmax = xent_output.per_example_argmax
      else:
        self.assertAllClose(per_example_xent, xent_output.per_example_xent)
        self.assertAllClose(per_example_argmax, xent_output.per_example_argmax)

  def _RunSimpleFullSoftmaxGradientChecker(self, batch_size, num_classes,
                                           chunk_size):
    for dtype, use_gpu, tolerance in [
        (tf.float32, True, 1e-2),
        (tf.float64, False, 1e-4),
    ]:
      with contextlib.ExitStack() as context_stack:
        g = layers_test._ResetTfStatus(self, context_stack)
      tf.logging.info('dtype %s tolerance %g', dtype, tolerance)
      with self.session(use_gpu=use_gpu, graph=g) as sess:
        input_dim = 10
        np.random.seed(12345)
        class_ids = tf.constant(
            np.random.randint(num_classes, size=(batch_size, 1)),
            dtype=tf.int32)
        class_weights = tf.constant(np.random.rand(batch_size, 1), dtype=dtype)
        inputs = tf.constant(np.random.rand(batch_size, input_dim), dtype=dtype)

        params = layers.SingleShardFullSoftmax.Params()
        params.name = 'softmax'
        params.dtype = dtype
        params.input_dim = input_dim
        params.num_classes = num_classes
        params.chunk_size = chunk_size
        params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)
        params.vn.global_vn = False
        softmax = params.Instantiate()
        softmax_vars = softmax.vars.Flatten()
        self.evaluate(tf.global_variables_initializer())
        # pylint: disable=cell-var-from-loop
        @test_utils.DefineAndTrace()
        def _Grad():
          xent_loss = softmax.FProp(
              softmax.theta,
              inputs,
              class_weights=class_weights,
              class_ids=class_ids)
          # Now add the backward graph.
          grads = tf.gradients(xent_loss.total_xent, softmax_vars)
          assert len(softmax_vars) == len(grads)
          return grads

        # pylint: enable=cell-var-from-loop

        grads = self.evaluate(_Grad)

        if tf.executing_eagerly():
          for var, grad_x in zip(softmax_vars, grads):
            x = tf.TensorSpec(shape=var.shape, dtype=var.dtype)
            # pylint: disable=cell-var-from-loop
            @test_utils.DefineAndTrace(x)
            def _TotalXent(x):
              var.assign(x)
              xent_loss = softmax.FProp(
                  softmax.theta,
                  inputs,
                  class_weights=class_weights,
                  class_ids=class_ids)
              return xent_loss.total_xent

            # pylint: enable=cell-var-from-loop

            grad_symbolic = grad_x
            grad_numeric = test_utils.ComputeNumericGradientEager(
                _TotalXent, var)
            self.assertAllClose(
                grad_symbolic, grad_numeric, atol=tolerance, rtol=tolerance)
        else:
          xent_loss = softmax.FProp(
              softmax.theta,
              inputs,
              class_weights=class_weights,
              class_ids=class_ids)
          for x, grad_symbolic in zip(softmax_vars, grads):
            grad_numeric = test_utils.ComputeNumericGradient(
                sess, xent_loss.total_xent, x)
            self.assertAllClose(
                grad_symbolic, grad_numeric, atol=tolerance, rtol=tolerance)

  def testSimpleFullSoftmaxGradientChecker(self):
    self._RunSimpleFullSoftmaxGradientChecker(3, 4, 0)
    self._RunSimpleFullSoftmaxGradientChecker(3, 4, 1)
    self._RunSimpleFullSoftmaxGradientChecker(3, 4, 3)


class SoftmaxLayerLogitsTest(test_utils.TestCase):
  """Testing SoftmaxLayer.Logits()."""

  def _Logits(self, params, batch_size=2, seq_length=None):
    with self.session(use_gpu=True, graph=tf.Graph()):
      np.random.seed(12345)
      tf.random.set_seed(1234)

      params.name = 'softmax'
      if not params.input_dim:
        params.input_dim = 3
      if not params.num_classes:
        params.num_classes = 4
      params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)
      softmax = params.Instantiate()

      input_dim = params.input_dim
      if seq_length:
        inputs = np.random.rand(batch_size, seq_length, input_dim)
      else:
        inputs = np.random.rand(batch_size, input_dim)
      inputs = tf.constant(inputs, dtype=py_utils.FPropDtype(params))
      logits = softmax.Logits(softmax.theta, inputs)

      if seq_length:
        logits = py_utils.HasShape(logits,
                                   [batch_size, seq_length, params.num_classes])
      else:
        logits = py_utils.HasShape(logits, [batch_size, params.num_classes])
      self.evaluate(tf.global_variables_initializer())
      return self.evaluate(logits)

  def testConvSoftmaxLogits(self):
    params = layers.ConvSoftmax.Params()
    self.assertAllClose([[0.52536774, -0.17598523, 0.38314393, -0.36068222],
                         [0.75792629, -0.18001975, 0.42298675, -0.35423514]],
                        self._Logits(params))

  def testSimpleFullSoftmax(self):
    params = layers.SimpleFullSoftmax.Params()
    self.assertAllClose([[0.52536774, -0.17598523, 0.38314393, -0.36068222],
                         [0.75792629, -0.18001975, 0.42298675, -0.35423514]],
                        self._Logits(params))

  def testConvSoftmaxLogitsWith3DInputs(self):
    params = layers.ConvSoftmax.Params()
    logits = self._Logits(params, seq_length=5)
    self.assertAllClose(6.9934864, np.sum(logits))


class SingleShardSharedEmbeddingSoftmaxLayerTest(test_utils.TestCase):

  def testSingleShardSharedEmbeddingSoftmaxLayer(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      params = layers.SingleShardSharedEmbeddingSoftmax.Params()
      params.name = 'emb'
      params.dtype = tf.float32
      params.vocab_size = 128
      params.num_classes = 128
      params.embedding_dim = 8
      params.input_dim = 8
      params.emb_with_matmul = True
      params.params_init = py_utils.WeightInit.Gaussian(0.01)
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      emb_layer = params.Instantiate()
      ids = tf.constant([[89], [100]])
      embs = emb_layer.EmbLookupDefaultTheta(ids)
      embs_sum = tf.reduce_sum(embs)
      self.evaluate(tf.global_variables_initializer())
      test_utils.CompareToGoldenSingleFloat(self, -0.031068, self.evaluate(embs_sum))  # pylint: disable=line-too-long

if __name__ == '__main__':
  test_utils.main()
