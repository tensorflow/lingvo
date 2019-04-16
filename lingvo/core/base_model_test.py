# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for base_model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
from six.moves import range

import tensorflow as tf
from lingvo.core import base_decoder
from lingvo.core import base_encoder
from lingvo.core import base_input_generator
from lingvo.core import base_layer
from lingvo.core import base_model
from lingvo.core import base_model_params
from lingvo.core import hyperparams
from lingvo.core import layers
from lingvo.core import py_utils
from lingvo.core import task_scheduler
from lingvo.core import test_utils

FLAGS = tf.flags.FLAGS

_NUMPY_RANDOM_SEED = 9885784


class BaseTaskTest(test_utils.TestCase):

  def testStatsCounter(self):
    with self.session() as sess:
      foo = base_model.StatsCounter('foo')
      val = foo.Value()
      params = base_layer.BaseLayer.Params()
      inc = foo.IncBy(params, 100)

      tf.global_variables_initializer().run()
      self.assertAllEqual(0, val.eval())
      self.assertAllEqual(100, sess.run(inc))
      self.assertAllEqual(100, val.eval())
      self.assertAllEqual([100, 200], sess.run([val, inc]))
      self.assertAllEqual([200, 300], sess.run([val, inc]))

  @classmethod
  def TestParams(cls):
    p = base_model.BaseTask.Params()
    p.name = 'base_mdl'
    p.encoder = base_encoder.BaseEncoder.Params()
    p.encoder.name = 'encoder'
    p.decoder = base_decoder.BaseDecoder.Params()
    p.decoder.name = 'decoder'
    return p

  def testInit(self):
    p = self.TestParams()
    p.input = base_input_generator.BaseSequenceInputGenerator.Params()
    _ = p.cls(p)

  def testScaleGradients(self):
    p = self.TestParams()
    p.input = base_input_generator.BaseSequenceInputGenerator.Params()
    task = p.cls(p)
    task.CreateVariable(
        'a',
        py_utils.WeightParams(shape=[], init=py_utils.WeightInit.Constant(0)))
    var_a = task.theta.a
    var_grads = py_utils.NestedMap(a=(var_a, tf.ones_like(var_a)))
    scaled_grads_map = task.ScaleGradients(var_grads)

    FLAGS.enable_check_numerics = False
    with self.session():
      tf.global_variables_initializer().run()
      self.assertFalse(scaled_grads_map.has_nan_or_inf.eval())
      self.assertEqual(1.0, scaled_grads_map.grad_scale.eval())
      # The final gradient must be finite.
      self.assertFalse(tf.is_nan(scaled_grads_map.final_var_grads.a[1]).eval())
      self.assertTrue(
          tf.is_finite(scaled_grads_map.final_var_grads.a[1]).eval())

  def testScaleGradientsInf(self):
    FLAGS.enable_check_numerics = False
    p = self.TestParams()
    p.input = base_input_generator.BaseSequenceInputGenerator.Params()
    task = p.cls(p)
    task.CreateVariable(
        'a',
        py_utils.WeightParams(shape=[], init=py_utils.WeightInit.Constant(0)))
    var_a = task.theta.a
    # Infinite gradient.
    var_grads = py_utils.NestedMap(a=(var_a, tf.log(0.)))
    scaled_grads_map = task.ScaleGradients(var_grads)

    with self.session():
      tf.global_variables_initializer().run()
      self.assertTrue(scaled_grads_map.has_nan_or_inf.eval())
      self.assertEqual(0., scaled_grads_map.grad_scale.eval())
      # The final gradient must be finite.
      self.assertFalse(tf.is_nan(scaled_grads_map.final_var_grads.a[1]).eval())
      self.assertTrue(
          tf.is_finite(scaled_grads_map.final_var_grads.a[1]).eval())

  def testScaleGradientsNaN(self):
    FLAGS.enable_check_numerics = False
    p = self.TestParams()
    p.input = base_input_generator.BaseSequenceInputGenerator.Params()
    task = p.cls(p)
    task.CreateVariable(
        'a',
        py_utils.WeightParams(shape=[], init=py_utils.WeightInit.Constant(0)))
    var_a = task.theta.a
    # Make a NaN gradient.
    var_grads = py_utils.NestedMap(a=(var_a, 0. * tf.log(0.)))
    scaled_grads_map = task.ScaleGradients(var_grads)

    with self.session():
      tf.global_variables_initializer().run()
      self.assertTrue(scaled_grads_map.has_nan_or_inf.eval())
      self.assertEqual(0., scaled_grads_map.grad_scale.eval())
      # The final gradient must be finite.
      self.assertFalse(tf.is_nan(scaled_grads_map.final_var_grads.a[1]).eval())
      self.assertTrue(
          tf.is_finite(scaled_grads_map.final_var_grads.a[1]).eval())

  def testScaleGradientsCheckNumerics(self):
    """ScaleGradients when enable_check_numerics=True."""
    FLAGS.enable_check_numerics = True
    p = self.TestParams()
    p.input = base_input_generator.BaseSequenceInputGenerator.Params()
    task = p.cls(p)
    task.CreateVariable(
        'a',
        py_utils.WeightParams(shape=[], init=py_utils.WeightInit.Constant(0)))
    var_a = task.theta.a
    # Make a NaN gradient.
    var_grads = py_utils.NestedMap(a=(var_a, 0. * tf.log(0.)))
    scaled_grads_map = task.ScaleGradients(var_grads)

    with self.session():
      tf.global_variables_initializer().run()
      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                   'is not finite'):
        self.assertTrue(scaled_grads_map.has_nan_or_inf.eval())
        self.assertEqual(0., scaled_grads_map.grad_scale.eval())
        # The final gradient must be finite.
        self.assertFalse(
            tf.is_nan(scaled_grads_map.final_var_grads.a[1]).eval())
        self.assertTrue(
            tf.is_finite(scaled_grads_map.final_var_grads.a[1]).eval())

  def testScaleGradientsError(self):
    p = self.TestParams()
    p.input = base_input_generator.BaseSequenceInputGenerator.Params()
    p.train.clip_gradient_single_norm_to_value = 1.0
    p.train.clip_gradient_norm_to_value = 1.0
    task = p.cls(p)
    task.CreateVariable(
        'a',
        py_utils.WeightParams(shape=[], init=py_utils.WeightInit.Constant(0)))
    var_a = task.theta.a
    var_grads = py_utils.NestedMap(a=(var_a, tf.ones_like(var_a)))
    self.assertRaises(ValueError, task.ScaleGradients, var_grads)

  def testScaleGradientsSingleTensorNorm(self):
    p = self.TestParams()
    p.input = base_input_generator.BaseSequenceInputGenerator.Params()
    p.train.clip_gradient_single_norm_to_value = 1.0
    p.train.clip_gradient_norm_to_value = None
    task = p.cls(p)
    task.CreateVariable(
        'a',
        py_utils.WeightParams(shape=[], init=py_utils.WeightInit.Constant(0)))
    task.CreateVariable(
        'b',
        py_utils.WeightParams(shape=[], init=py_utils.WeightInit.Constant(0)))

    var_a = task.theta.a
    var_b = task.theta.b
    var_grads = py_utils.NestedMap(
        a=(var_a, tf.ones_like(var_a) * 10.0),
        b=(var_b, tf.ones_like(var_b) * 0.5))
    scaled_grads_map = task.ScaleGradients(var_grads)

    FLAGS.enable_check_numerics = False
    with self.session():
      tf.global_variables_initializer().run()

      # Each variable is clipped indipendently to grad scale of 1.
      self.assertAllClose(scaled_grads_map.final_var_grads.a[1].eval(), 1.0)
      self.assertAllClose(scaled_grads_map.final_var_grads.b[1].eval(), 0.5)


class TeacherTask(base_model.BaseTask):

  @base_layer.initializer
  def __init__(self, params):
    super(TeacherTask, self).__init__(params)
    p = self.params
    with tf.variable_scope(p.name):
      self.CreateVariable('x',
                          py_utils.WeightParams(
                              shape=[], init=py_utils.WeightInit.Constant(0)))

  def ComputePredictions(self, theta, input_batch):
    return theta.x


class StudentTask(base_model.BaseTask):

  @base_layer.initializer
  def __init__(self, params):
    super(StudentTask, self).__init__(params)
    p = self.params
    with tf.variable_scope(p.name):
      self.CreateVariable('x',
                          py_utils.WeightParams(
                              shape=[], init=py_utils.WeightInit.Uniform()))

  def ComputePredictions(self, theta, input_batch):
    return theta.x


class TestInputGenerator(base_input_generator.BaseSequenceInputGenerator):

  def __init__(self, params):
    super(TestInputGenerator, self).__init__(params)
    self._input_batch_size = tf.constant(1)

  def InputBatch(self):
    return 0


class DistillationTestTask(base_model.DistillationTask):

  @classmethod
  def Params(cls):
    p = super(DistillationTestTask, cls).Params()
    p.name = 'distillation_test'
    p.teacher = TeacherTask.Params()
    p.student = StudentTask.Params()
    p.input = TestInputGenerator.Params()
    p.train.learning_rate = 1e3
    p.teacher.train = None
    p.teacher.eval = None
    p.student.train = None
    p.student.eval = None
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(DistillationTestTask, self).__init__(params)

  def ComputeLoss(self, theta, input_batch, predictions):
    return {'loss': (predictions.teacher - predictions.student, 1)}, {}


class DistillationTaskTest(test_utils.TestCase):

  def testFProp(self):
    p = DistillationTestTask.Params()
    task = p.cls(p)
    self.assertFalse(task.params.is_eval)
    self.assertFalse(task.teacher.params.is_eval)
    self.assertIsNotNone(task.teacher.params.input)
    self.assertFalse(task.student.params.is_eval)
    self.assertIsNotNone(task.student.params.input)
    metrics = task.FPropDefaultTheta()[0]
    self.assertItemsEqual(['loss', 'num_samples_in_batch'],
                          list(metrics.keys()))
    task.BProp()
    # Expected side effects of BProp().
    self.assertIsNotNone(task.train_op)
    self.assertIsNotNone(task.total_examples)

    with self.session() as sess:
      tf.global_variables_initializer().run()

      variables = {}
      values_before_training = {}
      values_after_training = {}
      for child in ('teacher', 'student'):
        variables[child] = {
            k: v
            for k, v in getattr(task, child).vars.FlattenItems()
        }
        values_before_training[child] = sess.run(variables[child])

      # Train for a few steps.
      for _ in range(10):
        sess.run(task.train_op)

      for child in ('teacher', 'student'):
        values_after_training[child] = sess.run(variables[child])
        for k, v in six.iteritems(values_after_training[child]):
          print('Comparing variable %s' % k)
          if child == 'teacher':
            # Teacher vars should not change after training.
            self.assertAllEqual(values_before_training[child][k], v)
          else:
            # Student vars should change after training.
            self.assertNotAlmostEqual(values_before_training[child][k], v)


class SingleTaskModelTest(test_utils.TestCase):

  def testInit(self):
    p = base_model.SingleTaskModel.Params()
    p.task = BaseTaskTest.TestParams()
    p.task.input = base_input_generator.BaseSequenceInputGenerator.Params()
    model = p.cls(p)
    self.assertEqual(model.params.name, model.GetTask().params.name)
    self.assertEqual(model.params.task, model.GetTask().params)
    self.assertEqual(len(model.tasks), 1)
    self.assertEqual(model.tasks[0], model.GetTask())
    self.assertEqual(model.tasks[0], model.SampleTask(None))

  def testExponentialMovingAverage(self):
    p = base_model.SingleTaskModel.Params()
    p.task = BaseTaskTest.TestParams()
    p.task.input = base_input_generator.BaseSequenceInputGenerator.Params()
    p.train.ema_decay = 0.9
    model = p.cls(p)
    model._task.CreateChild('a',
                            layers.BatchNormLayer.Params().Set(name='a', dim=1))
    model._task._train_op = tf.no_op()
    model._task.ApplyExponentialMovingAverage(model.ema)
    with tf.variable_scope('', reuse=True):
      beta = tf.get_variable('a/beta/var')
      mean = tf.get_variable('a/moving_mean/var')
      self.assertIsNotNone(model.ema.average(beta))
      self.assertIsNone(model.ema.average(mean))


class MultiTaskModelTest(test_utils.TestCase):

  def testInit(self):
    p = base_model.MultiTaskModel.Params()
    p.name = 'MultiTaskModel'
    p0 = BaseTaskTest.TestParams()
    p1 = BaseTaskTest.TestParams()

    p.input = base_model_params.MultiTaskModelParams().Train()
    p.input.Define('a',
                   base_input_generator.BaseSequenceInputGenerator.Params(), '')
    p.input.Define('b',
                   base_input_generator.BaseSequenceInputGenerator.Params(), '')

    p.task_params = hyperparams.Params()
    p.task_params.Define('a', p0, '')
    p.task_params.Define('b', p1, '')

    p.task_probs = hyperparams.Params()
    p.task_probs.Define('a', 0.5, '')
    p.task_probs.Define('b', 0.5, '')

    model = p.cls(p)
    self.assertEqual(len(model.tasks), 2)
    self.assertEqual(set(model.task_names), {'a', 'b'})
    self.assertEqual(set(model.tasks), {model.GetTask('a'), model.GetTask('b')})
    self.assertEqual(model.params.task_params.a, model.GetTask('a').params)
    self.assertEqual(model.params.task_params.b, model.GetTask('b').params)

  def _setUpTestSampleTask(self):
    np.random.seed(_NUMPY_RANDOM_SEED)

    # define and initialize tasks, model and params
    p = base_model.MultiTaskModel.Params()
    p.name = 'MultiTaskModel'
    p0 = BaseTaskTest.TestParams()
    p1 = BaseTaskTest.TestParams()

    p.input = base_model_params.MultiTaskModelParams().Train()
    p.input.Define('a',
                   base_input_generator.BaseSequenceInputGenerator.Params(), '')
    p.input.Define('b',
                   base_input_generator.BaseSequenceInputGenerator.Params(), '')

    p.task_params = hyperparams.Params()
    p.task_params.Define('a', p0, '')
    p.task_params.Define('b', p1, '')

    return p

  def _testSampleTaskHelper(self, p):
    model = p.cls(p)

    task_to_id = {model.children['a']: 'a', model.children['b']: 'b'}
    task_counts = {'a': 0, 'b': 0}

    # initialize tensorflow graph and global step
    with self.session() as sess:
      tf.global_variables_initializer().run()
      global_step = sess.run(model.global_step)
      for _ in range(100):
        task = model.SampleTask(global_step)
        task_counts[task_to_id[task]] += 1

      self.assertEqual(task_counts['a'], 83)
      self.assertEqual(task_counts['b'], 17)

  def testSampleTaskSpecifiedWithoutScheduler(self):
    """Expected distribution: 'a': 0.8 , 'b': 0.2."""
    p = self._setUpTestSampleTask()

    p.task_probs = hyperparams.Params()
    p.task_probs.Define('a', 0.8, '')
    p.task_probs.Define('b', 0.2, '')

    self._testSampleTaskHelper(p)

  def testSampleTask(self):
    """Expected distribution: 'a': 0.8 , 'b': 0.2."""
    p = self._setUpTestSampleTask()

    p.task_schedule = task_scheduler.ConstantScheduler.Params()
    p.task_schedule.task_probs = [('a', 0.8), ('b', 0.2)]

    self._testSampleTaskHelper(p)


if __name__ == '__main__':
  tf.test.main()
