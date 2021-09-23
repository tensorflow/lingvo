# Lint as: python3
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

from absl.testing import flagsaver
from absl.testing import parameterized
import lingvo.compat as tf
from lingvo.core import base_decoder
from lingvo.core import base_input_generator
from lingvo.core import base_layer
from lingvo.core import base_model
from lingvo.core import base_model_params
from lingvo.core import distillation_task
from lingvo.core import hyperparams
from lingvo.core import layers
from lingvo.core import learner
from lingvo.core import optimizer
from lingvo.core import py_utils
from lingvo.core import task_scheduler
from lingvo.core import test_utils
import numpy as np


FLAGS = tf.flags.FLAGS

_NUMPY_RANDOM_SEED = 9885784


class TestTask(base_model.BaseTask):

  def __init__(self, params):
    super().__init__(params)
    self.CreateChild('x', layers.BatchNormLayer.Params().Set(dim=1))
    self.CreateChild(
        'y',
        layers.ProjectionLayer.Params().Set(
            input_dim=1, output_dim=1, batch_norm=True, has_bias=True))

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    self.CreateVariable(
        'a',
        py_utils.WeightParams(shape=[], init=py_utils.WeightInit.Constant(0)))
    self.CreateVariable(
        'b',
        py_utils.WeightParams(shape=[], init=py_utils.WeightInit.Constant(0)))

  def FPropTower(self, theta, input_batch):
    return {'loss': (theta.a, 1.0)}, {}


class BaseTaskTest(test_utils.TestCase):

  @classmethod
  def TestParams(cls):
    p = TestTask.Params()
    p.name = 'base_mdl'
    p.encoder = base_layer.BaseLayer.Params()
    p.encoder.name = 'encoder'
    p.decoder = base_decoder.BaseDecoder.Params()
    p.decoder.name = 'decoder'
    return p

  def testInit(self):
    _ = self.TestParams().Instantiate()

  @flagsaver.flagsaver
  def testScaleGradients(self):
    p = self.TestParams()
    task = p.Instantiate()
    var_a = task.theta.a
    var_grads = py_utils.NestedMap(
        a=py_utils.VarGrad(var_a, tf.ones_like(var_a)))
    scaled_grads_map = task.learners[0].ScaleGradients(var_grads)

    FLAGS.enable_check_numerics = False
    with self.session():
      self.evaluate(tf.global_variables_initializer())
      self.assertEqual(1.0, scaled_grads_map.grad_scale.eval())
      # The final gradient must be finite.
      self.assertFalse(
          tf.math.is_nan(scaled_grads_map.final_var_grads.a[1]).eval())
      self.assertTrue(
          tf.math.is_finite(scaled_grads_map.final_var_grads.a[1]).eval())

  @flagsaver.flagsaver
  def testScaleGradientsInf(self):
    FLAGS.enable_check_numerics = False
    p = self.TestParams()
    task = p.Instantiate()
    var_a = task.theta.a
    # Infinite gradient.
    var_grads = py_utils.NestedMap(a=py_utils.VarGrad(var_a, tf.math.log(0.)))
    scaled_grads_map = task.learners[0].ScaleGradients(var_grads)

    with self.session():
      self.evaluate(tf.global_variables_initializer())
      self.assertEqual(0., scaled_grads_map.grad_scale.eval())
      # The final gradient must be finite.
      self.assertFalse(
          tf.math.is_nan(scaled_grads_map.final_var_grads.a[1]).eval())
      self.assertTrue(
          tf.math.is_finite(scaled_grads_map.final_var_grads.a[1]).eval())

  @flagsaver.flagsaver
  def testScaleGradientsNaN(self):
    FLAGS.enable_check_numerics = False
    p = self.TestParams()
    task = p.Instantiate()
    var_a = task.theta.a
    # Make a NaN gradient.
    var_grads = py_utils.NestedMap(
        a=py_utils.VarGrad(var_a, 0. * tf.math.log(0.)))
    scaled_grads_map = task.learners[0].ScaleGradients(var_grads)

    with self.session():
      self.evaluate(tf.global_variables_initializer())
      self.assertEqual(0., scaled_grads_map.grad_scale.eval())
      # The final gradient must be finite.
      self.assertFalse(
          tf.math.is_nan(scaled_grads_map.final_var_grads.a[1]).eval())
      self.assertTrue(
          tf.math.is_finite(scaled_grads_map.final_var_grads.a[1]).eval())

  @flagsaver.flagsaver
  def testScaleGradientsCheckNumerics(self):
    """ScaleGradients when enable_check_numerics=True."""
    FLAGS.enable_check_numerics = True
    p = self.TestParams()
    task = p.Instantiate()
    var_a = task.theta.a
    # Make a NaN gradient.
    var_grads = py_utils.NestedMap(
        a=py_utils.VarGrad(var_a, 0. * tf.math.log(0.)))
    scaled_grads_map = task.learners[0].ScaleGradients(var_grads)

    with self.session():
      self.evaluate(tf.global_variables_initializer())
      self.assertEqual(0., scaled_grads_map.grad_scale.eval())
      # Fetching the gradient raises an exception with enable_check_numerics.
      with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                  'is not finite'):
        _ = scaled_grads_map.final_var_grads.a[1].eval()

  def testScaleGradientsError(self):
    p = self.TestParams()
    p.train.clip_gradient_single_norm_to_value = 1.0
    p.train.clip_gradient_norm_to_value = 1.0
    task = p.Instantiate()
    var_a = task.theta.a
    var_grads = py_utils.NestedMap(
        a=py_utils.VarGrad(var_a, tf.ones_like(var_a)))
    self.assertRaises(ValueError, task.learners[0].ScaleGradients, var_grads)

  @flagsaver.flagsaver
  def testScaleGradientsSingleTensorNorm(self):
    p = self.TestParams()
    p.train.clip_gradient_single_norm_to_value = 1.0
    p.train.clip_gradient_norm_to_value = None
    task = p.Instantiate()

    var_a = task.theta.a
    var_b = task.theta.b
    var_grads = py_utils.NestedMap(
        a=py_utils.VarGrad(var_a,
                           tf.ones_like(var_a) * 10.0),
        b=py_utils.VarGrad(var_b,
                           tf.ones_like(var_b) * 0.5))
    scaled_grads_map = task.learners[0].ScaleGradients(var_grads)

    FLAGS.enable_check_numerics = False
    with self.session():
      self.evaluate(tf.global_variables_initializer())

      # Each variable is clipped indipendently to grad scale of 1.
      self.assertAllClose(scaled_grads_map.final_var_grads.a[1].eval(), 1.0)
      self.assertAllClose(scaled_grads_map.final_var_grads.b[1].eval(), 0.5)


class TeacherTask(base_model.BaseTask):

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    self.CreateVariable(
        'x',
        py_utils.WeightParams(shape=[], init=py_utils.WeightInit.Constant(0)))

  def ComputePredictions(self, theta, input_batch):
    return theta.x


class StudentTask(base_model.BaseTask):

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    self.CreateVariable(
        'x',
        py_utils.WeightParams(shape=[], init=py_utils.WeightInit.Uniform()))

  def ComputePredictions(self, theta, input_batch):
    return theta.x


class TestInputGenerator(base_input_generator.BaseSequenceInputGenerator):

  def InfeedBatchSize(self):
    """Override BaseSequenceInputGenerator."""
    return 1

  def GetPreprocessedInputBatch(self):
    return 0


class DistillationTestTask(distillation_task.DistillationTask):

  @classmethod
  def Params(cls):
    p = super().Params()
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

  def __init__(self, params):
    super().__init__(params)

  def ComputeLoss(self, theta, predictions, input_batch):
    return {'loss': (predictions.teacher - predictions.student, 1)}, {}


class DistillationTaskTest(test_utils.TestCase):

  def _GetVarValuesBeforeAndAfter(self, params, steps=10):
    task = params.Instantiate()
    self.assertIsNotNone(task.teacher.params.input)
    self.assertIsNotNone(task.student.params.input)
    metrics = task.FPropDefaultTheta()[0]
    self.assertCountEqual(['loss', 'num_samples_in_batch'],
                          list(metrics.keys()))
    task.BProp()
    # Expected side effects of BProp().
    self.assertIsNotNone(task.train_op)

    with self.session():
      self.evaluate(tf.global_variables_initializer())

      variables = {}
      values_before_training = {}
      values_after_training = {}
      for child in ('teacher', 'student'):
        variables[child] = {
            k: v
            for k, v in getattr(task, child).vars.FlattenItems()
        }
        values_before_training[child] = self.evaluate(variables[child])

      # Train for a few steps.
      for _ in range(10):
        self.evaluate(task.train_op)
      for child in ('teacher', 'student'):
        values_after_training[child] = self.evaluate(variables[child])
      return values_before_training, values_after_training

  def testFProp(self):
    values_before_training, values_after_training = (
        self._GetVarValuesBeforeAndAfter(DistillationTestTask.Params()))
    for child in ('teacher', 'student'):
      for k, v in values_after_training[child].items():
        print('Comparing variable %s' % k)
        if child == 'teacher':
          # Teacher vars should not change after training.
          self.assertAllEqual(values_before_training[child][k], v)
        else:
          # Student vars should change after training.
          self.assertNotAlmostEqual(values_before_training[child][k], v)

  def testFPropTeacherEnabled(self):
    params = DistillationTestTask.Params()
    params.train_teacher = True
    params.distillation_loss_weight.value = 0.5
    values_before_training, values_after_training = (
        self._GetVarValuesBeforeAndAfter(params))
    for child in ('teacher', 'student'):
      for k, v in values_after_training[child].items():
        print('Comparing variable %s' % k)
        if child == 'teacher':
          # Teacher vars should change after training.
          self.assertNotAlmostEqual(values_before_training[child][k], v)
        else:
          # Student vars should change after training.
          self.assertNotAlmostEqual(values_before_training[child][k], v)


class SingleTaskModelTest(test_utils.TestCase, parameterized.TestCase):

  def testInit(self):
    p = base_model.SingleTaskModel.Params()
    p.task = BaseTaskTest.TestParams()
    p.input = base_input_generator.BaseSequenceInputGenerator.Params()
    p.task.train.learner = learner.Learner.Params().Set(name='loss')
    model = p.Instantiate()
    self.assertEqual(model.params.name, model.GetTask().params.name)
    self.assertEqual(model.params.task, model.GetTask().params)
    self.assertLen(model.tasks, 1)
    self.assertEqual(model.tasks[0], model.GetTask())
    self.assertEqual(model.tasks[0], model.SampleTask(None))

  def testExponentialMovingAverage(self):
    p = base_model.SingleTaskModel.Params()
    p.task = BaseTaskTest.TestParams()
    p.task.input = base_input_generator.BaseSequenceInputGenerator.Params()
    p.task.train.ema_decay = 0.9
    p.task.train.ema_decay_moving_vars = False
    model = p.Instantiate()
    self.assertIsNotNone(model.ema)
    model.ConstructFPropBPropGraph()
    with tf.variable_scope('base_mdl', reuse=True):
      beta = tf.get_variable('x/beta/var')
      mean = tf.get_variable('x/moving_mean/var')
      self.assertIsNotNone(model.ema.average(beta))
      self.assertIsNone(model.ema.average(mean))

  def testExponentialMovingAverageIncludingMovingVars(self):
    p = base_model.SingleTaskModel.Params()
    p.task = BaseTaskTest.TestParams()
    p.task.input = base_input_generator.BaseSequenceInputGenerator.Params()
    p.task.train.ema_decay = 0.9
    p.task.train.ema_decay_moving_vars = True
    model = p.Instantiate()
    self.assertIsNotNone(model.ema)
    model.ConstructFPropBPropGraph()
    with tf.variable_scope('base_mdl', reuse=True):
      beta = tf.get_variable('x/beta/var')
      mean = tf.get_variable('x/moving_mean/var')
      self.assertIsNotNone(model.ema.average(beta))
      self.assertIsNotNone(model.ema.average(mean))

  @parameterized.named_parameters(
      ('SGD', optimizer.SGD.Params()),
      ('AdamV1', optimizer.Adam.Params()),
      ('AdamV2', optimizer.AdamV2.Params()),
  )
  def testModuleVarsTracking(self, optimizer_params):
    p = base_model.SingleTaskModel.Params()
    p.task = BaseTaskTest.TestParams()
    p.input = base_input_generator.BaseSequenceInputGenerator.Params()
    p.task.train.learner = learner.Learner.Params().Set(
        name='loss', optimizer=optimizer_params)
    model = p.Instantiate()
    model.ConstructFPropBPropGraph()
    self.assertEqual([
        'global_step:0',
        'base_mdl/a/var:0',
        'base_mdl/b/var:0',
        'base_mdl/x/beta/var:0',
        'base_mdl/x/gamma/var:0',
        'base_mdl/x/moving_mean/var:0',
        'base_mdl/x/moving_variance/var:0',
        'base_mdl/y/b/var:0',
        'base_mdl/y/w/var:0',
        'base_mdl/y/beta/var:0',
        'base_mdl/y/gamma/var:0',
        'base_mdl/y/moving_mean/var:0',
        'base_mdl/y/moving_variance/var:0',
    ], [x.name for x in model.variables])

  def testModuleVarsTrackingEMA(self):
    p = base_model.SingleTaskModel.Params()
    p.task = BaseTaskTest.TestParams()
    p.task.input = base_input_generator.BaseSequenceInputGenerator.Params()
    p.task.train.ema_decay = 0.9
    p.task.train.ema_decay_moving_vars = True
    model = p.Instantiate()
    self.assertIsNotNone(model.ema)
    model.ConstructFPropBPropGraph()
    self.assertCountEqual([
        'global_step:0',
        'base_mdl/a/var:0',
        'base_mdl/b/var:0',
        'base_mdl/x/beta/var:0',
        'base_mdl/x/gamma/var:0',
        'base_mdl/x/moving_mean/var:0',
        'base_mdl/x/moving_variance/var:0',
        'base_mdl/y/b/var:0',
        'base_mdl/y/w/var:0',
        'base_mdl/y/beta/var:0',
        'base_mdl/y/gamma/var:0',
        'base_mdl/y/moving_mean/var:0',
        'base_mdl/y/moving_variance/var:0',
        'base_mdl/a/var/ExponentialMovingAverage:0',
        'base_mdl/b/var/ExponentialMovingAverage:0',
        'base_mdl/x/beta/var/ExponentialMovingAverage:0',
        'base_mdl/x/gamma/var/ExponentialMovingAverage:0',
        'base_mdl/x/moving_mean/var/ExponentialMovingAverage:0',
        'base_mdl/x/moving_variance/var/ExponentialMovingAverage:0',
        'base_mdl/y/b/var/ExponentialMovingAverage:0',
        'base_mdl/y/beta/var/ExponentialMovingAverage:0',
        'base_mdl/y/gamma/var/ExponentialMovingAverage:0',
        'base_mdl/y/moving_mean/var/ExponentialMovingAverage:0',
        'base_mdl/y/moving_variance/var/ExponentialMovingAverage:0',
        'base_mdl/y/w/var/ExponentialMovingAverage:0',
    ], [x.name for x in model.variables])


class MultiTaskModelTest(test_utils.TestCase):

  def testInitMissingInputParams(self):
    p = base_model.MultiTaskModel.Params()
    p.name = 'MultiTaskModel'
    p0 = BaseTaskTest.TestParams()
    p0.train.learner = (learner.Learner.Params().Set(name='loss'))
    p1 = BaseTaskTest.TestParams()
    p1.train.learner = (learner.Learner.Params().Set(name='loss'))

    p.input = base_model_params.MultiTaskModelParams().Train()
    p.input.Define('a',
                   base_input_generator.BaseSequenceInputGenerator.Params(), '')

    p.task_params = hyperparams.Params()
    p.task_params.Define('a', p0, '')
    p.task_params.Define('b', p1, '')

    p.task_probs = hyperparams.Params()
    p.task_probs.Define('a', 0.5, '')
    p.task_probs.Define('b', 0.5, '')
    self.assertRaises(AttributeError, p.Instantiate)

  def testInit(self):
    p = base_model.MultiTaskModel.Params()
    p.name = 'MultiTaskModel'
    p0 = BaseTaskTest.TestParams()
    p0.train.learner = (learner.Learner.Params().Set(name='loss'))
    p1 = BaseTaskTest.TestParams()
    p1.train.learner = (learner.Learner.Params().Set(name='loss'))

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

    model = p.Instantiate()
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
    model = p.Instantiate()

    task_to_id = {model.children['a']: 'a', model.children['b']: 'b'}
    task_counts = {'a': 0, 'b': 0}

    # initialize tensorflow graph and global step
    with self.session():
      self.evaluate(tf.global_variables_initializer())
      global_step = self.evaluate(model.global_step)
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


class PostTrainingTask(base_model.BaseTask):

  def __init__(self, params):
    super().__init__(params)
    p = layers.FeedForwardNet.Params().Set(
        name='ffn',
        input_dim=10,
        hidden_layer_dims=[20, 30],
        batch_norm=True,
        activation='TANH',
        params_init=py_utils.WeightInit.Uniform(1.0))
    self.CreateChild('ffn', p)

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    self.CreateVariable(
        'counter1',
        py_utils.WeightParams(shape=[], init=py_utils.WeightInit.Constant(0)))
    self.CreateVariable(
        'counter2',
        py_utils.WeightParams(shape=[], init=py_utils.WeightInit.Constant(0)))

  def PostTrainingStepUpdate(self):
    # We expect the training step to be done, so capture
    # the value of counter1 into counter2.
    return tf.assign(self.vars.counter2, self.vars.counter1)

  def ComputePredictions(self, theta, input_batch):
    input_data = tf.random.normal([1, 10], dtype=tf.float32) + tf.cast(
        input_batch, tf.float32)
    add = tf.assign_add(self.vars.counter1, 1.)
    input_data += add
    result = self.ffn.FProp(theta.ffn, input_data)
    return {'result': result}

  def ComputeLoss(self, theta, predictions, input_batch):
    loss = tf.reduce_sum(predictions['result'])
    return {'loss': (loss, 1)}, {}


class PostTrainingTest(test_utils.TestCase):

  @classmethod
  def TestParams(cls):
    p = PostTrainingTask.Params()
    p.name = 'base_mdl'
    p.input = TestInputGenerator.Params()
    return p

  def testPost(self):
    p = self.TestParams()
    task = p.Instantiate()
    task.FPropDefaultTheta()
    task.BProp()
    train_op = task.train_op
    with self.session():
      self.evaluate(tf.global_variables_initializer())
      for _ in range(20):
        self.evaluate(train_op)
        c1, c2 = self.evaluate([task.vars.counter1, task.vars.counter2])
        # Both vars should have the same value if the PostTrainingStep
        # happens after the training step.
        self.assertEqual(c1, c2)


if __name__ == '__main__':
  tf.test.main()
