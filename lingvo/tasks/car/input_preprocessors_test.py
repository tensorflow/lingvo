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
"""Input preprocessors tests."""

from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.core import schedule
from lingvo.core import test_utils
from lingvo.tasks.car import input_preprocessors
import numpy as np

FLAGS = tf.flags.FLAGS


class InputPreprocessorsTest(test_utils.TestCase):

  def testIdentityPreprocessor(self):
    input_p = input_preprocessors.ConstantPreprocessor.Params().Set(constants={
        'value1': 1,
        'value2': np.array([2])
    })
    identity_p = input_preprocessors.IdentityPreprocessor.Params()
    features = py_utils.NestedMap()
    shapes = py_utils.NestedMap()
    dtypes = py_utils.NestedMap()

    preprocessors = [input_p.Instantiate(), identity_p.Instantiate()]
    for preprocessor in preprocessors:
      # Verify shape / dtypes.
      shapes = preprocessor.TransformShapes(shapes)
      dtypes = preprocessor.TransformDTypes(dtypes)
      features = preprocessor.TransformFeatures(features)

    self.assertEqual(self.evaluate(features.value1), 1)
    self.assertEqual(shapes.value1, tf.TensorShape([]))
    self.assertEqual(dtypes.value1, tf.int64)

    self.assertEqual(self.evaluate(features.value2), [2])
    self.assertEqual(shapes.value2, tf.TensorShape([1]))
    self.assertEqual(dtypes.value2, tf.int64)

  def testSchedulePreprocessor(self):
    input_p = input_preprocessors.SchedulePreprocessor.Params().Set(
        schedules=py_utils.NestedMap(
            value1=schedule.ConstantOne.Params(),
            value2=schedule.PiecewiseConstantSchedule.Params().Set(
                boundaries=[10], values=[1, 2], dtype=tf.int64),
        ))
    features = py_utils.NestedMap()
    shapes = py_utils.NestedMap()
    dtypes = py_utils.NestedMap()
    # Create global step because schedules depend on it.
    global_step = py_utils.GetOrCreateGlobalStepVar()
    preprocessor = input_p.Instantiate()

    # Verify shape / dtypes.
    shapes = preprocessor.TransformShapes(shapes)
    dtypes = preprocessor.TransformDTypes(dtypes)
    self.assertEqual(shapes.value1, tf.TensorShape([]))
    self.assertEqual(shapes.value2, tf.TensorShape([]))
    self.assertEqual(dtypes.value1, tf.float32)
    self.assertEqual(dtypes.value2, tf.int64)

    features = preprocessor.TransformFeatures(features)

    # Init global step
    self.evaluate(tf.global_variables_initializer())

    features_np = self.evaluate(features)
    self.assertEqual(features_np.value1, 1.)
    self.assertEqual(features_np.value2, 1)  # Global step is 0

    # Set global step to 11.
    self.evaluate(tf.assign(global_step, 11))
    features_np = self.evaluate(features)
    self.assertEqual(features_np.value1, 1.)
    self.assertEqual(features_np.value2, 2)  # Global step is 11

    # Validate all values are schedules.
    input_p = input_preprocessors.SchedulePreprocessor.Params().Set(
        schedules=py_utils.NestedMap(value1=tf.constant(1.),))
    with self.assertRaises(TypeError):
      input_p.Instantiate()

  def testRandomChoicePreprocessor(self):
    p = input_preprocessors.RandomChoicePreprocessor.Params()
    p.weight_tensor_key = 'weights'
    # Construct 4 preprocessors each producing a different value.
    p.subprocessors = [
        input_preprocessors.ConstantPreprocessor.Params().Set(
            constants={'value': 1}),
        input_preprocessors.ConstantPreprocessor.Params().Set(
            constants={'value': 2}),
        input_preprocessors.ConstantPreprocessor.Params().Set(
            constants={'value': 3}),
        input_preprocessors.ConstantPreprocessor.Params().Set(
            constants={'value': 4}),
    ]

    preprocessor = p.Instantiate()

    # Construct test data.
    features = py_utils.NestedMap()
    features.weights = tf.constant([1., 2., 3., 4.])
    shapes = py_utils.NestedMap()
    shapes.weights = tf.TensorShape([4])
    dtypes = py_utils.NestedMap()
    dtypes.weights = tf.float32

    # Verify shape / dtypes.
    new_shapes = preprocessor.TransformShapes(shapes)
    new_dtypes = preprocessor.TransformDTypes(dtypes)
    self.assertEqual(new_shapes.value, tf.TensorShape([]))
    self.assertEqual(new_dtypes.value, tf.int64)

    new_features = preprocessor.TransformFeatures(features)

    counts = [0, 0, 0, 0]
    with self.session() as sess:
      # Run 10000 times to get probability distribution.
      for _ in range(10000):
        new_features_np = sess.run(new_features)
        counts[new_features_np.value - 1] += 1

      # Check distribution roughly matches [0.1, 0.2, 0.3, 0.4]
      self.assertTrue(counts[0] > 800 and counts[0] < 1200)
      self.assertTrue(counts[1] > 1800 and counts[1] < 2200)
      self.assertTrue(counts[2] > 2800 and counts[2] < 3200)
      self.assertTrue(counts[3] > 3800 and counts[3] < 4200)

  def testRandomChoicePreprocessorErrors(self):
    p = input_preprocessors.RandomChoicePreprocessor.Params()
    p.weight_tensor_key = 'weights'
    # Subprocessors produce different shapes
    p.subprocessors = [
        input_preprocessors.ConstantPreprocessor.Params().Set(
            constants={'value': 1}),
        input_preprocessors.ConstantPreprocessor.Params().Set(
            constants={'value': [2, 3]}),
    ]
    preprocessor = p.Instantiate()
    # Construct test data.
    shapes = py_utils.NestedMap()
    shapes.weights = tf.TensorShape([2])
    with self.assertRaises(ValueError):
      preprocessor.TransformShapes(shapes)

    # Subprocessors produce different keys
    p.subprocessors = [
        input_preprocessors.ConstantPreprocessor.Params().Set(
            constants={'value': 1}),
        input_preprocessors.ConstantPreprocessor.Params().Set(
            constants={'foo': 2}),
    ]
    preprocessor = p.Instantiate()
    # Construct test data.
    shapes = py_utils.NestedMap()
    shapes.weights = tf.TensorShape([2])
    with self.assertRaises(ValueError):
      preprocessor.TransformShapes(shapes)

    # Subprocessors produce different dtypes
    p.subprocessors = [
        input_preprocessors.ConstantPreprocessor.Params().Set(
            constants={'value': 1}),
        input_preprocessors.ConstantPreprocessor.Params().Set(
            constants={'value': 2.}),
    ]
    preprocessor = p.Instantiate()
    # Construct test data.
    dtypes = py_utils.NestedMap()
    dtypes.weights = tf.float32
    with self.assertRaises(ValueError):
      preprocessor.TransformDTypes(dtypes)


if __name__ == '__main__':
  tf.test.main()
