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

  def testRandomChoicePreprocessor(self):
    p = input_preprocessors.RandomChoicePreprocessor.Params()
    # Construct 4 preprocessors each producing a different value.
    base = input_preprocessors.ConstantPreprocessor.Params()
    c1 = (base.Copy().Set(constants={'value': 1}),
          schedule.Constant.Params().Set(value=1))
    c2 = (base.Copy().Set(constants={'value': 2}),
          schedule.Constant.Params().Set(value=2))
    c3 = (base.Copy().Set(constants={'value': 3}),
          schedule.Constant.Params().Set(value=3))
    c4 = (base.Copy().Set(constants={'value': 4}),
          schedule.Constant.Params().Set(value=4))

    p.subprocessors = [c1, c2, c3, c4]

    # Create global step because schedules depend on it.
    _ = py_utils.GetOrCreateGlobalStepVar()
    preprocessor = p.Instantiate()

    features = py_utils.NestedMap()
    shapes = py_utils.NestedMap()
    dtypes = py_utils.NestedMap()

    # Verify shape / dtypes.
    new_shapes = preprocessor.TransformShapes(shapes)
    new_dtypes = preprocessor.TransformDTypes(dtypes)
    self.assertEqual(new_shapes.value, tf.TensorShape([]))
    self.assertEqual(new_dtypes.value, tf.int64)

    self.evaluate(tf.global_variables_initializer())
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
    base = input_preprocessors.ConstantPreprocessor.Params()
    # Subprocessors produce different shapes
    c1 = (base.Copy().Set(constants={'value': 1}),
          schedule.Constant.Params().Set(value=1))
    c2 = (base.Copy().Set(constants={'value': [2, 3]}),
          schedule.Constant.Params().Set(value=2))

    p.subprocessors = [c1, c2]
    preprocessor = p.Instantiate()
    shapes = py_utils.NestedMap()
    with self.assertRaises(ValueError):
      preprocessor.TransformShapes(shapes)

    # Subprocessors produce different keys
    p.subprocessors[1][0].Set(constants={'foo': 2})
    preprocessor = p.Instantiate()
    shapes = py_utils.NestedMap()
    with self.assertRaises(ValueError):
      preprocessor.TransformShapes(shapes)

    # Subprocessors produce different dtypes
    p.subprocessors[1][0].Set(constants={'value': 2.})
    preprocessor = p.Instantiate()
    dtypes = py_utils.NestedMap()
    with self.assertRaises(ValueError):
      preprocessor.TransformDTypes(dtypes)

    # Not a schedule
    p.subprocessors[1] = (p.subprocessors[1][0], tf.constant(1.))
    with self.assertRaises(TypeError):
      preprocessor = p.Instantiate()

  def testSequencePreprocessor(self):
    sub1_p = input_preprocessors.ConstantPreprocessor.Params().Set(
        name='sub1', constants={'foo': 1})
    sub2_p = input_preprocessors.ConstantPreprocessor.Params().Set(
        name='sub2', constants={'bar': 2})
    preprocessor_p = input_preprocessors.Sequence.Params().Set(
        name='list', preprocessors=[sub1_p, sub2_p])

    features = py_utils.NestedMap()
    shapes = py_utils.NestedMap()
    dtypes = py_utils.NestedMap()

    preprocessor = preprocessor_p.Instantiate()
    new_features = preprocessor.TransformFeatures(features)
    new_shapes = preprocessor.TransformShapes(shapes)
    new_dtypes = preprocessor.TransformDTypes(dtypes)

    # Verify shape and dtype
    self.assertEqual(new_shapes.foo, tf.TensorShape([]))
    self.assertEqual(new_shapes.bar, tf.TensorShape([]))

    self.assertEqual(new_dtypes.foo, tf.int64)
    self.assertEqual(new_dtypes.bar, tf.int64)

    with self.session() as sess:
      np_new_features = sess.run(new_features)
      # Check the new constants exist in the features for both preprocessors
      self.assertEqual(np_new_features.foo, 1)
      self.assertEqual(np_new_features.bar, 2)


if __name__ == '__main__':
  tf.test.main()
