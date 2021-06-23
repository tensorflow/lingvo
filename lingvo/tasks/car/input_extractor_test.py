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
"""Tests for input_extractor."""

from lingvo import compat as tf
from lingvo.core import hyperparams
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.tasks.car import base_extractor
from lingvo.tasks.car import input_extractor
from lingvo.tasks.car import input_preprocessors


# Create two extractors that differ in their key names.
class E1(input_extractor.FieldsExtractor):
  KEY_NAME = 'foo'

  def FeatureMap(self):
    return {self.KEY_NAME: tf.io.FixedLenFeature([], dtype=tf.float32)}

  def DType(self):
    return py_utils.NestedMap({self.KEY_NAME: tf.float32})

  def Shape(self):
    return py_utils.NestedMap({self.KEY_NAME: tf.TensorShape([])})

  def _Extract(self, features):
    return py_utils.NestedMap({'foo': features['foo']})


class E2(E1):
  KEY_NAME = 'bar'

  def _Extract(self, features):
    return py_utils.NestedMap({'bar': features['bar']})


class E1WithCheck(E1):

  def _Extract(self, features):
    if 'bar' in features:
      raise ValueError('This should never happen')
    return super()._Extract(features)


class E2WithCheck(E2):

  def _Extract(self, features):
    if 'foo' in features:
      raise ValueError('This should never happen')
    return super()._Extract(features)


class InputExtractorTest(test_utils.TestCase):

  def testNestedFieldsExtractor(self):
    p = input_extractor.NestedFieldsExtractor.Params()
    p.extractors.e1 = E1.Params()
    p.extractors.e2 = E2.Params()
    p.extractors.sub = py_utils.NestedMap(e1=E1.Params(), e2=E2.Params())

    extractors = hyperparams.Params()
    extractors.Define('nested_ext', p, '')

    example = tf.train.Example()
    example.features.feature['foo'].float_list.value[:] = [1.]
    example.features.feature['bar'].float_list.value[:] = [2.]
    serialized = example.SerializeToString()

    ext = base_extractor._BaseExtractor.Params(extractors).Instantiate()
    _, result = ext.ExtractUsingExtractors(serialized)

    self.assertEqual(self.evaluate(result.nested_ext.e1.foo), 1.)
    self.assertEqual(self.evaluate(result.nested_ext.e2.bar), 2.)
    self.assertEqual(self.evaluate(result.nested_ext.sub.e1.foo), 1.)
    self.assertEqual(self.evaluate(result.nested_ext.sub.e2.bar), 2.)

  def testBaseExtractorRaisesErrorWithMissingPreprocessorKeys(self):
    extractors = hyperparams.Params()
    preprocessors = hyperparams.Params()
    preprocessors.Define(
        'count_points',
        input_preprocessors.CountNumberOfPointsInBoxes3D.Params(), '')
    preprocessors.Define('viz_copy',
                         input_preprocessors.CreateDecoderCopy.Params(), '')
    p = input_extractor.BaseExtractor.Params(extractors).Set(
        preprocessors=preprocessors,
        preprocessors_order=['count_points', 'missing_key', 'viz_copy'])
    with self.assertRaisesRegex(
        ValueError,
        r'preprocessor_order specifies keys which were not found .*'):
      p.Instantiate()

  def testExtractorFilters(self):

    class ExampleExtractor(base_extractor._BaseExtractor):

      @classmethod
      def Params(cls):
        extractors = hyperparams.Params()
        extractors.Define('e1', E1WithCheck.Params(), '')
        extractors.Define('e2', E2WithCheck.Params(), '')
        return super().Params(extractors).Set(
            preprocessors=hyperparams.Params(), preprocessors_order=[])

    # Construct record as parseable data.
    example = tf.train.Example()
    example.features.feature['foo'].float_list.value[:] = [1.]
    example.features.feature['bar'].float_list.value[:] = [2.]
    serialized = example.SerializeToString()

    # Extract and verify that the values were extracted.
    extractor = ExampleExtractor.Params().Instantiate()
    _, result = extractor.ExtractUsingExtractors(serialized)
    self.assertEqual(self.evaluate(result.e1.foo), 1.)
    self.assertEqual(self.evaluate(result.e2.bar), 2.)

    # Test that a missing key in input for ProcessFeatures() raises
    # the right extra exception info.
    with self.assertRaisesRegexp(
        RuntimeError, 'Failed running extractor E1WithCheck: KeyError'):
      _, result = extractor.ProcessFeatures({})

  def testBatchedInterface(self):

    class AddTen(input_extractor.FieldsExtractor):

      def FeatureMap(self):
        return {'foo': tf.io.FixedLenFeature([], dtype=tf.float32)}

      def DType(self):
        return py_utils.NestedMap({'foo': tf.float32})

      def Shape(self):
        return py_utils.NestedMap({'foo': tf.TensorShape([])})

      def _Extract(self, features):
        return py_utils.NestedMap({'foo': features['foo'] + 10.})

    fe = AddTen.Params().Instantiate()
    tensor_input = {'foo': tf.constant([1., 2., 3.], shape=(3,))}
    result = fe.ExtractBatch(tensor_input)
    result_np = self.evaluate(result)
    self.assertAllClose([11., 12., 13.], result_np.foo)

    class Add20Batch(AddTen):

      def _ExtractBatch(self, features):
        # A pretend implementation that may be more efficient for batches.
        #
        # Returns a different result to ensure that it is exercised.
        return py_utils.NestedMap(features).Transform(lambda x: x + 20.)

    fe2 = Add20Batch.Params().Instantiate()
    result2 = fe2.ExtractBatch(tensor_input)
    result2_np = self.evaluate(result2)
    self.assertAllClose([21., 22., 23.], result2_np.foo)


if __name__ == '__main__':
  tf.test.main()
