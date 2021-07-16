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
"""Tests for base_input_generator."""

import contextlib
import copy
import os
import shutil
import tempfile
from absl.testing import flagsaver
import lingvo.compat as tf
from lingvo.core import base_input_generator
from lingvo.core import cluster_factory
from lingvo.core import datasource
from lingvo.core import hyperparams
from lingvo.core import py_utils
from lingvo.core import test_utils
import mock
import numpy as np

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tpu import device_assignment
# pylint: enable=g-direct-tensorflow-import


def _CreateFakeTFRecordFiles(record_count=10):
  tmpdir = tempfile.mkdtemp()
  data_path = os.path.join(tmpdir, 'fake.tfrecord')
  with tf.io.TFRecordWriter(data_path) as w:
    for _ in range(record_count):
      feature = {
          'audio':
              tf.train.Feature(
                  float_list=tf.train.FloatList(
                      value=np.random.uniform(-1.0, 1.0, 48000))),
      }
      example = tf.train.Example(features=tf.train.Features(feature=feature))
      w.write(example.SerializeToString())
  return tmpdir, data_path


class BaseInputGeneratorTest(test_utils.TestCase):

  @flagsaver.flagsaver(xla_device='tpu', enable_asserts=False)
  def testBatchSizeSingleHostInfeed(self):
    with cluster_factory.ForTestingWorker(tpus=128):
      p = base_input_generator.BaseInputGenerator.Params()
      p.batch_size = 16
      p.use_per_host_infeed = False
      input_generator = p.Instantiate()

      self.assertEqual(2048, input_generator.InfeedBatchSize())
      self.assertEqual(2048, input_generator.GlobalBatchSize())

  @flagsaver.flagsaver(xla_device='tpu', enable_asserts=False)
  def testBatchSizePerHostInfeed(self):
    with cluster_factory.ForTestingWorker(tpus=128, num_tpu_hosts=8):
      p = base_input_generator.BaseInputGenerator.Params()
      p.batch_size = 16
      p.use_per_host_infeed = True
      input_generator = p.Instantiate()

      self.assertEqual(256, input_generator.InfeedBatchSize())
      self.assertEqual(2048, input_generator.GlobalBatchSize())

  @contextlib.contextmanager
  def _DeviceAssignment(self):
    """A context for tpu device assignment of a JF 8x8 slice."""
    mesh_shape = [8, 8, 1, 2]
    device_coordinates = np.zeros([16, 8, 4], dtype=np.int32)
    for i in range(np.prod(mesh_shape)):
      x = i // 16
      y = i % 16 // 2
      core = i % 2
      task = x // 2 * 4 + y // 2
      device = x % 2 * 4 + y % 2 * 2 + core
      device_coordinates[task, device] = [x, y, 0, core]
    topology = tf.tpu.experimental.Topology(
        mesh_shape=mesh_shape, device_coordinates=device_coordinates)
    assignment = device_assignment.device_assignment(
        topology, computation_shape=[1, 1, 1, 1], num_replicas=128)
    py_utils.SetTpuDeviceAssignment(assignment)
    try:
      yield
    finally:
      py_utils.SetTpuDeviceAssignment(None)

  @flagsaver.flagsaver(xla_device='tpu', enable_asserts=False)
  def testCreateTpuEnqueueOpsSingleHostInfeed(self):

    class FooInputGenerator(base_input_generator.BaseInputGenerator):

      def _InputBatch(self):
        return py_utils.NestedMap(
            inp=tf.constant(1.0, shape=[2048, 3], dtype=tf.float32))

    with cluster_factory.ForTestingWorker(
        tpus=128, num_tpu_hosts=16, add_summary=True):
      with self._DeviceAssignment():
        p = FooInputGenerator.Params()
        p.use_per_host_infeed = False
        input_generator = p.Instantiate()
        input_generator.CreateTpuEnqueueOps()
        batch = input_generator.TpuDequeueBatch()
        self.assertEqual(batch.inp.shape.as_list(), [16, 3])

  @flagsaver.flagsaver(xla_device='tpu', enable_asserts=False)
  def testCreateTpuEnqueueOpsPerHostInfeed(self):

    class FooInputGenerator(base_input_generator.BaseInputGenerator):

      def _InputBatch(self):
        return py_utils.NestedMap(
            inp=tf.constant(1.0, shape=[128, 3], dtype=tf.float32))

    with cluster_factory.ForTestingWorker(tpus=128, num_tpu_hosts=16):
      with self._DeviceAssignment():
        p = FooInputGenerator.Params()
        p.use_per_host_infeed = True
        input_generator = p.Instantiate()
        input_generator.CreateTpuEnqueueOps()
        batch = input_generator.TpuDequeueBatch()
        self.assertEqual(batch.inp.shape.as_list(), [16, 3])

  @flagsaver.flagsaver(xla_device='tpu', enable_asserts=False)
  def testCreateTpuEnqueueOpsPerHostInfeed_Sharded(self):

    class FooInputGenerator(base_input_generator.BaseInputGenerator):

      def _InputBatch(self):
        return [
            py_utils.NestedMap(
                inp=tf.constant(1.0, shape=[16, 3], dtype=tf.float32))
            for _ in range(8)
        ]

    with cluster_factory.ForTestingWorker(tpus=128, num_tpu_hosts=16):
      with self._DeviceAssignment():
        p = FooInputGenerator.Params()
        p.use_per_host_infeed = True
        input_generator = p.Instantiate()
        input_generator.CreateTpuEnqueueOps()
        batch = input_generator.TpuDequeueBatch()
        self.assertEqual(batch.inp.shape.as_list(), [16, 3])

  def testGetPreprocessedBatchWithDatasource(self):

    class TestDataset(datasource.TFDatasetSource):

      def GetDataset(self):
        return tf.data.Dataset.from_tensors(0)

    with self.subTest('AllowedWithNoOverrides'):
      p = base_input_generator.BaseInputGenerator.Params()
      p.file_datasource = TestDataset.Params()
      p.Instantiate().GetPreprocessedInputBatch()

    with self.subTest('AllowedWithBaseInputGeneratorFromFiles'):
      p = base_input_generator.BaseInputGeneratorFromFiles.Params()
      p.file_datasource = TestDataset.Params()
      p.Instantiate().GetPreprocessedInputBatch()

    msg = 'Batches obtained through p.file_datasource'

    with self.subTest('DisallowedWhenOverridingInputBatch'):

      class OverrideInputBatch(base_input_generator.BaseInputGenerator):

        def _InputBatch(self):
          return 0

      p = OverrideInputBatch.Params()
      p.file_datasource = TestDataset.Params()
      with self.assertRaisesRegex(ValueError, msg):
        p.Instantiate().GetPreprocessedInputBatch()

    with self.subTest('DisallowedWhenOverridingPreprocessInputBatch'):

      class OverridePreprocessInputBatch(base_input_generator.BaseInputGenerator
                                        ):

        def _PreprocessInputBatch(self, batch):
          return batch

      p = OverridePreprocessInputBatch.Params()
      p.file_datasource = TestDataset.Params()
      with self.assertRaisesRegex(ValueError, msg):
        p.Instantiate().GetPreprocessedInputBatch()

    with self.subTest('DisallowedWithTrainerInputReplicas'):

      def WithInputTargets():
        ret = copy.deepcopy(cluster_factory.Current())
        ret.params.input.targets = 'a,b'
        ret.params.input.replicas = 2
        return ret

      p = base_input_generator.BaseInputGenerator.Params()
      p.file_datasource = TestDataset.Params()
      msg = 'TFDatasetSource subclassed DataSources do not support'
      with WithInputTargets(), self.assertRaisesRegex(ValueError, msg):
        p.Instantiate().GetPreprocessedInputBatch()

  def testFilePatternToDataSourceNoMixingSourceIdOffset(self):
    p = base_input_generator.BaseInputGeneratorFromFiles.Params()
    p.all_zero_source_id_without_within_batch_mixing = False

    p.file_pattern = [
        ('fp1', 1),
        ('fp2', 1, 'arg3'),
        ('fp3', 1),
    ]
    p.use_within_batch_mixing = False
    ds_params = base_input_generator.FilePatternToDataSource(p)
    sub_params = ds_params.sub

    self.assertEqual(sub_params[0].source_id_offset, 0)
    self.assertEqual(sub_params[1].source_id_offset, 1)
    self.assertEqual(sub_params[2].source_id_offset, 2)


class ToyInputGenerator(base_input_generator.BaseDataExampleInputGenerator):

  def GetFeatureSpec(self):
    return {'audio': tf.io.FixedLenFeature([48000], tf.float32)}


class BaseExampleInputGeneratorTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    tf.reset_default_graph()

  def tearDown(self):
    super().tearDown()
    if hasattr(self, '_tmpdir'):
      shutil.rmtree(self._tmpdir)

  def testTfRecordFile(self):
    p = ToyInputGenerator.Params()
    p.batch_size = 2
    self._tmpdir, p.input_files = _CreateFakeTFRecordFiles()
    p.dataset_type = tf.data.TFRecordDataset
    p.randomize_order = False
    p.parallel_readers = 1
    ig = p.Instantiate()
    with self.session():
      inputs = ig.GetPreprocessedInputBatch()
      eval_inputs = self.evaluate(inputs)
      input_shapes = eval_inputs.Transform(lambda t: t.shape)
      self.assertEqual(input_shapes.audio, (2, 48000))

  def testTfRecordFileLargeBatch(self):
    p = ToyInputGenerator.Params()
    p.batch_size = 200
    self._tmpdir, p.input_files = _CreateFakeTFRecordFiles()
    p.dataset_type = tf.data.TFRecordDataset
    p.randomize_order = False
    p.parallel_readers = 1
    ig = p.Instantiate()
    with self.session():
      inputs = ig.GetPreprocessedInputBatch()
      eval_inputs = self.evaluate(inputs)
      input_shapes = eval_inputs.Transform(lambda t: t.shape)
      self.assertEqual(input_shapes.audio, (200, 48000))

  def testNumEpochs(self):
    p = ToyInputGenerator.Params()
    p.batch_size = 3
    p.num_epochs = 7
    self._tmpdir, p.input_files = _CreateFakeTFRecordFiles(
        record_count=p.batch_size)
    p.dataset_type = tf.data.TFRecordDataset
    p.randomize_order = False
    p.parallel_readers = 1
    ig = p.Instantiate()
    with self.session():
      inputs = ig.GetPreprocessedInputBatch()
      for _ in range(p.num_epochs):
        eval_inputs = self.evaluate(inputs)
        self.assertEqual(eval_inputs.audio.shape, (p.batch_size, 48000))
      with self.assertRaisesRegex(tf.errors.OutOfRangeError, 'End of sequence'):
        self.evaluate(inputs)

  def testRespectsInfeedBatchSize(self):
    p = ToyInputGenerator.Params()
    p.batch_size = 3
    self._tmpdir, p.input_files = _CreateFakeTFRecordFiles()
    p.dataset_type = tf.data.TFRecordDataset

    ig = p.Instantiate()
    batch = ig.GetPreprocessedInputBatch()
    self.assertEqual(batch.audio.shape[0], p.batch_size)
    self.assertEqual(p.batch_size, ig.InfeedBatchSize())

    tf.reset_default_graph()
    ig = p.Instantiate()
    with mock.patch.object(
        ig, 'InfeedBatchSize', return_value=42) as mock_method:
      batch = ig.GetPreprocessedInputBatch()
      self.assertEqual(batch.audio.shape[0], 42)
    mock_method.assert_called()


# Dataset pipelines for TFDataInputTest.
def _TestDatasetFn(begin=0, end=10):
  """Test tf.data pipeline."""
  ds = tf.data.Dataset.from_tensor_slices(tf.range(begin, end))
  return ds.map(lambda x: {'value': x})


def _TestDatasetFnWithoutDefault(begin, end=10):
  """Test tf.data pipeline with non-defaulted parameters."""
  ds = tf.data.Dataset.from_tensor_slices(tf.range(begin, end))
  return ds.map(lambda x: {'value': x})


def _TestDatasetFnWithRepeat(begin=0, end=10):
  """Test tf.data pipeline with repeat."""
  ds = tf.data.Dataset.from_tensor_slices(tf.range(begin, end)).repeat()
  return ds.map(lambda x: {'value': x})


def _TestDatasetFnV1(begin=0, end=10):
  """Similar to _TestDatasetFn but returns TFv1's dataset explicitly."""
  ds = tf.tf1.data.Dataset.from_tensor_slices(tf.range(begin, end))
  return ds.map(lambda x: {'value': x})


def _TestDatasetFnV2(begin=0, end=10):
  """Similar to _TestDatasetFn but returns TFv2's dataset explicitly."""
  ds = tf.tf2.data.Dataset.from_tensor_slices(tf.range(begin, end))
  return ds.map(lambda x: {'value': x})


class _TestDatasetClass:
  """A class that generates tf.data by its member function."""

  def __init__(self, begin):
    self._begin = begin

  def DatasetFn(self, end=10):
    ds = tf.data.Dataset.from_tensor_slices(tf.range(self._begin, end))
    return ds.map(lambda x: {'value': x})


# A class object which will be instantiated at importing the module.
# It can be used in DefineTFDataInput().
_TestDatasetObject = _TestDatasetClass(begin=0)

# InputGenerators for TFDataInputTest.
_TestTFDataInput = base_input_generator.DefineTFDataInput(
    '_TestTFDataInput', _TestDatasetFn)
_TestTFDataInputWithIgnoreArgs = base_input_generator.DefineTFDataInput(
    '_TestTFDataInputWithIgnoreArgs', _TestDatasetFn, ignore_args=('begin',))
_TestTFDataInputWithMapArgs = base_input_generator.DefineTFDataInput(
    '_TestTFDataInputWithMapArgs',
    _TestDatasetFn,
    map_args={'end': 'num_samples'})
_TestTFDataInputWithoutDefault = base_input_generator.DefineTFDataInput(
    '_TestTFDataInputWithoutDefault', _TestDatasetFnWithoutDefault)
_TestTFDataInputWithRepeat = base_input_generator.DefineTFDataInput(
    '_TestTFDataInputWithRepeat', _TestDatasetFnWithRepeat)
_TestTFDataInputWithBoundMethod = base_input_generator.DefineTFDataInput(
    '_TestTFDataInputWithBoundMethod', _TestDatasetObject.DatasetFn)
_TestTFDataInputV1 = base_input_generator.DefineTFDataInput(
    '_TestTFDataInputV1', _TestDatasetFnV1)
_TestTFDataInputV2 = base_input_generator.DefineTFDataInput(
    '_TestTFDataInputV2', _TestDatasetFnV2)


class TFDataInputTest(test_utils.TestCase):

  def testModule(self):
    self.assertEqual(_TestTFDataInput.__module__, '__main__')
    self.assertEqual(_TestTFDataInputWithIgnoreArgs.__module__, '__main__')
    self.assertEqual(_TestTFDataInputWithMapArgs.__module__, '__main__')
    self.assertEqual(_TestTFDataInputWithoutDefault.__module__, '__main__')
    self.assertEqual(_TestTFDataInputWithRepeat.__module__, '__main__')
    self.assertEqual(_TestTFDataInputWithBoundMethod.__module__, '__main__')
    self.assertEqual(_TestTFDataInputV1.__module__, '__main__')
    self.assertEqual(_TestTFDataInputV2.__module__, '__main__')

  def testExample(self):
    """Tests the example code in the function docstring."""
    p = _TestTFDataInput.Params()
    self.assertIn('args', p)
    self.assertIn('begin', p.args)
    self.assertIn('end', p.args)
    self.assertEqual(p.args.begin, 0)
    self.assertEqual(p.args.end, 10)

    ig = p.Instantiate()
    self.assertIsInstance(ig, _TestTFDataInput)

    with self.session() as sess:
      data = ig.GetPreprocessedInputBatch()
      self.assertIsInstance(data, py_utils.NestedMap)
      self.assertIsInstance(data.value, tf.Tensor)
      self.assertAllEqual(data.value.shape, ())
      self.assertEqual(data.value.dtype, tf.int32)

      # Consumes all data.
      for i in range(p.args.begin, p.args.end):
        self.assertEqual(sess.run(data).value, i)

      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(data)

  def testToFromProto(self):
    """Similar to `testExample` but params will be restored from a proto."""
    serialized_proto = _TestTFDataInput.Params().ToProto()
    p = hyperparams.Params.FromProto(serialized_proto)
    self.assertIn('args', p)
    self.assertIn('begin', p.args)
    self.assertIn('end', p.args)
    self.assertEqual(p.args.begin, 0)
    self.assertEqual(p.args.end, 10)

    ig = p.Instantiate()
    self.assertIsInstance(ig, _TestTFDataInput)

    with self.session() as sess:
      data = ig.GetPreprocessedInputBatch()
      self.assertIsInstance(data, py_utils.NestedMap)
      self.assertIsInstance(data.value, tf.Tensor)
      self.assertAllEqual(data.value.shape, ())
      self.assertEqual(data.value.dtype, tf.int32)

      # Consumes all data.
      for i in range(p.args.begin, p.args.end):
        self.assertEqual(sess.run(data).value, i)

      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(data)

  def testWithIgnoreArgs(self):
    """Tests the `ignore_args` parameter."""
    p = _TestTFDataInputWithIgnoreArgs.Params()
    self.assertIn('args', p)
    self.assertNotIn('begin', p.args)
    self.assertIn('end', p.args)
    self.assertEqual(p.args.end, 10)

    ig = p.Instantiate()
    self.assertIsInstance(ig, _TestTFDataInputWithIgnoreArgs)

    with self.session() as sess:
      data = ig.GetPreprocessedInputBatch()
      self.assertIsInstance(data, py_utils.NestedMap)
      self.assertIsInstance(data.value, tf.Tensor)
      self.assertAllEqual(data.value.shape, ())
      self.assertEqual(data.value.dtype, tf.int32)

      # Consumes all data.
      for i in range(p.args.end):
        self.assertEqual(sess.run(data).value, i)

      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(data)

  def testWithMapArgs(self):
    """Tests the `map_args` parameter."""
    p = _TestTFDataInputWithMapArgs.Params()
    self.assertIn('args', p)
    self.assertIn('num_samples', p)  # Defined by BaseInputGenerator.
    self.assertIn('begin', p.args)
    self.assertNotIn('end', p.args)
    self.assertEqual(p.num_samples, 0)
    self.assertEqual(p.args.begin, 0)

    p.num_samples = 20
    ig = p.Instantiate()
    self.assertIsInstance(ig, _TestTFDataInputWithMapArgs)

    with self.session() as sess:
      data = ig.GetPreprocessedInputBatch()
      self.assertIsInstance(data, py_utils.NestedMap)
      self.assertIsInstance(data.value, tf.Tensor)
      self.assertAllEqual(data.value.shape, ())
      self.assertEqual(data.value.dtype, tf.int32)

      # Consumes all data.
      for i in range(p.args.begin, p.num_samples):
        self.assertEqual(sess.run(data).value, i)

      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(data)

  def testWithoutDefault(self):
    """Tests parameters without defaults."""
    p = _TestTFDataInputWithoutDefault.Params()
    self.assertIn('args', p)
    self.assertIn('begin', p.args)
    self.assertIn('end', p.args)
    self.assertIsNone(p.args.begin)
    self.assertEqual(p.args.end, 10)

    p.args.begin = 0
    ig = p.Instantiate()
    self.assertIsInstance(ig, _TestTFDataInputWithoutDefault)

    with self.session() as sess:
      data = ig.GetPreprocessedInputBatch()
      self.assertIsInstance(data, py_utils.NestedMap)
      self.assertIsInstance(data.value, tf.Tensor)
      self.assertAllEqual(data.value.shape, ())
      self.assertEqual(data.value.dtype, tf.int32)

      # Consumes all data.
      for i in range(p.args.begin, p.args.end):
        self.assertEqual(sess.run(data).value, i)

      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(data)

  def testWithRepeat(self):
    """Tests if the repeated dataset runs forever."""
    p = _TestTFDataInputWithRepeat.Params()
    self.assertIn('args', p)
    self.assertIn('begin', p.args)
    self.assertIn('end', p.args)
    self.assertEqual(p.args.begin, 0)
    self.assertEqual(p.args.end, 10)

    ig = p.Instantiate()
    self.assertIsInstance(ig, _TestTFDataInputWithRepeat)

    with self.session() as sess:
      data = ig.GetPreprocessedInputBatch()
      self.assertIsInstance(data, py_utils.NestedMap)
      self.assertIsInstance(data.value, tf.Tensor)
      self.assertAllEqual(data.value.shape, ())
      self.assertEqual(data.value.dtype, tf.int32)

      # Runs the dataset several times: it should not raise OutOfRangeError.
      for _ in range(3):
        for i in range(p.args.begin, p.args.end):
          self.assertEqual(sess.run(data).value, i)

  def testWithBoundMethod(self):
    """Tests pipeline defined by a bound method: member function with self."""
    p = _TestTFDataInputWithBoundMethod.Params()
    self.assertIn('args', p)
    self.assertNotIn('begin', p.args)
    self.assertIn('end', p.args)
    self.assertEqual(p.args.end, 10)

    ig = p.Instantiate()

    self.assertIsInstance(ig, _TestTFDataInputWithBoundMethod)
    with self.session() as sess:
      data = ig.GetPreprocessedInputBatch()
      self.assertIsInstance(data, py_utils.NestedMap)
      self.assertIsInstance(data.value, tf.Tensor)
      self.assertAllEqual(data.value.shape, ())
      self.assertEqual(data.value.dtype, tf.int32)

      # Consumes all data.
      for i in range(p.args.end):
        self.assertEqual(sess.run(data).value, i)

      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(data)

  def testDatasetV1(self):
    """Tests the TFv1 Dataset."""
    p = _TestTFDataInputV1.Params()
    self.assertIn('args', p)
    self.assertIn('begin', p.args)
    self.assertIn('end', p.args)
    self.assertEqual(p.args.begin, 0)
    self.assertEqual(p.args.end, 10)

    ig = p.Instantiate()
    self.assertIsInstance(ig, _TestTFDataInputV1)

    with self.session() as sess:
      data = ig.GetPreprocessedInputBatch()
      self.assertIsInstance(data, py_utils.NestedMap)
      self.assertIsInstance(data.value, tf.Tensor)
      self.assertAllEqual(data.value.shape, ())
      self.assertEqual(data.value.dtype, tf.int32)

      # Consumes all data.
      for i in range(p.args.begin, p.args.end):
        self.assertEqual(sess.run(data).value, i)

      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(data)

  def testDatasetV2(self):
    """Tests the TFv2 Dataset."""
    p = _TestTFDataInputV2.Params()
    self.assertIn('args', p)
    self.assertIn('begin', p.args)
    self.assertIn('end', p.args)
    self.assertEqual(p.args.begin, 0)
    self.assertEqual(p.args.end, 10)

    ig = p.Instantiate()
    self.assertIsInstance(ig, _TestTFDataInputV2)

    # We keep the TFv1's Session here since v1/v2 behaviors would not coexist.
    # TODO(oday): write TFv2-specific tests.
    with self.session() as sess:
      data = ig.GetPreprocessedInputBatch()
      self.assertIsInstance(data, py_utils.NestedMap)
      self.assertIsInstance(data.value, tf.Tensor)
      self.assertAllEqual(data.value.shape, ())
      self.assertEqual(data.value.dtype, tf.int32)

      # Consumes all data.
      for i in range(p.args.begin, p.args.end):
        self.assertEqual(sess.run(data).value, i)

      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(data)


if __name__ == '__main__':
  tf.test.main()
