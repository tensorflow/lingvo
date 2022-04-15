# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for lingvo-JAX checkpoint_managers."""

import datetime
import os
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from lingvo.jax import checkpoint_managers
from lingvo.jax import checkpoint_pb2
import tensorflow.compat.v2 as tf

CheckpointType = checkpoint_pb2.CheckpointType
FLAGS = flags.FLAGS
CHECKPOINT_PREFIX = checkpoint_managers.CHECKPOINT_PREFIX


def _create_dummy_checkpoint(root_dir, step, checkpoint_type):
  """Creates dummy checkpoint files for a given global_step_id."""
  if checkpoint_type == CheckpointType.CHECKPOINT_FLAX:
    filename = os.path.join(root_dir, f'{CHECKPOINT_PREFIX}{step}')
    with tf.io.gfile.GFile(filename, 'wb') as writer:
      writer.write('')
  elif checkpoint_type == CheckpointType.CHECKPOINT_MULTI_HOST_FLAX:
    for i in range(2):
      process_dir = os.path.join(root_dir, f'{i:03d}')
      tf.io.gfile.makedirs(process_dir)
      filename = os.path.join(process_dir, f'{CHECKPOINT_PREFIX}{step}')
      with tf.io.gfile.GFile(filename, 'wb') as writer:
        writer.write('')
  elif checkpoint_type in {
      CheckpointType.CHECKPOINT_PERSISTENCE,
      CheckpointType.CHECKPOINT_GDA,
  }:
    checkpoint_dir = os.path.join(root_dir, f'{CHECKPOINT_PREFIX}{step:08d}')
    tf.io.gfile.makedirs(checkpoint_dir)
    for f in {'a', 'b'}:
      filename = os.path.join(checkpoint_dir, f)
      with tf.io.gfile.GFile(filename, 'wb') as writer:
        writer.write('')
  else:
    raise ValueError(f'Unsupported checkpoint_type `{checkpoint_type}`.')


def _base_checkpoint_filenames(steps, checkpoint_type):
  """Returns checkpoint basenames corresponding to all the `steps`."""
  if checkpoint_type == CheckpointType.CHECKPOINT_FLAX:
    results = []
    for step in steps:
      results.append(f'{CHECKPOINT_PREFIX}{step}')
    return results
  elif checkpoint_type == CheckpointType.CHECKPOINT_MULTI_HOST_FLAX:
    results = []
    for i in range(2):
      process_dir = f'{i:03d}'
      for step in steps:
        results.append(os.path.join(process_dir, f'{CHECKPOINT_PREFIX}{step}'))
    return results
  elif checkpoint_type in {
      CheckpointType.CHECKPOINT_PERSISTENCE,
      CheckpointType.CHECKPOINT_GDA,
  }:
    results = []
    for step in steps:
      results.append(f'{CHECKPOINT_PREFIX}{step:08d}')
    return results
  else:
    raise ValueError(f'Unsupported checkpoint_type `{checkpoint_type}`.')


def _create_reference_checkpoint_history(config_name, root_dir, checkpoint_type,
                                         steps, checkpoint_datetimes):
  checkpoint_history = checkpoint_pb2.CheckpointHistory(
      config_name=config_name,
      root_directory=root_dir,
      checkpoint_type=checkpoint_type)
  for step, checkpoint_datetime in zip(steps, checkpoint_datetimes):
    timestamp = checkpoint_managers.to_timestamp(checkpoint_datetime)
    checkpoint_history.checkpoints.add(
        global_step_id=step, timestamp_sec=timestamp)
  return checkpoint_history


class CheckpointManagerTest(parameterized.TestCase):

  def assertCheckpointsFileProto(self, checkpoints_filename, expected_proto):
    self.assertTrue(tf.io.gfile.exists(checkpoints_filename))
    checkpoint_history = checkpoint_managers.read_checkpoint_file(
        checkpoints_filename)
    self.assertEqual(checkpoint_history, expected_proto)

  def test_extract_latest_checkpoint_id(self):

    config_name = 'test.test_module.ConfigName'
    root_dir = os.path.join(FLAGS.test_tmpdir, 'test1')
    steps = [100, 300, 700]
    cdt = datetime.datetime.now()
    datetimes = [
        cdt, cdt + datetime.timedelta(hours=1),
        cdt + datetime.timedelta(hours=2)
    ]
    checkpoint_history = _create_reference_checkpoint_history(
        config_name, root_dir, CheckpointType.CHECKPOINT_FLAX, steps, datetimes)
    latest_checkpoint_id = checkpoint_managers.extract_latest_checkpoint_id(
        checkpoint_history)
    self.assertEqual(latest_checkpoint_id, steps[-1])

  @parameterized.named_parameters(
      {
          'testcase_name': 'flax',
          'checkpoint_type': CheckpointType.CHECKPOINT_FLAX
      }, {
          'testcase_name': 'persistence',
          'checkpoint_type': CheckpointType.CHECKPOINT_PERSISTENCE
      }, {
          'testcase_name': 'gda',
          'checkpoint_type': CheckpointType.CHECKPOINT_GDA
      })
  def test_save_no_max_to_keep(self, checkpoint_type):
    config_name = 'test.test_module.ConfigName'
    root_dir = os.path.join(FLAGS.test_tmpdir, 'test2', str(checkpoint_type),
                            'checkpoints')
    tf.io.gfile.makedirs(root_dir)
    current_datetime = datetime.datetime.now()
    zero_datetime = datetime.datetime.fromtimestamp(0)
    with mock.patch('datetime.datetime', autospec=True) as dt:
      dt.utcnow.return_value = current_datetime
      dt.fromtimestamp.return_value = zero_datetime
      checkpoint_manager = checkpoint_managers.CheckpointManager(
          config_name=config_name,
          root_dir=root_dir,
          checkpoint_type=checkpoint_type,
          save_interval_steps=1000,
          max_to_keep=None)
    steps = list(range(0, 10000, 1000))
    checkpoint_datetimes = []
    for step in steps:
      with mock.patch('datetime.datetime', autospec=True) as dt:
        dt.utcnow.return_value = current_datetime
        dt.fromtimestamp.return_value = zero_datetime
        if checkpoint_manager.should_save(step):
          _create_dummy_checkpoint(root_dir, step, checkpoint_type)
          checkpoint_manager.save_metadata(step)
        checkpoint_datetimes.append(current_datetime)
        current_datetime += datetime.timedelta(hours=1)

    saved_steps = steps
    saved_checkpoint_datetimes = checkpoint_datetimes
    filenames = [
        os.path.basename(v) for v in tf.io.gfile.glob(
            os.path.join(root_dir, f'{CHECKPOINT_PREFIX}*'))
    ]
    self.assertSameElements(
        filenames, _base_checkpoint_filenames(saved_steps, checkpoint_type))
    checkpoints_filename = os.path.join(root_dir,
                                        checkpoint_managers.CHECKPOINT_BASENAME)
    expected_proto = _create_reference_checkpoint_history(
        config_name, root_dir, checkpoint_type, saved_steps,
        saved_checkpoint_datetimes)
    self.assertCheckpointsFileProto(checkpoints_filename, expected_proto)

  @parameterized.named_parameters(
      {
          'testcase_name': 'flax',
          'checkpoint_type': CheckpointType.CHECKPOINT_FLAX
      }, {
          'testcase_name': 'persistence',
          'checkpoint_type': CheckpointType.CHECKPOINT_PERSISTENCE
      }, {
          'testcase_name': 'gda',
          'checkpoint_type': CheckpointType.CHECKPOINT_GDA
      })
  def test_save_max_to_keep(self, checkpoint_type):
    config_name = 'test.test_module.ConfigName'
    root_dir = os.path.join(FLAGS.test_tmpdir, 'test3', str(checkpoint_type),
                            'checkpoints')
    tf.io.gfile.makedirs(root_dir)
    current_datetime = datetime.datetime.now()
    zero_datetime = datetime.datetime.fromtimestamp(0)
    with mock.patch('datetime.datetime', autospec=True) as dt:
      dt.utcnow.return_value = current_datetime
      dt.fromtimestamp.return_value = zero_datetime
      checkpoint_manager = checkpoint_managers.CheckpointManager(
          config_name=config_name,
          root_dir=root_dir,
          checkpoint_type=checkpoint_type,
          save_interval_steps=1000,
          max_to_keep=2)
    steps = list(range(0, 10000, 1000))
    checkpoint_datetimes = []
    for step in steps:
      with mock.patch('datetime.datetime', autospec=True) as dt:
        dt.utcnow.return_value = current_datetime
        dt.fromtimestamp.return_value = zero_datetime
        if checkpoint_manager.should_save(step):
          _create_dummy_checkpoint(root_dir, step, checkpoint_type)
          checkpoint_manager.save_metadata(step)
        checkpoint_datetimes.append(current_datetime)
        current_datetime += datetime.timedelta(hours=1)

    saved_steps = [8000, 9000]
    saved_checkpoint_datetimes = checkpoint_datetimes[8:]
    filenames = [
        os.path.basename(v) for v in tf.io.gfile.glob(
            os.path.join(root_dir, f'{CHECKPOINT_PREFIX}*'))
    ]
    self.assertSameElements(
        filenames, _base_checkpoint_filenames(saved_steps, checkpoint_type))
    checkpoints_filename = os.path.join(root_dir,
                                        checkpoint_managers.CHECKPOINT_BASENAME)
    expected_proto = _create_reference_checkpoint_history(
        config_name, root_dir, checkpoint_type, saved_steps,
        saved_checkpoint_datetimes)
    self.assertCheckpointsFileProto(checkpoints_filename, expected_proto)

  @parameterized.named_parameters(
      {
          'testcase_name': 'flax',
          'checkpoint_type': CheckpointType.CHECKPOINT_FLAX
      }, {
          'testcase_name': 'persistence',
          'checkpoint_type': CheckpointType.CHECKPOINT_PERSISTENCE
      }, {
          'testcase_name': 'gda',
          'checkpoint_type': CheckpointType.CHECKPOINT_GDA
      })
  def test_save_checkpoint_keep_interval_timedelta(self, checkpoint_type):
    config_name = 'test.test_module.ConfigName'
    root_dir = os.path.join(FLAGS.test_tmpdir, 'test4', str(checkpoint_type),
                            'checkpoints')
    tf.io.gfile.makedirs(root_dir)
    current_datetime = datetime.datetime.now()
    zero_datetime = datetime.datetime.fromtimestamp(0)
    with mock.patch('datetime.datetime', autospec=True) as dt:
      dt.utcnow.return_value = current_datetime
      dt.fromtimestamp.return_value = zero_datetime
      checkpoint_manager = checkpoint_managers.CheckpointManager(
          config_name=config_name,
          root_dir=root_dir,
          checkpoint_type=checkpoint_type,
          save_interval_steps=1000,
          max_to_keep=2,
          keep_interval_timedelta=datetime.timedelta(hours=2))

    steps = list(range(0, 10000, 1000))
    checkpoint_datetimes = []
    for step in steps:
      with mock.patch('datetime.datetime', autospec=True) as dt:
        dt.utcnow.return_value = current_datetime
        dt.fromtimestamp.return_value = zero_datetime
        if checkpoint_manager.should_save(step):
          _create_dummy_checkpoint(root_dir, step, checkpoint_type)
          checkpoint_manager.save_metadata(step)
        checkpoint_datetimes.append(current_datetime)
        current_datetime += datetime.timedelta(hours=1)

    saved_steps = [0, 2000, 4000, 6000, 8000, 9000]
    saved_checkpoint_datetimes = checkpoint_datetimes[::2] + [
        checkpoint_datetimes[-1]
    ]
    filenames = [
        os.path.basename(v) for v in tf.io.gfile.glob(
            os.path.join(root_dir, f'{CHECKPOINT_PREFIX}*'))
    ]
    self.assertSameElements(
        filenames, _base_checkpoint_filenames(saved_steps, checkpoint_type))
    checkpoints_filename = os.path.join(root_dir,
                                        checkpoint_managers.CHECKPOINT_BASENAME)
    expected_proto = _create_reference_checkpoint_history(
        config_name, root_dir, checkpoint_type, saved_steps,
        saved_checkpoint_datetimes)
    self.assertCheckpointsFileProto(checkpoints_filename, expected_proto)

  @parameterized.named_parameters(
      {
          'testcase_name': 'flax',
          'checkpoint_type': CheckpointType.CHECKPOINT_FLAX
      }, {
          'testcase_name': 'persistence',
          'checkpoint_type': CheckpointType.CHECKPOINT_PERSISTENCE
      }, {
          'testcase_name': 'gda',
          'checkpoint_type': CheckpointType.CHECKPOINT_GDA
      })
  def test_save_restore_manager_case_1_default(self, checkpoint_type):
    config_name = 'test.test_module.ConfigName'
    root_dir = os.path.join(FLAGS.test_tmpdir, 'test5', str(checkpoint_type),
                            'checkpoints')
    tf.io.gfile.makedirs(root_dir)
    current_datetime = datetime.datetime.now()
    zero_datetime = datetime.datetime.fromtimestamp(0)
    with mock.patch('datetime.datetime', autospec=True) as dt:
      dt.utcnow.return_value = current_datetime
      dt.fromtimestamp.return_value = zero_datetime
      checkpoint_manager = checkpoint_managers.CheckpointManager(
          config_name=config_name,
          root_dir=root_dir,
          checkpoint_type=checkpoint_type,
          save_interval_steps=2000,
          max_to_keep=4)

    steps = list(range(0, 10000, 1000))
    checkpoint_datetimes = []
    for step in steps:
      with mock.patch('datetime.datetime', autospec=True) as dt:
        dt.utcnow.return_value = current_datetime
        dt.fromtimestamp.return_value = zero_datetime
        if checkpoint_manager.should_save(step):
          _create_dummy_checkpoint(root_dir, step, checkpoint_type)
          checkpoint_manager.save_metadata(step)
        checkpoint_datetimes.append(current_datetime)
        current_datetime += datetime.timedelta(hours=1)

    saved_steps = [2000, 4000, 6000, 8000]
    saved_checkpoint_datetimes = checkpoint_datetimes[2:10:2]
    filenames = [
        os.path.basename(v) for v in tf.io.gfile.glob(
            os.path.join(root_dir, f'{CHECKPOINT_PREFIX}*'))
    ]
    self.assertSameElements(
        filenames, _base_checkpoint_filenames(saved_steps, checkpoint_type))
    checkpoints_filename = os.path.join(root_dir,
                                        checkpoint_managers.CHECKPOINT_BASENAME)
    expected_proto = _create_reference_checkpoint_history(
        config_name, root_dir, checkpoint_type, saved_steps,
        saved_checkpoint_datetimes)
    self.assertCheckpointsFileProto(checkpoints_filename, expected_proto)

    del checkpoint_manager
    with mock.patch('datetime.datetime', autospec=True) as dt:
      dt.utcnow.return_value = current_datetime
      dt.fromtimestamp.return_value = zero_datetime
      checkpoint_manager = checkpoint_managers.CheckpointManager(
          config_name=config_name,
          root_dir=root_dir,
          checkpoint_type=checkpoint_type,
          save_interval_steps=3000,
          max_to_keep=6,
          keep_interval_timedelta=datetime.timedelta(hours=3))

    saved_steps_2_init = [2000, 4000, 6000, 8000]
    saved_checkpoint_datetimes_2_init = saved_checkpoint_datetimes
    filenames = [
        os.path.basename(v) for v in tf.io.gfile.glob(
            os.path.join(root_dir, f'{CHECKPOINT_PREFIX}*'))
    ]
    self.assertSameElements(
        filenames,
        _base_checkpoint_filenames(saved_steps_2_init, checkpoint_type))
    checkpoints_filename = os.path.join(root_dir,
                                        checkpoint_managers.CHECKPOINT_BASENAME)
    expected_proto = _create_reference_checkpoint_history(
        config_name, root_dir, checkpoint_type, saved_steps_2_init,
        saved_checkpoint_datetimes_2_init)
    self.assertCheckpointsFileProto(checkpoints_filename, expected_proto)

    steps_2 = list(range(10000, 20000, 1000))
    checkpoint_datetimes_2 = []
    for step in steps_2:
      with mock.patch('datetime.datetime', autospec=True) as dt:
        dt.utcnow.return_value = current_datetime
        dt.fromtimestamp.return_value = zero_datetime
        if checkpoint_manager.should_save(step):
          _create_dummy_checkpoint(root_dir, step, checkpoint_type)
          checkpoint_manager.save_metadata(step)
        checkpoint_datetimes_2.append(current_datetime)
        current_datetime += datetime.timedelta(hours=1)

    saved_steps_2 = saved_steps_2_init + [11000, 14000, 17000]
    saved_checkpoint_datetimes_2 = (
        saved_checkpoint_datetimes_2_init + checkpoint_datetimes_2[1:10:3])
    filenames = [
        os.path.basename(v) for v in tf.io.gfile.glob(
            os.path.join(root_dir, f'{CHECKPOINT_PREFIX}*'))
    ]
    self.assertSameElements(
        filenames, _base_checkpoint_filenames(saved_steps_2, checkpoint_type))
    checkpoints_filename = os.path.join(root_dir,
                                        checkpoint_managers.CHECKPOINT_BASENAME)
    expected_proto = _create_reference_checkpoint_history(
        config_name, root_dir, checkpoint_type, saved_steps_2,
        saved_checkpoint_datetimes_2)
    self.assertCheckpointsFileProto(checkpoints_filename, expected_proto)

  @parameterized.named_parameters(
      {
          'testcase_name': 'flax',
          'checkpoint_type': CheckpointType.CHECKPOINT_FLAX
      }, {
          'testcase_name': 'persistence',
          'checkpoint_type': CheckpointType.CHECKPOINT_PERSISTENCE
      }, {
          'testcase_name': 'gda',
          'checkpoint_type': CheckpointType.CHECKPOINT_GDA
      })
  def test_save_restore_manager_case_2_mutant(self, checkpoint_type):
    config_name = 'test.test_module.ConfigName'
    root_dir = os.path.join(FLAGS.test_tmpdir, 'test6', str(checkpoint_type),
                            'checkpoints')
    tf.io.gfile.makedirs(root_dir)
    current_datetime = datetime.datetime.now()
    zero_datetime = datetime.datetime.fromtimestamp(0)
    with mock.patch('datetime.datetime', autospec=True) as dt:
      dt.utcnow.return_value = current_datetime
      dt.fromtimestamp.return_value = zero_datetime
      checkpoint_manager = checkpoint_managers.CheckpointManager(
          config_name=config_name,
          root_dir=root_dir,
          checkpoint_type=checkpoint_type,
          save_interval_steps=100,
          max_to_keep=None)

    steps = list(range(0, 10000, 1000))
    checkpoint_datetimes = []
    for step in steps:
      with mock.patch('datetime.datetime', autospec=True) as dt:
        dt.utcnow.return_value = current_datetime
        dt.fromtimestamp.return_value = zero_datetime
        if checkpoint_manager.should_save(step):
          _create_dummy_checkpoint(root_dir, step, checkpoint_type)
          checkpoint_manager.save_metadata(step)
        checkpoint_datetimes.append(current_datetime)
        current_datetime += datetime.timedelta(hours=1)

    saved_steps = steps
    saved_checkpoint_datetimes = checkpoint_datetimes
    filenames = [
        os.path.basename(v) for v in tf.io.gfile.glob(
            os.path.join(root_dir, f'{CHECKPOINT_PREFIX}*'))
    ]
    self.assertSameElements(
        filenames, _base_checkpoint_filenames(saved_steps, checkpoint_type))
    checkpoints_filename = os.path.join(root_dir,
                                        checkpoint_managers.CHECKPOINT_BASENAME)
    expected_proto = _create_reference_checkpoint_history(
        config_name, root_dir, checkpoint_type, saved_steps,
        saved_checkpoint_datetimes)
    self.assertCheckpointsFileProto(checkpoints_filename, expected_proto)

    del checkpoint_manager
    max_to_keep = 5
    with mock.patch('datetime.datetime', autospec=True) as dt:
      dt.utcnow.return_value = current_datetime
      dt.fromtimestamp.return_value = zero_datetime
      checkpoint_manager = checkpoint_managers.CheckpointManager(
          config_name=config_name,
          root_dir=root_dir,
          checkpoint_type=checkpoint_type,
          save_interval_steps=1000,
          max_to_keep=max_to_keep)

    step = 10000
    steps.append(step)
    with mock.patch('datetime.datetime', autospec=True) as dt:
      dt.utcnow.return_value = current_datetime
      dt.fromtimestamp.return_value = zero_datetime
      if checkpoint_manager.should_save(step):
        _create_dummy_checkpoint(root_dir, step, checkpoint_type)
        checkpoint_manager.save_metadata(step)
      saved_checkpoint_datetimes.append(current_datetime)
      current_datetime += datetime.timedelta(hours=1)

    saved_steps_2 = steps[-max_to_keep:]
    saved_checkpoint_datetimes_2 = saved_checkpoint_datetimes[-max_to_keep:]
    filenames = [
        os.path.basename(v) for v in tf.io.gfile.glob(
            os.path.join(root_dir, f'{CHECKPOINT_PREFIX}*'))
    ]
    self.assertSameElements(
        filenames, _base_checkpoint_filenames(saved_steps_2, checkpoint_type))
    checkpoints_filename = os.path.join(root_dir,
                                        checkpoint_managers.CHECKPOINT_BASENAME)
    expected_proto = _create_reference_checkpoint_history(
        config_name, root_dir, checkpoint_type, saved_steps_2,
        saved_checkpoint_datetimes_2)
    self.assertCheckpointsFileProto(checkpoints_filename, expected_proto)


if __name__ == '__main__':
  absltest.main()
