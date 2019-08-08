# Lint as: python2, python3
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
"""An experimental new unified TPU executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lingvo import base_runner
from lingvo import compat as tf
from lingvo.core import py_utils

from tensorflow.contrib.tpu.python.tpu import device_assignment as device_assignment_lib


class ExecutorTpu(base_runner.BaseRunner):
  """This is an experimental BaseRunner that does arbitrary multi-program execution on a TPU.

  Overview of operation:

  - During construction, all programs construct their sub-graphs, in a sense
    creating a mega-graph.
  - A sequence of programs is then executed in-whole associated with that task.
    eg: [train x 1000 steps, checkpoint, eval 4 steps, decode 2 steps]
  - In this manner, programs and higher-level tasks cooperatively time-slice
    share the TPU.
  """

  def __init__(self, task_dict, program_schedule_params, model_task_name,
               logdir, tf_master, **kwargs):
    """Construct an ExecutorTpu BaseRunner.

    Args:
      task_dict: A dict of dataset_name -> task params.
      program_schedule_params: A ProgramSchedule params.
      model_task_name: An override for multi-task models, currently unused.
      logdir:  String path to the log directory to output to.
      tf_master: String path to the master job, e.g. 'local'.
      **kwargs: keyword args to pass through to BaseRunner.
    """
    # TODO(blee): fix this.
    train_params = task_dict['Train']
    super(ExecutorTpu, self).__init__(train_params, model_task_name, logdir,
                                      tf_master, **kwargs)

    # There is a single Executor task
    assert self._cluster.num_replicas == 1
    data_parallelism = self._cluster.num_splits_per_client

    assert data_parallelism
    num_devices_per_split = self._cluster.num_devices_per_split
    tf.logging.info('data_parallelism: %d, num_devices_per_split: %d',
                    data_parallelism, num_devices_per_split)

    # Update run-time params
    program_schedule_params.task_dict = task_dict
    program_schedule_params.logdir = logdir
    program_schedule_params.num_splits_per_client = data_parallelism

    self._programs = []
    self._program_schedule = program_schedule_params.Instantiate()

    tf.logging.info('program_schedule_params: %s',
                    program_schedule_params.ToText())

    self._programs += self._program_schedule.Programs()

    # BaseRunner legacy
    self.enqueue_ops = None

    def ComputationShape(split_size):
      """Decides the computation shape based on the split_size."""
      computation_shape = None
      if split_size == 1:
        computation_shape = [1, 1, 1]
      elif split_size == 2:
        computation_shape = [1, 1, 2]
      elif split_size == 4:
        computation_shape = [1, 2, 2]
      elif split_size == 8:
        computation_shape = [2, 2, 2]
      elif split_size == 16:
        computation_shape = [4, 2, 2]
      else:
        assert False, ('Model parallelism with %d devices is currently not'
                       ' supported.' % split_size)
      assert computation_shape is not None
      return computation_shape

    @py_utils.RetryOnTransientTfError()
    def _WaitTillInit():
      """Wait until the model is ready."""
      try:
        with self._GetSession() as sess:
          topology = sess.run(
              tf.tpu.initialize_system(embedding_config=None, job=None))
          device_assignment = device_assignment_lib.device_assignment(
              topology,
              computation_shape=ComputationShape(num_devices_per_split),
              num_replicas=data_parallelism)
          py_utils.SetTpuDeviceAssignment(device_assignment)
          tf.logging.info('device_assignment.core_assignment: %s',
                          str(device_assignment.core_assignment))
          tf.logging.info('device_assignment.topology.device_coordinates: %s',
                          str(device_assignment.topology.device_coordinates))
      except py_utils.transient_tf_errors as e:
        tf.logging.info('TPU initialization failed: %s', e)
        raise

    _WaitTillInit()

    with self._graph.as_default(), tf.container(self._container_id):
      with self._cluster, tf.device(self._cluster.job_spec.name):
        for program in self._programs:
          program.BuildTpuSubgraph()
        self.initialize_tables = tf.tables_initializer()
        self._initialize_local_vars = tf.local_variables_initializer()

  def Start(self):
    # Run training.
    self._RunLoop('executor_tpu', self._Loop)

  def _Loop(self):
    with tf.container(self._container_id), self._GetSession() as sess:
      sess.run(self.initialize_tables)
      sess.run(self._initialize_local_vars)
      sess.run(tf.tpu.initialize_system(embedding_config=None, job=None))
      while True:
        # Single task
        # Get program_schedule associated with single task and run it
        self._program_schedule.Run(sess)

        global_step = sess.run(py_utils.GetGlobalStep())
        if self._ShouldStop(sess, global_step):
          tf.logging.info('Training finished.')
          return

        # TODO(blee): Multi-task.
        # Sample a task
        # Get program schedule associated with sampled task and run it
