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
"""Specification of a training cluster."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import heapq
import threading

import numpy as np
from six.moves import range
import tensorflow as tf

from lingvo.core import hyperparams
from lingvo.core import py_utils


class _LocalClusterStack(threading.local):

  def __init__(self):
    super(_LocalClusterStack, self).__init__()
    self.stack = []


_CLUSTER_STACK = _LocalClusterStack()


class _Cluster(object):
  """The whole training cluster from a single task's point of view."""

  @classmethod
  def _JobSpec(cls, replicas):
    """Construct a job spec param with the given number of replicas."""
    p = hyperparams.Params()
    # By default, we use /job:localhost so that most of tests can just
    # work out of the box. trainer.py will then set job names accordingly.
    p.Define('name', '/job:localhost',
             'TensorFlow job spec, e.g., /job:trainer, /job:ps')
    p.Define('replicas', replicas, 'The number of tasks of a job.')
    p.Define('cpus_per_replica', 1, 'The number of CPU devices to use per '
             'replica.')
    p.Define('gpus_per_replica', 0, 'The number of GPU devices to use per '
             'replica.')
    p.Define(
        'devices_per_split', 1, 'Devices of a replica are grouped into '
        'splits. Each split contains these many devices. One split is a '
        'group of devices on which the computation nodes of a graph is '
        'placed upon.E.g., one can place the forward lstm on device 0 of '
        'a split and place the backward lstm on device 1. etc.')
    p.Define('tpus_per_replica', 0,
             'The number of tpu cores to use per replica.')
    p.Define('num_tpu_hosts', 0, 'The number of tpu hosts.')
    return p

  @classmethod
  def Params(cls):
    """Defaults parameters for a cluster."""
    p = hyperparams.Params()
    p.Define('cls', cls, 'The class that this param is associated with.')
    p.Define(
        'mode', 'async', 'A string noting the overall training method. '
        'Valid values: sync, async.')
    p.Define(
        'job', 'trainer', 'The role of this job in the training cluster. '
        'E.g., trainer_client, trainer, controller,  etc.')
    p.Define('task', 0, 'This process is the task-th task in the job.')

    # How the cluster is composed.
    #
    # A typical training cluster has a few jobs (controller, worker, ps, etc).
    # One can potentially place computation on any device of these jobs.
    # Here, we specify how each job is configured. E.g., number of GPUs each
    # task is equipped with, the number of replicas, etc.
    #
    # Note that trainer client may dispatch operations on just a
    # smaller subset of jobs. For example, the controller only places
    # computations onto the controller and ps devices; while evaler
    # only places computations on the evaler devices.
    #
    # cluster.job refers to the role of a client process performs.  It
    # can be 'controller', 'trainer', 'trainer_client', 'evaler' and
    # 'decoder', etc. Often, a client can be the same process as one
    # of the compute devices (e.g., controller). Sometimes, they can
    # be a separate processes. E.g., trainer_client is a separate
    # standalone process. It places computations on the worker and
    # ps devices, while itself does not host any.
    p.Define('controller', cls._JobSpec(1), 'The controller job.')
    p.Define('worker', cls._JobSpec(1), 'The worker job.')
    p.Define('ps', cls._JobSpec(1), 'The ps job.')
    p.Define('input', cls._JobSpec(0), 'The input job.')
    p.Define('evaler', cls._JobSpec(0), 'The evaler job.')
    p.Define('decoder', cls._JobSpec(0), 'The decoder job.')

    # A few 'global' knobs.
    p.Define(
        'add_summary', None, 'Whether to add summaries. If None, '
        'decides based on the job type.')
    return p

  @classmethod
  def _MakeDeviceString(_, job_name, task_id, device_name, device_id):
    return '%s/replica:0/task:%d/device:%s:%d' % (job_name, task_id,
                                                  device_name, device_id)

  @classmethod
  def ListDevices(cls, job_spec):
    """Lists devices in the job.

    Args:
      job_spec: A param object specifying a job in a training cluster.

    Returns:
      Returns a 2D np string array. ret[i, j] is the i-th replica's j-th
      devices.
    """
    if not job_spec.gpus_per_replica:
      cpus = job_spec.cpus_per_replica
      ret = np.empty((job_spec.replicas, cpus), np.object)
      for i in range(job_spec.replicas):
        for j in range(cpus):
          ret[i, j] = cls._MakeDeviceString(job_spec.name, i, 'CPU', j)
    else:
      ret = np.empty((job_spec.replicas, job_spec.gpus_per_replica), np.object)
      for i in range(job_spec.replicas):
        for j in range(job_spec.gpus_per_replica):
          ret[i, j] = cls._MakeDeviceString(job_spec.name, i, 'GPU', j)
    return ret

  @staticmethod
  def _cluster_stack():
    return _CLUSTER_STACK

  def __enter__(self):
    _CLUSTER_STACK.stack.append(self)

  def __exit__(self, type_arg, value_arg, traceback_arg):
    stack = _CLUSTER_STACK.stack
    assert stack
    assert stack[-1] is self
    _CLUSTER_STACK.stack.pop()

  def __init__(self, params):
    self._params = params.Copy()
    p = self.params

    # A set of invariants about the setup of the cluster.
    #
    # NOTE. Two job specs can be identical. E.g., if p.worker.name is
    # the same as p.ps.name, that means ps is colocated with worker.
    assert p.ps.replicas >= 0
    assert p.ps.gpus_per_replica >= 0
    assert p.input.replicas <= 1
    if p.mode == 'async' and p.job == 'controller':
      # There is only 1 controller.
      assert p.controller.replicas == 1
      assert p.task == 0
    elif p.mode == 'async' and p.job == 'trainer':
      assert p.worker.replicas >= 1
      assert p.worker.gpus_per_replica >= 0
      assert p.worker.devices_per_split >= 1
      # In async mode, trainers colocate with workers.
      assert 0 <= p.task and p.task < p.worker.replicas
      if p.ps.replicas == 0:
        # There is no ps. We are doing single-replica training.
        assert p.worker.replicas == 1
    elif p.mode == 'async' and p.job == 'evaler':
      assert 0 <= p.task and p.task < p.evaler.replicas
    elif p.mode == 'async' and p.job == 'decoder':
      assert 0 <= p.task and p.task < p.decoder.replicas
    elif p.mode == 'sync' and p.job == 'controller':
      # There is only 1 controller.
      assert p.controller.replicas == 1
      assert p.task == 0
    elif p.mode == 'sync' and p.job == 'trainer_client':
      assert p.worker.replicas >= 1
      assert p.worker.gpus_per_replica >= 0
      assert p.worker.devices_per_split >= 1
    elif p.mode == 'sync' and p.job == 'evaler':
      assert 0 <= p.task and p.task < p.evaler.replicas
    elif p.mode == 'sync' and p.job == 'decoder':
      assert 0 <= p.task and p.task < p.decoder.replicas
    else:
      assert False, (p.mode, p.job)

    if p.job == 'controller':
      self._job_spec = p.controller
    elif p.job in ('trainer', 'worker', 'trainer_client'):
      self._job_spec = p.worker
    elif p.job == 'evaler':
      self._job_spec = p.evaler
    elif p.job == 'decoder':
      self._job_spec = p.decoder

  @property
  def params(self):
    return self._params

  @property
  def mode(self):
    return self.params.mode

  @property
  def job(self):
    return self.params.job

  @property
  def task(self):
    return self.params.task

  @property
  def job_spec(self):
    return self._job_spec

  @property
  def asynchronous(self):
    """Returns True if configured for asynchronous training."""
    return self.params.mode == 'async'

  @property
  def synchronous(self):
    """Returns True if configured for synchronous training."""
    return self.params.mode == 'sync'

  @property
  def num_replicas(self):
    return self._job_spec.replicas

  @property
  def tpus_per_replica(self):
    return self._job_spec.tpus_per_replica

  @property
  def num_tpu_hosts(self):
    return self._job_spec.num_tpu_hosts

  @property
  def num_devices_per_replica(self):
    return (self._job_spec.gpus_per_replica or
            self._job_spec.tpus_per_replica or self._job_spec.cpus_per_replica)

  @property
  def total_worker_devices(self):
    """Return the total number of discrete worker devices in the cluster."""
    worker_spec = self.params.worker
    devices_per_replica = (
        worker_spec.gpus_per_replica or worker_spec.tpus_per_replica or
        self._job_spec.cpus_per_replica)
    num_replicas = worker_spec.replicas
    return devices_per_replica * num_replicas

  @property
  def num_devices_per_split(self):
    """Return number of accelerators to use per split."""
    return self._job_spec.devices_per_split

  @property
  def num_splits_per_replica(self):
    # Note that a split must be within a replica.
    assert self.num_devices_per_replica % self.num_devices_per_split == 0
    return int(self.num_devices_per_replica / self.num_devices_per_split)

  @property
  def num_splits_per_client(self):
    """The number of splits visible by one trainer client."""
    if self.synchronous and self.job == 'trainer_client':
      # One client drives all the workers.
      return self.num_splits_per_replica * self.num_replicas
    else:
      # One client colocates with one worker and drives the worker only.
      return self.num_splits_per_replica

  @property
  def available_devices(self):
    """Returns all compute devices available in a 2D array.

    Returns:
      A 2D array (python list of python lists) of strings. ret[i, j]
      is the j-th visible device on i-th visible replica.
    """
    if self._job_spec.tpus_per_replica:
      ret = np.empty((1, self.num_devices_per_split), np.object)
      for i in range(self.num_devices_per_split):
        ret[0, i] = tf.contrib.tpu.core(i)
      return ret

    if self.job == 'trainer' and self.asynchronous:
      # In async mode, each trainer task can only use its own devices.
      return self.ListDevices(self._job_spec)[self.task:(self.task + 1), :]

    if self.job == 'trainer_client' and self.synchronous:
      # In sync mode, trainer_client can use every device.
      return self.ListDevices(self._job_spec)

    if self.job in ('controller', 'evaler', 'decoder'):
      # Our current policy is that each controller/evaler/decoder task
      # only uses 1 replica.
      return self.ListDevices(self._job_spec)[self.task:(self.task + 1), :]

    assert False, (self.job, self.mode)

  @property
  def input_device(self):
    """Returns the tensorflow device name to place input op on."""
    p = self.params
    if self.synchronous and p.input.replicas > 0:
      # Uses a separate job for input processing.
      assert p.input.replicas == 1
      return self.ListDevices(p.input)[0, 0]
    else:
      return ''

  def WorkerDeviceInModelSplit(self, device_index):
    """Returns the device to use for 'device_index' for the current model split.

    Args:
      device_index: An int, the device index within 'model_split'.

    Returns:
      A string. The device to place ops onto.
    """
    devices = self.available_devices.reshape([-1]).tolist()
    if not devices:
      return ''
    else:
      model_split = py_utils.GetModelSplit()
      assert model_split < self.num_splits_per_client, (
          '%d %d' % (model_split, self.num_splits_per_client))
      devices_per_split = self.num_devices_per_split
      return devices[devices_per_split * model_split +
                     device_index % devices_per_split]

  def GetPlacer(self, strategy=None):
    """Returns a device function for placing ops within the cluster.

    Args:
      strategy: A string. Identifier for a placement strategy. By default, we
        use a least loaded policy to place variables.

    Returns:
      Returns a device function can be used in tf.device().

    Raises:
      ValueError: when strategy is not supported.
    """
    if self.job == 'evaler' or self.job == 'decoder':
      # Currently, we only support evaler/decoder uses 1 accelerator.
      return self.ListDevices(self.job_spec)[self.task, 0]
    elif strategy is None:
      return _LeastLoadedPlacer(self).DeviceFunction
    raise ValueError('Unsupported placement policy: ', strategy)

  @property
  def add_summary(self):
    p = self.params
    if p.add_summary is None:
      return self.job in ['controller', 'decoder']
    else:
      return p.add_summary


# Ops that must be placed on the 'ps' devices.
_VAR_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable', 'VarHandleOp']


class VarPlacer(object):
  """Placer which places variables across a set of devices.

  VarPlacer places non-variable ops on the worker device.
  """

  def __init__(self, cluster):
    self._cluster = cluster
    self._devices = cluster.ListDevices(cluster.job_spec)

  def _AssignVar(self, _):
    raise ValueError('Unimplemented')

  def DeviceFunction(self, op):
    """Choose a device for 'op'.

    Args:
      op: an Operation.

    Returns:
      The device to use for the Operation.
    """
    # Op has already assigned to a device explicitly. Don't change it.
    if op.device:
      return op.device

    # Place vars according our policy.
    if op.type in _VAR_OPS:
      return self._AssignVar(op)

    # The default policy is to place the op on the 1st device visible
    # to this task.
    assert self._devices is not None, ('Unexpected job: %s' % self._cluster.job)
    task = self._cluster.params.task
    assert 0 <= task and task < len(self._devices)
    return self._devices[task, 0]


class _LeastLoadedPlacer(VarPlacer):
  """Placer which places a variable on the least loaded var device.

  We use total byte sizes of variables placed on a device to indicate
  the device's load.

  """

  def __init__(self, cluster):
    super(_LeastLoadedPlacer, self).__init__(cluster)
    # A min heap of (size, device)
    var_devices = cluster.ListDevices(cluster.params.ps).flatten().tolist()
    tf.logging.info('_LeastLoadedPlacer : %s', var_devices)
    self._var_space_pq = [(0, d) for d in var_devices]

  def _AssignVar(self, var_op):
    size = var_op.get_attr('dtype').size
    shape = tf.TensorShape(var_op.get_attr('shape'))
    assert self._var_space_pq, ('No ps devices to use.')
    allocated, device = heapq.heappop(self._var_space_pq)
    if shape.num_elements() is None:
      assert var_op.name.endswith(
          'wb/var'), 'Unexpected name pattern: %s' % var_op.name
      # CuDNN RNN vars shape aren't known statically, decide to make a constant
      # estimate to avoid introducing more complexities.
      allocated += 10 * 1024**2 * size
    else:
      allocated += shape.num_elements() * size
    heapq.heappush(self._var_space_pq, (allocated, device))
    tf.logging.info('Place variable %s on %s %d', var_op.name, device,
                    allocated)
    return device
