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
"""Cluster factory."""

import copy
from lingvo.core import cluster

Cluster = cluster._Cluster  # pylint: disable=protected-access


def SetCluster(cls):
  """Sets Cluster implementation."""
  global Cluster  # pylint: disable=invalid-name
  Cluster = cls


def Current():
  """Returns the current cluster specification.

  E.g.::

    with Cluster(cluster_params) as foo:
      model = p.Instantiate()
      model.FProp()  # FProp can access foo through cluster_factory.Current().
  """
  current = Cluster.Top()
  if not current:
    current = Cluster(Cluster.Params())
  return current


def ForTestingWorker(mode=None,
                     job=None,
                     gpus=None,
                     split_size=None,
                     tpus=None,
                     add_summary=None,
                     cpus=None,
                     do_eval=None,
                     num_tpu_hosts=None):
  """Returns a Cluster for unittesting with a worker."""
  ret = copy.deepcopy(Current())
  p = ret.params
  if mode is not None:
    p.mode = mode
  if job is not None:
    p.job = job
  if gpus is not None:
    p.worker.gpus_per_replica = gpus
  if tpus is not None:
    p.worker.tpus_per_replica = tpus
    p.worker.num_tpu_hosts = 1
  if cpus is not None:
    p.worker.cpus_per_replica = cpus
  if split_size is not None:
    p.worker.devices_per_split = split_size
  if add_summary is not None:
    p.add_summary = add_summary
  if do_eval is not None:
    p.do_eval = do_eval
  if num_tpu_hosts is not None:
    p.worker.num_tpu_hosts = num_tpu_hosts
  return ret


def SetEval(mode):
  """Returns a cluster with do_eval option turned on/off.

  E.g.::

    def FProp(...):
      with SetEval(mode=True):
        # Turns off dropout, noise, etc.
        y = self.foo.FProp(..., x)
        z = self.bar.FProp(..., y)
      # Returns to previous state (e.g., training).
      y = self.foo.FProp(..., x)
      z = self.foo.FProp(..., y)

  Args:
    mode: True, False or None.

  Returns:
    A new Cluster instance.
  """
  ret = copy.deepcopy(Current())
  ret.params.do_eval = mode
  return ret


def SetImmediatelyInstantiateVariables(mode):
  """Returns a cluster with immediately_instantiate_variables option set.

  Args:
    mode: True, False or None.

  Returns:
    A new Cluster instance.
  """
  ret = copy.deepcopy(Current())
  ret.params.immediately_instantiate_variables = mode
  return ret


def SetRequireSequentialInputOrder(mode):
  """Returns a cluster with require_sequential_input_order option set.

  Args:
    mode: True, False or None.

  Returns:
    A new Cluster instance.
  """
  ret = copy.deepcopy(Current())
  ret.params.require_sequential_input_order = mode
  return ret


def SetModelSplit(split_id):
  """Returns the current cluster with the model split id set.

  E.g.::

    def FProp(...):
      with cluster_factory.SetModelSplit(1) as c:
        with tf.device(c.WorkerDeviceInModelSplit(0)):
          ...

  Args:
    split_id: Integer split id for the model.

  Returns:
    A new Cluster instance.
  """
  ret = copy.deepcopy(Current())
  ret.params.split_id = split_id
  return ret
