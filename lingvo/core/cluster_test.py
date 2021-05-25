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
"""Tests for cluster."""

import lingvo.compat as tf
from lingvo.core import cluster
from lingvo.core import cluster_factory
from lingvo.core import test_utils
import numpy as np


class ClusterTest(test_utils.TestCase):

  def testDefaultParams(self):
    p = cluster_factory.Cluster.Params()
    c = cluster_factory.Cluster(p)
    self.assertFalse(c.add_summary)
    g = tf.Graph()
    vs = []
    with g.as_default():
      with tf.device(c.GetPlacer()):
        for i in range(10):
          vs.append(tf.get_variable('x%d' % i, (10, 10, 10)))
        sum_all = tf.add_n(vs)
    for v in vs:
      self.assertEqual(
          v.device,
          cluster.MakeDeviceString(
              job_name='/job:localhost',
              replica_id=0,
              task_id=0,
              device_name='CPU',
              device_id=0))
    self.assertEqual(
        sum_all.device,
        cluster.MakeDeviceString(
            job_name='/job:localhost',
            replica_id=0,
            task_id=0,
            device_name='CPU',
            device_id=0))

  def testDefaultParamsWithDynamicShape(self):
    p = cluster_factory.Cluster.Params()
    c = cluster_factory.Cluster(p)
    g = tf.Graph()
    vs = []
    with g.as_default():
      with tf.device(c.GetPlacer()):
        for i in range(10):
          dyn_shape = tf.constant([2], dtype=tf.int32)
          dyn_shape = tf.placeholder_with_default(dyn_shape, shape=[None])
          v = tf.get_variable(
              'x%d_wb/var' % i,
              initializer=tf.random.uniform(dyn_shape, dtype=tf.float64),
              validate_shape=False)
          vs.append(v)
        sum_all = tf.add_n(vs)
    for v in vs:
      self.assertEqual(
          v.device,
          cluster.MakeDeviceString(
              job_name='/job:localhost',
              replica_id=0,
              task_id=0,
              device_name='CPU',
              device_id=0))
    self.assertEqual(
        sum_all.device,
        cluster.MakeDeviceString(
            job_name='/job:localhost',
            replica_id=0,
            task_id=0,
            device_name='CPU',
            device_id=0))

  def testNoPS(self):
    p = cluster_factory.Cluster.Params()
    p.worker.name = '/job:trainer'
    p.worker.replicas = 1
    p.ps.name = '/job:trainer'
    p.ps.replicas = 1
    c = cluster_factory.Cluster(p)
    g = tf.Graph()
    vs = []
    with g.as_default():
      with tf.device(c.GetPlacer()):
        for i in range(10):
          vs.append(tf.get_variable('x%d' % i, (10, 10, 10)))
        sum_all = tf.add_n(vs)
    for v in vs:
      self.assertEqual(
          v.device,
          cluster.MakeDeviceString(
              job_name='/job:trainer',
              replica_id=0,
              task_id=0,
              device_name='CPU',
              device_id=0))
    self.assertEqual(
        sum_all.device,
        cluster.MakeDeviceString(
            job_name='/job:trainer',
            replica_id=0,
            task_id=0,
            device_name='CPU',
            device_id=0))

  def testNoPSWithGPUs(self):
    p = cluster_factory.Cluster.Params()
    p.worker.name = '/job:trainer'
    p.worker.replicas = 1
    p.worker.gpus_per_replica = 4
    p.ps.name = '/job:trainer'
    p.ps.replicas = 1
    p.ps.gpus_per_replica = 4

    c = cluster_factory.Cluster(p)
    g = tf.Graph()
    vs = []
    with g.as_default():
      with tf.device(c.GetPlacer()):
        for i in range(10):
          vs.append(tf.get_variable('x%d' % i, (10, 10, 10)))
        sum_all = tf.add_n(vs)
    for i, v in enumerate(vs):
      self.assertEqual(
          v.device,
          cluster.MakeDeviceString(
              job_name='/job:trainer',
              replica_id=0,
              task_id=0,
              device_name='GPU',
              device_id=i % 4))
      self.assertEqual(
          sum_all.device,
          cluster.MakeDeviceString(
              job_name='/job:trainer',
              replica_id=0,
              task_id=0,
              device_name='GPU',
              device_id=0))

  def testPS(self):
    p = cluster_factory.Cluster.Params()
    p.worker.name = '/job:trainer'
    p.worker.replicas = 1
    p.ps.name = '/job:ps'
    p.ps.replicas = 4
    c = cluster_factory.Cluster(p)
    g = tf.Graph()
    vs = []
    with g.as_default():
      with tf.device(c.GetPlacer()):
        for i in range(10):
          vs.append(tf.get_variable('x%d' % i, (10, 10, 10)))
        sum_all = tf.add_n(vs)
    for i, v in enumerate(vs):
      self.assertEqual(
          v.device,
          cluster.MakeDeviceString(
              job_name='/job:ps',
              replica_id=0,
              task_id=i % 4,
              device_name='CPU',
              device_id=0))
    self.assertEqual(
        sum_all.device,
        cluster.MakeDeviceString(
            job_name='/job:trainer',
            replica_id=0,
            task_id=0,
            device_name='CPU',
            device_id=0))

  def testPSWithGPUs(self):
    p = cluster_factory.Cluster.Params()
    p.worker.name = '/job:trainer'
    p.worker.replicas = 1
    p.ps.name = '/job:ps'
    p.ps.replicas = 4
    p.ps.gpus_per_replica = 2
    c = cluster_factory.Cluster(p)
    g = tf.Graph()
    vs = []
    with g.as_default():
      with tf.device(c.GetPlacer()):
        for i in range(10):
          vs.append(tf.get_variable('x%d' % i, (10, 10, 10)))
        sum_all = tf.add_n(vs)
    for i, v in enumerate(vs):
      self.assertEqual(
          v.device,
          cluster.MakeDeviceString(
              job_name='/job:ps',
              replica_id=0,
              task_id=(i / 2) % 4,
              device_name='GPU',
              device_id=i % 2))
    self.assertEqual(
        sum_all.device,
        cluster.MakeDeviceString(
            job_name='/job:trainer',
            replica_id=0,
            task_id=0,
            device_name='CPU',
            device_id=0))

  def testPSRandomSize(self):
    p = cluster_factory.Cluster.Params()
    p.worker.name = '/job:trainer'
    p.ps.name = '/job:ps'
    p.ps.replicas = 10
    c = cluster_factory.Cluster(p)
    g = tf.Graph()
    vs = []
    np.random.seed(301)
    with g.as_default():
      with tf.device(c.GetPlacer()):
        # Creates 200 variables with different sizes.
        for i in range(200):
          if i % 13:
            size = np.random.randint(10000)
          elif i % 7:
            size = np.random.randint(100)
          else:
            size = np.random.randint(10)
          vs.append(tf.get_variable('x%d' % i, shape=(size)))
        sum_all = tf.add_n([tf.reduce_sum(x) for x in vs])
    # Computes the total size of variables placed on each device.
    total_size = {}  # device name -> size
    for v in vs:
      size = tf.TensorShape(v.op.get_attr('shape')).num_elements()
      if v.device in total_size:
        total_size[v.device] += size
      else:
        total_size[v.device] = size
    for (device, allocated) in zip(
        sorted(total_size),
        [91701, 91361, 90346, 88738, 87240, 89265, 91944, 92472, 88051, 95053]):
      self.assertEqual(total_size[device], allocated)
    self.assertEqual(
        sum_all.device,
        cluster.MakeDeviceString(
            job_name='/job:trainer',
            replica_id=0,
            task_id=0,
            device_name='CPU',
            device_id=0))

  def testDeviceListOneRepliaCpu(self):
    p = cluster_factory.Cluster.Params()
    p.mode = 'async'
    p.job = 'trainer'
    p.worker.cpus_per_replica = 2
    c = cluster_factory.Cluster(p)
    cpu_devices = c.available_devices
    expected_cpu_devices = [[
        cluster.MakeDeviceString(
            job_name='/job:localhost',
            replica_id=0,
            task_id=0,
            device_name='CPU',
            device_id=0),
        cluster.MakeDeviceString(
            job_name='/job:localhost',
            replica_id=0,
            task_id=0,
            device_name='CPU',
            device_id=1),
    ]]
    print(expected_cpu_devices)
    self.assertAllEqual(cpu_devices, expected_cpu_devices)

  def testDeviceListOneReplicaGpu(self):
    p = cluster_factory.Cluster.Params()
    p.mode = 'async'
    p.job = 'trainer'
    p.worker.gpus_per_replica = 2
    c = cluster_factory.Cluster(p)
    gpu_devices = c.available_devices
    expected_gpu_devices = [[
        cluster.MakeDeviceString(
            job_name='/job:localhost',
            replica_id=0,
            task_id=0,
            device_name='GPU',
            device_id=0),
        cluster.MakeDeviceString(
            job_name='/job:localhost',
            replica_id=0,
            task_id=0,
            device_name='GPU',
            device_id=1),
    ]]
    self.assertAllEqual(gpu_devices, expected_gpu_devices)

  def testDeviceListMultiReplicaNoSyncSgd(self):
    p = cluster_factory.Cluster.Params()
    p.mode = 'async'
    p.job = 'trainer'
    p.task = 1
    p.worker.replicas = 2
    p.worker.gpus_per_replica = 2
    c = cluster_factory.Cluster(p)
    gpu_devices = c.available_devices
    expected_gpu_devices = [[
        cluster.MakeDeviceString(
            job_name='/job:localhost',
            replica_id=0,
            task_id=1,
            device_name='GPU',
            device_id=0),
        cluster.MakeDeviceString(
            job_name='/job:localhost',
            replica_id=0,
            task_id=1,
            device_name='GPU',
            device_id=1),
    ]]
    self.assertAllEqual(gpu_devices, expected_gpu_devices)

    # Compute the total number of worker devices for a multi
    # replica setup.
    self.assertEqual(4, c.total_worker_devices)

    # Even when the job is different, we still look at the worker
    # information.
    p.job = 'controller'
    p.task = 0
    c = cluster_factory.Cluster(p)
    self.assertEqual(4, c.total_worker_devices)

  def testDeviceListMultiReplicaSyncSgd(self):
    p = cluster_factory.Cluster.Params()
    p.mode = 'sync'
    p.job = 'trainer_client'
    p.worker.name = '/job:localhost'
    p.worker.replicas = 2
    p.worker.gpus_per_replica = 2
    c = cluster_factory.Cluster(p)
    gpu_devices = c.available_devices
    expected_gpu_devices = [[
        cluster.MakeDeviceString(
            job_name='/job:localhost',
            replica_id=0,
            task_id=0,
            device_name='GPU',
            device_id=0),
        cluster.MakeDeviceString(
            job_name='/job:localhost',
            replica_id=0,
            task_id=0,
            device_name='GPU',
            device_id=1),
    ],
                            [
                                cluster.MakeDeviceString(
                                    job_name='/job:localhost',
                                    replica_id=0,
                                    task_id=1,
                                    device_name='GPU',
                                    device_id=0),
                                cluster.MakeDeviceString(
                                    job_name='/job:localhost',
                                    replica_id=0,
                                    task_id=1,
                                    device_name='GPU',
                                    device_id=1),
                            ]]
    self.assertAllEqual(gpu_devices, expected_gpu_devices)

  def testInputDevice(self):
    p = cluster_factory.Cluster.Params()
    p.mode = 'sync'
    p.job = 'decoder'
    p.decoder.replicas = 1
    p.task = 0
    p.input.name = '/job:input'
    p.input.replicas = 1
    c = cluster_factory.Cluster(p)
    input_device = c.input_device
    expected_device = cluster.MakeDeviceString(
        job_name='/job:input',
        replica_id=0,
        task_id=0,
        device_name='CPU',
        device_id=0)
    self.assertEqual(input_device, expected_device)

  def testModelSplit(self):
    p = cluster_factory.Cluster.Params()
    with p.Instantiate() as c:
      with cluster_factory.SetModelSplit(2) as c1:
        self.assertEqual(c1.params.split_id, 2)
        with cluster_factory.SetModelSplit(3) as c2:
          self.assertEqual(c2.params.split_id, 3)
      self.assertEqual(c.params.split_id, 0)

  def testInputTargets(self):
    p = cluster_factory.Cluster.Params()
    p.input.name = '/job:input'
    p.input.replicas = 2
    p.input.targets = '10.100.1.1:10001,10.100.1.2:10002'
    c = cluster_factory.Cluster(p)
    self.assertEqual(c.input_targets, ['10.100.1.1:10001', '10.100.1.2:10002'])

  def testWorkerDeviceInModelSplitSync(self):
    p = cluster_factory.Cluster.Params()
    p.mode = 'sync'
    p.job = 'trainer_client'
    p.worker.name = '/job:trainer'
    p.worker.replicas = 4
    p.worker.gpus_per_replica = 4
    p.worker.devices_per_split = 2
    with cluster_factory.Cluster(p):
      with cluster_factory.SetModelSplit(1) as c:
        d = c.WorkerDeviceInModelSplit(1)
        expected_device = cluster.MakeDeviceString(
            job_name='/job:trainer',
            replica_id=0,
            task_id=0,
            device_name='GPU',
            device_id=3)
    self.assertEqual(expected_device, d)

  def testWorkerDeviceInModelSplit(self):
    p = cluster_factory.Cluster.Params()
    p.mode = 'async'
    p.job = 'trainer'
    p.task = 3
    p.worker.name = '/job:trainer'
    p.worker.replicas = 4
    p.worker.gpus_per_replica = 4
    p.worker.devices_per_split = 2
    with cluster_factory.Cluster(p):
      with cluster_factory.SetModelSplit(1) as c:
        d = c.WorkerDeviceInModelSplit(1)
        expected_device = cluster.MakeDeviceString(
            job_name='/job:trainer',
            replica_id=0,
            task_id=3,
            device_name='GPU',
            device_id=3)
    self.assertEqual(expected_device, d)

  def testWorkerClusterDef(self):
    p = cluster_factory.Cluster.Params()

    p.worker.name = '/job:trainer_client'
    p.worker.targets = ','.join([
        'grpc://localhost:8470',
        'grpc://localhost:8471',
        'grpc://localhost:8472',
        'grpc://localhost:8473',
    ])

    cluster_def = tf.train.ClusterSpec({
        'trainer_client': [
            'localhost:8470', 'localhost:8471', 'localhost:8472',
            'localhost:8473'
        ]
    }).as_cluster_def()

    self.assertEqual(p.Instantiate().worker_cluster_def, cluster_def)

  def testWorkerClusterDefWithoutTargets(self):
    p = cluster_factory.Cluster.Params()

    p.worker.name = '/job:trainer_client'

    self.assertIsNone(p.Instantiate().worker_cluster_def)


if __name__ == '__main__':
  tf.test.main()
