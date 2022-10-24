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
r"""GShard decoder class."""
import threading
import time

from lingvo import compat as tf
from lingvo.core import cluster as lingvo_cluster
from lingvo.core import cluster_factory
from lingvo.core import gshard_utils
from lingvo.core import py_utils
from lingvo.core import tpu_summary
import numpy as np

# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.tpu import device_assignment as tpu_device_assignment
from tensorflow.python.tpu import tpu as tpu_lib
from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import training_loop
from tensorflow.python.tpu.ops import tpu_ops
# pylint: enable=g-direct-tensorflow-import


def preload_zero(n=None, batch_size=None, max_len=None, key_size=2):
  """Returns the same structure as preload_unpacked but with zeros."""
  batch = (
      # key: (fileno + lineno) in batch mode
      #      (rpc id + timestamp) in rpc mode
      np.zeros([n, batch_size, key_size], np.int32),
      # tgt_id
      np.zeros([n, batch_size, max_len], np.int32),
      # tgt_segment_id
      np.zeros([n, batch_size, max_len], np.float32),
      # tgt_segment_pos
      np.zeros([n, batch_size, max_len], np.int32),
      # tgt_labels
      np.zeros([n, batch_size, max_len], np.int32),
      # tgt_sample_temperature
      np.zeros([n, batch_size], np.float32),
  )
  return batch


def get_zero_batch(batch_size=None,
                   max_len=None,
                   key_size=2,
                   return_tgt_mask=False,
                   return_scorer_alpha=False):
  """Returns zero batch.

  Args:
    batch_size: batch size.
    max_len: max length.
    key_size: key size.
    return_tgt_mask: if to return tgt_mask.
    return_scorer_alpha: if to return scorer_alpha used to set scaling factor
      for controlled decoding.
  Returns: a tuple of tensors
    key: int32 tensor [batch_size, key_size]
    tgt_id: int32 tensor [batch_size, max_len]
    tgt_segment_id: float32 tensor [batch_size, max_len]
    tgt_segment_pos: int32 tensor [batch_size, max_len]
    tgt_labels: int32 tensor [batch_size, max_len]
    tgt_sample_temperature: float32 tensor [batch_size]
    tgt_mask: optional float32 tensor [batch_size, max_len, max_len]
    tgt_scorer_alpha: float32 tensor [batch_size]
  """
  batch = preload_zero(
      n=1, batch_size=batch_size, max_len=max_len, key_size=key_size)
  batch = py_utils.Transform(lambda x: np.squeeze(x, 0), batch)
  if return_tgt_mask:
    tgt_mask = np.zeros([batch_size, max_len, max_len], np.float32)
    batch = (*batch, tgt_mask)
  if return_scorer_alpha:
    assert not return_tgt_mask
    scorer_alpha = np.zeros([batch_size], np.float32)
    batch = (*batch, scorer_alpha)
  return batch


# mimic training_loop.repeat(), but make it repeat forever.
def infinite_repeat(body_fn, infeed_queue):
  """Builds infinite loop.

  Args:
    body_fn: a Python function that builds the loop body.
    infeed_queue: if not None, the infeed queue from which to append a tuple of
      arguments as inputs to condition.

  Returns:
    The final values of the loop-carried tensors.
  """

  def to_list(x):
    if isinstance(x, (list, tuple)):
      return list(x)
    else:
      return [x]

  def body_fn_wrapper(i, *args):
    return [i + 1] + to_list(body_fn(*args))

  outputs = training_loop.while_loop(
      lambda i, *args: tf.constant(True),  # Infinite loop.
      body_fn_wrapper,
      inputs=[0],
      infeed_queue=infeed_queue)
  outputs = to_list(outputs)
  if len(outputs) == 1:
    # Returns the Op rather than an empty list.
    return outputs[0].op
  else:
    return outputs[1:]


def daemon(closure):
  """Runs the closure in a background thread."""
  thread = threading.Thread(target=closure)
  thread.daemon = True
  thread.start()
  return thread


class GShardDecode:
  """Base decoder class.

  Implements the main computation loop.

  Attrs:
    tpu: name or addresss of the tpu node.
    worker_job_name: job name of tpu.
    prefix_max_len: Length of prefix.
    cluster_params: cluster params.
    cluster: cluster object.
    graph: a tf.Graph() in which ops are build.
    task: the task object.
    compile_op: tpu program compile op.
    init_vars_op: op to init vars randomly.
    infeed_op: the tf op to infeed data.
    infeed_args: a list of placeholder nodes.
    outfeed_op: the tf op to poll outfeed.
    outfeed: a list of outfeed tensors. used as structure reference.
    decode_loop: the op to start decode loop.
    saver: tf.train.Saver object.
    num_batches: num of decode steps to run. If None, run infinitely.
    spm: SentencePieceModel object
  """

  def __init__(self,
               tpu=None,
               worker_job_name=None,
               prefix_max_len=128,
               is_cloud_tpu_node=False):
    self._tpu = tpu
    self._worker_job = worker_job_name
    self._prefix_max_len = prefix_max_len
    self._c = threading.Condition()  # lock
    # set in reset_session
    self._sess = None
    # set in configure_cluster_params
    self.cluster_params = None
    # set in init_graph
    self.cluster = None
    self.graph = tf.Graph()
    self.task = None
    self.compile_op = None
    self.init_vars_op = None
    self.infeed_op = None
    self.infeed_args = None
    self.outfeed_op = None
    self.outfeed = None
    self.decode_loop = None
    self.saver = None
    self.num_batches = None
    self.session_timeout_in_ms = None

    self._heartbeat = False
    self._saver_reshape = True
    # set in load_spm
    self.spm = None

    if worker_job_name is not None:
      if worker_job_name.startswith('/job:'):
        worker_job_name = worker_job_name.split(':')[1]
      else:
        self._worker_job = '/job:' + worker_job_name
    if is_cloud_tpu_node:
      cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          self._tpu, job_name=worker_job_name)
      self._cluster_def = cluster_resolver.cluster_spec().as_cluster_def()
      self._tpu = cluster_resolver.master()
    else:
      self._cluster_def = None

  def load_spm(self, spm):
    self.spm = gshard_utils.LoadSpm(spm)

  def reset_tpu_cluster(self):
    tf.logging.info('Connecting to tpu %s', self._tpu)
    with tf.container('') as container:
      # Kills all sessions on this cluster.
      tf.Session.reset(
          target=self._tpu,
          containers=[container],
          config=self._no_opt_sess_cfg())

  def get_session(self):
    self._c.acquire()
    while not self._sess:
      tf.logging.info('Waiting for session to be setup ...')
      self._c.wait()
    sess = self._sess
    self._c.release()
    return sess

  def _no_opt_sess_cfg(self):
    # Disable constant folding for convenience.
    return tf.config_pb2.ConfigProto(
        graph_options=tf.GraphOptions(
            optimizer_options=tf.OptimizerOptions(
                opt_level=tf.OptimizerOptions.L0,
                do_common_subexpression_elimination=False,
                do_function_inlining=False,
                do_constant_folding=False)),
        cluster_def=self._cluster_def)

  def reset_session(self, target=None):
    """Resets session on target worker with current graph."""
    self._c.acquire()
    if self._sess is not None:
      try:
        self._sess.close()
      except tf.errors.AbortedError as e:
        # It's ok if the session is already aborted.
        tf.logging.error('Exception %s', str(e))
        pass

    tf.logging.info('Creating new session ...')
    if target is None:
      target = self._tpu
    self._sess = tf.Session(
        target=target, graph=self.graph, config=self._no_opt_sess_cfg())
    tf.logging.info('Done creating new session.')
    self._c.notify()
    self._c.release()
    return self._sess

  def run_init_sequence(self):
    """Runs init sequences before decoding."""

    assert self.init_vars_op is not None
    assert self.compile_op is not None

    sess = self.reset_session(self._tpu)
    if self._heartbeat:
      self._start_heartbeat()

    if self.ckpt:

      def run_restore():
        tf.logging.info('Restoring vars from ckpt: start')
        try:
          self.saver.restore(sess, self.ckpt)
        except Exception as e:
          tf.logging.fatal('Restoring vars exception: %r %s', e, e)
          raise
        tf.logging.info('Restoring vars from ckpt: done')

      init_thread = daemon(run_restore)
    else:

      def run_init():
        tf.logging.info('Init vars randomly: start')
        try:
          sess.run(self.init_vars_op)
        except Exception as e:
          tf.logging.fatal('Init vars exception: %r %s', e, e)
          raise
        tf.logging.info('Init vars randomly: done')

      init_thread = daemon(run_init)

    if hasattr(self.task, 'input'):
      tf.logging.info('Init data')
      self.task.input.Initialize(sess)
      tf.logging.info('Init data done')

    tf.logging.info('Compile: start')
    run_options = tf.RunOptions(timeout_in_ms=86400 * 1000)
    sess.run(self.compile_op, options=run_options)
    tf.logging.info('Compile: done')

    init_thread.join()

  def _configure_cluster_params(self, tpu_cores=None, cpu_hosts=None):
    """Initialize cluster params."""
    tf.logging.info(cpu_hosts)
    cluster_factory.SetCluster(lingvo_cluster._Cluster)  # pylint: disable=protected-access
    cluster_params = cluster_factory.Cluster.Params()
    cluster_params.mode = 'sync'
    cluster_params.job = 'trainer_client'
    cluster_params.do_eval = True  # turn off dropout
    cluster_params.worker.name = self._worker_job
    cluster_params.worker.tpus_per_replica = tpu_cores
    cluster_params.worker.devices_per_split = tpu_cores
    cluster_params.worker.num_tpu_hosts = cpu_hosts
    cluster_params.ps.name = self._worker_job
    cluster_params.ps.replicas = cpu_hosts
    return cluster_params

  def _start_heartbeat(self):
    """Start the heartbeat."""

    def run_heartbeat_loop():
      count = 0
      # Set a timeout of 30 seconds for each heartbeat.
      if self.session_timeout_in_ms:
        timeout_in_ms = self.session_timeout_in_ms
      else:
        timeout_in_ms = 30 * 1000
      run_options = tf.RunOptions(timeout_in_ms=self.session_timeout_in_ms)
      while True:
        try:
          if count % 100 == 0:
            tf.logging.info('heartbeat: request_%d ...', count)
          t_begin = time.time()
          sess = self.get_session()
          ret = sess.run(self.heartbeat, options=run_options)
          if self.streamz_heartbeat_latency is not None:
            self.streamz_heartbeat_latency.Record((time.time() - t_begin) * 1e3)
          if count % 100 == 0:
            tf.logging.info('heartbeat: done request_%d ... %s', count, ret)
        except Exception as e:
          tf.logging.fatal('Exception in heartbeat loop thread: %r %s', e, e)
          raise
        count += 1
        # Once every 10 seconds.
        time.sleep(10)

    daemon(run_heartbeat_loop)

  def _config_infeed(self,
                     num_partitions,
                     device_assignment,
                     batch_size,
                     key_size=2,
                     return_tgt_mask=False,
                     return_scorer_alpha=False,
                     use_partitioned_infeed_queue=False):
    """Config the infeed ops and args."""
    zero_batch = get_zero_batch(
        batch_size=batch_size,
        max_len=self._prefix_max_len,
        key_size=key_size,
        return_tgt_mask=return_tgt_mask,
        return_scorer_alpha=return_scorer_alpha)

    host_device = device_assignment.host_device(replica=0, job=self._tpu)
    host_id = int(host_device.split('/task:')[1].split('/device:')[0])
    input_partition_dims = [
        [num_partitions] + [1] * (len(x.shape) - 1) for x in zero_batch
    ]

    if use_partitioned_infeed_queue:
      infeed = tpu_feed._PartitionedInfeedQueue(  # pylint: disable=protected-access
          number_of_tuple_elements=len(zero_batch),
          host_id=host_id,
          input_partition_dims=input_partition_dims,
          device_assignment=device_assignment)
    else:
      infeed = tpu_feed.InfeedQueue(number_of_tuple_elements=len(zero_batch))

    self.infeed_args = []
    for x in zero_batch:
      p = tf.placeholder(tf.as_dtype(x.dtype), shape=x.shape)
      self.infeed_args += [p]
    if use_partitioned_infeed_queue:
      self.infeed_op = infeed.generate_enqueue_ops([self.infeed_args])
    else:
      self.infeed_op = infeed.split_inputs_and_generate_enqueue_ops(
          self.infeed_args, device_assignment=device_assignment)
    return infeed

  def _init_tpu(self, num_partitions, device_order_mode):
    """Initialize tpu device assignment."""
    tf.logging.info('Initializing TPU to get device assignment: start')

    graph = tf.Graph()
    with graph.as_default():
      init_tpu_op = tf.tpu.initialize_system()
    try:
      sess = tf.Session(
          target=self._tpu, graph=graph, config=self._no_opt_sess_cfg())
      topology = sess.run(init_tpu_op)
    except Exception as e:
      tf.logging.fatal('TPU initialization failed: %s', e)
      raise

    topology_proto = topology_pb2.TopologyProto()
    topology_proto.ParseFromString(topology)
    tf.logging.info('topology.num_tasks: %r', topology_proto.num_tasks)
    tf.logging.info('topology.num_tpu_devices_per_task: %r',
                    topology_proto.num_tpu_devices_per_task)
    tf.logging.info('topology.mesh_shape: %r', topology_proto.mesh_shape)
    self.cluster_params = self._configure_cluster_params(
        tpu_cores=(topology_proto.num_tpu_devices_per_task *
                   topology_proto.num_tasks),
        cpu_hosts=topology_proto.num_tasks)

    # We assume the topology and device assignment does not change
    # for a single address space.
    device_assignment = tpu_device_assignment.device_assignment(
        topology,
        computation_shape=py_utils.ComputationShape(num_partitions, topology),
        num_replicas=1,
        device_order_mode=device_order_mode)
    py_utils.SetTpuDeviceAssignment(device_assignment)

    tf.logging.info('Initializing TPU to get device assignment: done')

  def init_graph(self, model_params):
    """Builds moe decode graph.

    Args:
      model_params: the hyperparams of the specified model.
    """
    assert self.graph
    self.model_params = model_params
    batch_size = model_params.task.batch_size
    if (hasattr(model_params.task.builder, 'device_mesh_shape') and
        model_params.task.builder.device_mesh_shape):
      num_partitions = np.prod(model_params.task.builder.device_mesh_shape)
    else:
      num_partitions = model_params.task.builder.num_devices

    device_order_mode = (
        model_params.task.train.tpu_device_order_mode or
        tpu_device_assignment.DeviceOrderMode.AUTO)
    self._init_tpu(num_partitions, device_order_mode)
    assert self.cluster_params  # configured by init_tpu
    self.cluster = self.cluster_params.Instantiate()

    with self.graph.as_default(), self.cluster, tf.device(
        self.cluster.GetPlacer()):
      _ = py_utils.GetOrCreateGlobalStepVar()
      self.heartbeat = tf.constant(np.pi)

      device_assignment = py_utils.GetTpuDeviceAssignment()

      tf.logging.info('Instantiating model')
      model = model_params.Instantiate()
      xformer = model.GetTask()
      self.task = xformer

      self.init_vars_op = tf.global_variables_initializer()
      self.saver = tf.train.Saver(sharded=True, reshape=self._saver_reshape)

      infeed = self._config_infeed(
          num_partitions=num_partitions,
          device_assignment=device_assignment,
          batch_size=batch_size)

      self.outfeed = []

      def decode_fn(*infeed_batch):  # pylint: disable=missing-docstring
        # Length 6 is passed when there is no tgt_mask (e.g. decoding) and
        # length 7 is passed when there is a tgt_mask (e.g. fprop).

        self.outfeed = self._config_outfeed(xformer, infeed_batch)

        with tf.device(tf.tpu.core(0)):
          outfeed_op = tpu_ops.outfeed_enqueue_tuple(
              tf.nest.flatten(self.outfeed))

        return [outfeed_op]

      @tpu_function.on_device_training_loop
      def decode_loop_fn():
        if not self.num_batches:
          infinite_repeat(decode_fn, infeed)
        else:
          training_loop.repeat(self.num_batches, decode_fn, infeed_queue=infeed)

      self.compile_op, self.decode_loop = tpu_lib.split_compile_and_shard(
          decode_loop_fn, num_shards=1, device_assignment=device_assignment)

      assert self.outfeed
      with tf.device(device_assignment.tpu_device(0, 0)):
        self.outfeed_op = tpu_ops.outfeed_dequeue_tuple(
            dtypes=[x.dtype for x in tf.nest.flatten(self.outfeed)],
            shapes=[x.shape for x in tf.nest.flatten(self.outfeed)])

  def _config_outfeed(self, xformer, infeed_batch):
    """Setup the outfeed ops."""
    fprop_dtype = py_utils.FPropDtype(self.model_params.task)

    assert len(infeed_batch) == 6 or len(infeed_batch) == 7, len(infeed_batch)
    if len(infeed_batch) == 7:
      (key, tgt_ids, tgt_segment_id, tgt_segment_pos, tgt_labels, _,
       _) = infeed_batch
    elif len(infeed_batch) == 6:
      (key, tgt_ids, tgt_segment_id, tgt_segment_pos, tgt_labels,
       _) = infeed_batch
    tgt_segment_id = tf.cast(tgt_segment_id, fprop_dtype)

    input_batch = py_utils.NestedMap()
    input_batch.src = py_utils.NestedMap()
    input_batch.src.ids = (0 * tgt_ids)  # unused
    input_batch.src.segment_ids = (0 * tgt_segment_id)  # unused
    input_batch.src.segment_pos = (0 * tgt_segment_pos)  # unused
    input_batch.tgt = py_utils.NestedMap()
    input_batch.tgt.ids = tgt_ids
    input_batch.tgt.segment_ids = tgt_segment_id
    input_batch.tgt.segment_pos = tgt_segment_pos
    input_batch.tgt.labels = tgt_labels  # only used when --fprop=true

    with tpu_summary.context(rewrite_while_loop=True):
      dec_ret = xformer.DecodeIds(xformer.theta, input_batch)
      dec_metrics = tpu_summary.merge_all()
      key = infeed_batch[0]
      return [
          key, tgt_ids, tgt_segment_id, dec_ret.topk_ids, dec_ret.topk_lens,
          dec_ret.topk_scores, dec_metrics
      ]
