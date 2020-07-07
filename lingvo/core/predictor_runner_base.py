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
"""Interface for binaries built around predictor.

To use: subclass PredictorRunnerBase, implement the InputGenerator and RunBatch
functions, and call Run().

To run on TPU, set:
  --device_type=tpu
  --xla_device=tpu
  --tf_master=url/to/tpu/server
  --inference_threads=num_tpu_cores
"""

import concurrent.futures
import itertools
import os
import re
import threading
import time

from absl import flags
from lingvo import compat as tf
from lingvo.core import predictor
from lingvo.core import py_utils
import six

flags.DEFINE_string(
    'checkpoint', None, 'Either a checkpoint file to load,'
    ' or a directory containing multiple checkpoints, where'
    ' the latest checkpoint will be loaded.')
flags.DEFINE_string(
    'inference_graph', None, 'Path to an inference graph. '
    'If not specified, will be inferred from the checkpoint path.')
flags.DEFINE_string(
    'inference_subgraph_name', '', 'The name of the inference subgraph to use. '
    'Defaults to the default subgraph.')
flags.DEFINE_enum('device_type', 'gpu', ['cpu', 'gpu', 'tpu'], 'Device type.')
flags.DEFINE_string('tf_master', 'local', 'tf_master for predictor session.')
flags.DEFINE_integer('inference_threads', '1', 'Number of inference threads.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer(
    'prediction_step_interval', 3000, 'Number of steps between outputs. '
    'Only meaningful if FLAGS.checkpoint is a directory.')

flags.DEFINE_integer('max_inputs', 0, 'Only process the first n inputs.')
flags.DEFINE_list(
    'input_id_filter', [],
    'If not empty, only process the input ids in the given list.')
flags.DEFINE_string(
    'output_dir', None, 'Output directory. '
    'If FLAGS.checkpoint is a directory, a subdirectory will be created '
    'for each checkpoint evaluated.')
flags.DEFINE_integer(
    'output_num_shards', 1, 'Each replica generates one shard of output '
    'according to --output_shard_id.')
flags.DEFINE_integer(
    'output_shard_id', 0,
    'The output shard id in range [0, output_num_shards - 1].')

FLAGS = flags.FLAGS

_RETRY_SLEEP_SECONDS = 10


class PredictorRunnerBase:
  """Manages state for running predictor.

  Abstract class. Subclasses should override `InputGenerator` and `RunBatch`.
  Call `Subclass().Run()` in `main()` function to run.
  """

  def __init__(self,
               checkpoint,
               output_dir=None,
               inference_graph=None,
               inference_subgraph_name='',
               device_type='cpu',
               output_num_shards=1,
               output_shard_id=0,
               max_inputs=0,
               input_id_filter=None,
               tf_master='local',
               inference_threads=1,
               batch_size=64,
               prediction_step_interval=3000):
    """Constructor.

    Args:
      checkpoint: Either a checkpoint file to load, or a directory containing
        multiple checkpoints, where the latest checkpoint will be loaded.
      output_dir: Output directory. If `checkpoint` is a directory, a
        subdirectory will be created for each checkpoint evaluated.
      inference_graph: Path to an inference graph. If not specified, will be
        inferred from the checkpoint path.
      inference_subgraph_name: The name of the inference subgraph to use.
        Defaults to the default subgraph.
      device_type: Device type, either cpu, gpu, or tpu.
      output_num_shards: Each replica generates one shard of output according to
        `output_shard_id`.
      output_shard_id: The output shard id in range `[0, output_num_shards -
        1]`.
      max_inputs: Only process the first n inputs. 0 means process all inputs.
      input_id_filter: If not empty, only process the input ids in the given
        list.
      tf_master: tf_master for predictor session.
      inference_threads: Number of inference threads.
      batch_size: Batch size.
      prediction_step_interval: Number of steps between outputs. Only meaningful
        if `checkpoint` is a directory.
    """
    self._checkpoint = checkpoint
    self._output_dir = output_dir
    self._output_num_shards = output_num_shards
    self._output_shard_id = output_shard_id
    self._max_inputs = max_inputs
    input_id_filter = input_id_filter or []
    self._input_id_filter = [str(x) for x in input_id_filter]
    self._batch_size = batch_size
    self._prediction_step_interval = prediction_step_interval

    if device_type == 'tpu' and FLAGS.xla_device != 'tpu':
      raise ValueError('xla_device=tpu should be set with device_type=tpu!')

    if tf.io.gfile.isdir(self._checkpoint):
      initial_checkpoint = tf.train.latest_checkpoint(self._checkpoint)
      while (not initial_checkpoint or
             not tf.io.gfile.exists(initial_checkpoint + '.index')):
        tf.logging.log_first_n(tf.logging.INFO,
                                    'Waiting for checkpoint to be available.',
                                    10)
        time.sleep(_RETRY_SLEEP_SECONDS)
        initial_checkpoint = tf.train.latest_checkpoint(self._checkpoint)
    else:
      initial_checkpoint = self._checkpoint
      if not tf.io.gfile.exists(initial_checkpoint + '.index'):
        raise ValueError('Could not find checkpoint %s' % initial_checkpoint)

    # Use saved inference graph.
    if inference_graph:
      self._inference_graph = inference_graph
    else:
      checkpoint_dir = self._checkpoint
      if not tf.io.gfile.isdir(checkpoint_dir):
        checkpoint_dir = os.path.dirname(checkpoint_dir)
      logdir = os.path.dirname(checkpoint_dir)
      inference_graph_filename = 'inference.pbtxt'
      if device_type == 'tpu':
        inference_graph_filename = 'inference_tpu.pbtxt'
      self._inference_graph = os.path.join(logdir, 'inference_graphs',
                                           inference_graph_filename)
    self._predictor = predictor.Predictor(
        inference_graph=self._inference_graph,
        subgraph_name=inference_subgraph_name,
        checkpoint=initial_checkpoint,
        device_type=device_type,
        tf_master=tf_master)
    self._threadpool = concurrent.futures.ThreadPoolExecutor(inference_threads)
    self._locks = [threading.Lock() for _ in range(inference_threads)]

  @classmethod
  def FromFlags(cls, **kwargs):
    """Constructs an instance of this class from FLAGS."""
    return cls(
        checkpoint=FLAGS.checkpoint,
        output_dir=FLAGS.output_dir,
        inference_graph=FLAGS.inference_graph,
        inference_subgraph_name=FLAGS.inference_subgraph_name,
        device_type=FLAGS.device_type,
        output_num_shards=FLAGS.output_num_shards,
        output_shard_id=FLAGS.output_shard_id,
        max_inputs=FLAGS.max_inputs,
        input_id_filter=FLAGS.input_id_filter,
        tf_master=FLAGS.tf_master,
        inference_threads=FLAGS.inference_threads,
        batch_size=FLAGS.batch_size,
        prediction_step_interval=FLAGS.prediction_step_interval,
        **kwargs)

  def _ShouldProcessInputId(self, input_id):
    if self._max_inputs > 0 and input_id >= self._max_inputs:
      return False
    if self._input_id_filter and str(input_id) not in self._input_id_filter:
      return False
    return input_id % self._output_num_shards == self._output_shard_id

  def _OutputFilename(self, output_dir, name):
    assert self._output_shard_id >= 0
    assert self._output_shard_id < self._output_num_shards
    return '%s-%.5d-of-%.5d' % (os.path.join(
        output_dir, name), self._output_shard_id, self._output_num_shards)

  def InputGenerator(self):
    """Generator that yields the next input.

    Must yield in a deterministic order or raise an exception when
    self._output_num_shards > 1.
    """
    raise NotImplementedError('Abstract method.')

  def RunBatch(self, output_dir, batch):
    """Runs predictor on a single batch of data.

    Args:
      output_dir: the output directory.
      batch: a list of (input_id, element) pairs, where element is yielded from
        InputGenerator and input_id is a unique counter starting from 0.
    """
    raise NotImplementedError('Abstract method.')

  def _PredictOneCheckpoint(self, checkpoint, output_dir):
    """Runs predictor."""
    tf.logging.info('Processing checkpoint %s.', checkpoint)
    self._predictor.Load(checkpoint)

    def LockedRunBatch(batch, batch_id):
      """TPU inference runs the i-th batch on the i%num_cores-th core.

      Make sure that core is available before scheduling the next batch on it.

      Args:
        batch: The input to be passed to RunBatch.
        batch_id: The id of this batch, which determins which core it runs on.
      """
      with self._locks[batch_id % len(self._locks)]:
        self.RunBatch(output_dir, batch)

    batch_id = 0
    batch = []
    futures = []
    # Iterate through the input and process it one batch at a time.
    it = self.InputGenerator()
    if self._max_inputs > 0:
      it = itertools.islice(it, self._max_inputs)
    for next_id, element in enumerate(it):
      if self._ShouldProcessInputId(next_id):
        batch.append((next_id, element))
        if len(batch) == self._batch_size:
          futures.append(
              self._threadpool.submit(LockedRunBatch, batch, batch_id))
          batch_id += 1
          batch = []
    # Last batch.
    if batch:
      futures.append(self._threadpool.submit(LockedRunBatch, batch, batch_id))
    # Wait for completion.
    for f in futures:
      f.result()

  def _PredictContinuously(self):
    """Waits for new checkpoints and runs predictor continuously."""
    prev_step = -1000000
    while True:
      # TODO(jonathanasdf): how to determine when training finished?
      path = tf.train.latest_checkpoint(self._checkpoint)
      step_str = re.search(r'ckpt-(\d{8})', six.ensure_str(path)).group(1)
      step = int(step_str)
      if step - prev_step >= self._prediction_step_interval:
        if not self._output_dir:
          raise ValueError(
              'output_dir must be specified for _PredictContinuously.')
        output_dir = os.path.join(self._output_dir, 'step_' + step_str)
        tf.io.gfile.makedirs(output_dir)
        self._PredictOneCheckpoint(path, output_dir)
        prev_step = step
        tf.logging.info('Waiting for next checkpoint...')
      time.sleep(_RETRY_SLEEP_SECONDS)

  @py_utils.RetryOnTransientTfError()
  def Run(self):
    """Monitor checkpoints and runs predictor."""
    if self._output_dir:
      tf.io.gfile.makedirs(self._output_dir)
    if tf.io.gfile.isdir(self._checkpoint):
      self._PredictContinuously()
    else:
      self._PredictOneCheckpoint(self._checkpoint, self._output_dir)
