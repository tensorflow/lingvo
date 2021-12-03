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
# ==============================================================================
"""Common flags."""

from lingvo import compat as tf
from lingvo.core import cluster_factory

tf.flags.DEFINE_string(
    'model', None, 'Name of the model class to train.'
    'Must be a model defined in the model_registry.')
tf.flags.DEFINE_string(
    'model_task_name', '', 'For multitask models: '
    'select task to train/evaluate/decode. '
    'Empty means to sample a task (training only).')
tf.flags.DEFINE_string('logdir', '', 'Log directory.')
tf.flags.DEFINE_string('job', '', 'trainer/controller/eval, etc.')
tf.flags.DEFINE_integer('task', 0, 'Task id within the job.')

tf.flags.DEFINE_string('tf_master', '', 'URL to tensorflow server.')

tf.flags.DEFINE_string('worker_job', '/job:trainer', 'Job name.')
tf.flags.DEFINE_list('additional_worker_jobs', [],
                     'Additional worker job names.')
tf.flags.DEFINE_integer('worker_tpus', 0, 'Number of tpus to use per replica.')
tf.flags.DEFINE_integer('worker_num_tpu_hosts', 0, 'Number of tpu hosts.')

tf.flags.DEFINE_string('evaler_job', '/job:evaler', 'Job name')
tf.flags.DEFINE_integer('evaler_replicas', 0, 'Number of replicas.')
tf.flags.DEFINE_integer('evaler_gpus', 0, 'Number of gpus to use per replica.')

tf.flags.DEFINE_string('decoder_job', '/job:decoder', 'Job name')
tf.flags.DEFINE_integer('decoder_replicas', 0, 'Number of replicas.')
tf.flags.DEFINE_integer('decoder_gpus', 0, 'Number of gpus to use per replica.')

tf.flags.DEFINE_integer(
    'enqueue_max_steps', None, 'Max enqueue steps. -1 meaning no limit.'
    ' This flag should be set for unit-test only.')

tf.flags.DEFINE_integer('saver_max_to_keep', None,
                        'Maximum number of recent checkpoints to keep.')
tf.flags.DEFINE_float('saver_keep_checkpoint_every_n_hours', None,
                      'How often to keep a checkpoint.')

tf.flags.DEFINE_bool('run_functions_eagerly', False,
                     'Whether to enable eager execution of `tf.function`s.')
tf.flags.DEFINE_bool(
    'write_v2_checkpoints', False,
    'Whether to write v2 object based checkpoints or v1 checkpoints.')

FLAGS = tf.flags.FLAGS


def UpdateClusterParamsFromFlags(p=None):
  """Returns cluster params based on flag settings."""
  if p is None:
    p = cluster_factory.Current().params.Copy()

  p.mode = 'sync'
  p.task = FLAGS.task
  p.logdir = FLAGS.logdir
  p.job = FLAGS.job
  if p.job.startswith('evaler_'):
    p.job = 'evaler'
  elif p.job.startswith('decoder_'):
    p.job = 'decoder'

  if FLAGS.xla_device == 'tpu':
    is_tpu = True
  else:
    assert FLAGS.xla_device in ('', 'cpu')
    is_tpu = False

  p.enable_asserts = not is_tpu
  p.enable_check_numerics = not is_tpu
  FLAGS.enable_asserts = not is_tpu
  FLAGS.enable_check_numerics = not is_tpu
  FLAGS.tpu_compatible = is_tpu

  p.worker.replicas = 1
  p.worker.devices_per_split = 1
  if is_tpu:
    p.worker.name = FLAGS.worker_job
    p.worker.tpus_per_replica = FLAGS.worker_tpus
    p.worker.num_tpu_hosts = FLAGS.worker_num_tpu_hosts
  else:
    p.worker.name = '/job:localhost'
    p.worker.tpus_per_replica = 0
    p.worker.num_tpu_hosts = 0

  p.evaler.replicas = FLAGS.evaler_replicas
  p.evaler.gpus_per_replica = FLAGS.evaler_gpus
  p.decoder.replicas = FLAGS.decoder_replicas
  p.decoder.gpus_per_replica = FLAGS.decoder_gpus
  return p
