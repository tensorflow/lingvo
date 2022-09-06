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
    'enable_tf_data_debug_mode', False,
    'Whether to enable debug mode in tf.data. Only works in tf2 eager mode.')
