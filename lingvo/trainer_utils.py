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

tf.flags.DEFINE_string('worker_job', '/job:trainer', 'Job name.')
tf.flags.DEFINE_list('additional_worker_jobs', [],
                     'Additional worker job names.')
tf.flags.DEFINE_integer('worker_tpus', 0, 'Number of tpus to use per replica.')
tf.flags.DEFINE_integer('worker_num_tpu_hosts', 0, 'Number of tpu hosts.')
