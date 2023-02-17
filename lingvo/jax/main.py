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
r"""Main file for running a JAX training and evaluation loop.

Example usage:
$ bazel run -c opt \
    lingvo/jax:main -- \
    --model=lm.ptb.PTBCharTransformerSmallSgd \
    --job_log_dir=/tmp/jax_log_dir/exp01 --alsologtostderr
"""

import os
import random
import re
import time
from typing import Optional, Sequence

from absl import app
from absl import flags
from absl import logging
# Required import to setup work units when running through XManager.
from clu import platform
import jax
from lingvo.jax import eval as eval_lib
from lingvo.jax import py_utils
from lingvo.jax import train
import tensorflow.compat.v2 as tf


FLAGS = flags.FLAGS

flags.DEFINE_string('model', None, 'Lingvo Jax model name.')
flags.DEFINE_string('job_log_dir', None,
                    'Directory where all experiment assets will be stored.')
flags.DEFINE_enum('mode', 'train', ['train', 'eval', 'decode', 'decode_once'],
                  'Flag to control which job is called.')
flags.DEFINE_bool(
    'eval_on_test', False, 'If True, then the training loop '
    'includes a full evaluation on all the test set splits. '
    'This can be set to True if we do not want an additional job '
    'to run continuous eval.')
flags.DEFINE_bool(
    'multi_host_checkpointing', False,
    'Whether to use multi-host checkpointing or not. Only useful for '
    'multi-host SPMD models.')
flags.DEFINE_bool(
    'maybe_use_persistence_checkpointing', False,
    'If suitable, will try to rely on persistence-based checkpointing rather '
    'than Flax-based checkpointing for SPMD models.')
flags.DEFINE_string(
    'checkpoint_todelete_subdir', None,
    'If set, checkpoints to be deleted will be only renamed into a '
    'subdirectory with the provided string. Otherwise, they will be directly '
    'deleted from the file system. Useful if checkpoint deletion is time '
    'consuming. By default, delete the checkpoint assets.')
flags.DEFINE_string(
    'restore_checkpoint_dir', None,
    'If set, the directory from which to restore checkpoint. If unset, the '
    'script tries to restore from --job_log_dir\'s `checkpoints` subdirectory.')
flags.DEFINE_integer(
    'restore_checkpoint_step', None,
    'If set, the checkpoint step to restore. If unset, default to the latest '
    'checkpoint.')
flags.DEFINE_bool(
    'globally_use_hardware_rng', True,
    'Whether to globally use fast hardware RNG. Deterministic only at the '
    'same compiler version and with the same sharding')
flags.DEFINE_integer(
    'jax_profiler_port', None,
    'If set, the jax.profiler port to use. Only needed for profiling in open source.'
)
# Flag --jax_parallel_functions_output_gda is available through JAX.
# Flags --jax_backend_target and --jax_xla_backend are available through JAX.


def setup_jax(globally_use_hardware_rng: bool, jax_use_gda: bool,
              jax_backend_target: Optional[str],
              jax_xla_backend: Optional[str]) -> None:
  """Setups JAX and logs information about this job."""
  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')
  if globally_use_hardware_rng:
    py_utils.set_globally_use_rbg_prng_key()

  # We use xmap only with SPMD.
  jax.config.update('experimental_xmap_spmd_lowering', True)
  # Use the manual partitioning lowering of xmap to avoid vectorization.
  jax.config.update('experimental_xmap_spmd_lowering_manual', True)

  if jax_use_gda:
    logging.info('Using JAX GSDA for pjit and checkpointing')

  if jax_backend_target:
    logging.info('Using JAX backend target %s', jax_backend_target)
    jax_xla_backend = 'None' if jax_xla_backend is None else jax_xla_backend
    logging.info('Using JAX XLA backend %s', jax_xla_backend)

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX devices: %r', jax.devices())
  logging.info('jax.device_count(): %d', jax.device_count())
  logging.info('jax.local_device_count(): %d', jax.local_device_count())
  logging.info('jax.process_count(): %d', jax.process_count())


def wait_with_random_jitter(min_secs: int, max_secs: int) -> None:
  """Sleeps for a random short interval to avoid thundering herd RPC calls."""
  time.sleep(random.randint(min_secs, max_secs))


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  setup_jax(FLAGS.globally_use_hardware_rng,
            FLAGS.jax_parallel_functions_output_gda, FLAGS.jax_backend_target,
            FLAGS.jax_xla_backend)

  # Add a note so that we can tell which Borg task is which JAX host.
  # (Borg task 0 is not guaranteed to be host 0)
  if jax.process_count() > 128:
    wait_with_random_jitter(min_secs=0, max_secs=60)
  work_unit = platform.work_unit()
  work_unit.set_task_status(f'process_index: {jax.process_index()}, '
                            f'process_count: {jax.process_count()}')
  work_unit.create_artifact(platform.ArtifactType.DIRECTORY, FLAGS.job_log_dir,
                            'job_log_dir')

  # Start jax.profiler for Tensorboard and profiling in open source.
  if FLAGS.jax_profiler_port is not None:
    server = jax.profiler.start_server(FLAGS.jax_profiler_port)  # pylint:disable=unused-variable
  if FLAGS.mode == 'train':
    train.train_and_evaluate(
        model_name=FLAGS.model,
        job_log_dir=FLAGS.job_log_dir,
        multi_host_checkpointing=FLAGS.multi_host_checkpointing,
        maybe_use_persistence_checkpointing=FLAGS
        .maybe_use_persistence_checkpointing,
        restore_checkpoint_dir=FLAGS.restore_checkpoint_dir,
        restore_checkpoint_step=FLAGS.restore_checkpoint_step,
        eval_on_test=FLAGS.eval_on_test,
        checkpoint_todelete_subdir=FLAGS.checkpoint_todelete_subdir)
  elif FLAGS.mode == 'eval':
    eval_lib.evaluate(
        model_name=FLAGS.model,
        job_log_dir=FLAGS.job_log_dir,
        multi_host_checkpointing=FLAGS.multi_host_checkpointing,
        maybe_use_persistence_checkpointing=FLAGS
        .maybe_use_persistence_checkpointing)
  elif FLAGS.mode == 'decode':
    eval_lib.decode(
        model_name=FLAGS.model,
        job_log_dir=FLAGS.job_log_dir,
        multi_host_checkpointing=FLAGS.multi_host_checkpointing,
        maybe_use_persistence_checkpointing=FLAGS
        .maybe_use_persistence_checkpointing,
        restore_checkpoint_dir=None,
        restore_checkpoint_step=None,
        continuous_decode=True,
    )
  elif FLAGS.mode == 'decode_once':
    if not FLAGS.restore_checkpoint_dir:
      raise ValueError('--mode=decode_once requires --restore_checkpoint_dir.')
    eval_lib.decode(
        model_name=FLAGS.model,
        job_log_dir=FLAGS.job_log_dir,
        multi_host_checkpointing=FLAGS.multi_host_checkpointing,
        maybe_use_persistence_checkpointing=FLAGS
        .maybe_use_persistence_checkpointing,
        restore_checkpoint_dir=FLAGS.restore_checkpoint_dir,
        restore_checkpoint_step=FLAGS.restore_checkpoint_step,
        continuous_decode=False,
    )


_TASK_HANDLE_RE = re.compile(r'(?:logs\.)?(\d+)\.(.*)\.([^.]+)\.\d+')


if __name__ == '__main__':
  # Only dump from Borg task 0.
  if 'BORG_TASK_HANDLE' in os.environ:
    handle = os.getenv('BORG_TASK_HANDLE')
    task_id, _, _ = _TASK_HANDLE_RE.match(handle).groups()  # pytype: disable=attribute-error  # re-none
    if int(task_id) == 0:
      dump_dir = os.getenv('XLA_DUMP_TO')
      if dump_dir:
        os.environ['XLA_FLAGS'] = f'--xla_dump_to={dump_dir}'

  # Provide access to --jax_backend_target and --jax_xla_backend flags.
  jax.config.config_with_absl()

  # TODO(shafey): Make `job_log_dir` mandatory?
  flags.mark_flags_as_required(['model'])

  @flags.multi_flags_validator(
      ['multi_host_checkpointing', 'maybe_use_persistence_checkpointing'],
      message='Multi-host checkpointing only supported with Flax checkpointing.'
  )
  def _validate_checkpoints(flags_dict):
    if (flags_dict['multi_host_checkpointing'] and
        flags_dict['maybe_use_persistence_checkpointing']):
      return False
    return True

  app.run(main)
