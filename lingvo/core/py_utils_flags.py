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
"""Common utilities."""

# ==============================================================================
# Note: Avoid adding dependencies to py_utils beyond standard python packages
#       and tensorflow.
# ==============================================================================


import lingvo.compat as tf

from lingvo.core import cluster_factory

tf.flags.DEFINE_bool('enable_asserts', True,
                     'If False, we disable all asserts.')

tf.flags.DEFINE_bool('enable_check_numerics', True,
                     'If False, we bypass calls to CheckNumerics.')

tf.flags.DEFINE_bool('print_debug_tensors', False,
                     'Whether to print debug tensors.')

tf.flags.DEFINE_bool(
    'testonly_skip_norm_layers', False,
    'Disable normalization layers, used for checking goldens '
    'in unittests. Normalizations make differences harder to '
    'catch.')

tf.flags.DEFINE_string(
    'xla_device', '', 'If non-empty, can be cpu, gpu, or tpu (case sensitive)')

tf.flags.DEFINE_bool(
    'tpu_compatible', False, 'Create variables in a way compatible with TPU. '
    'This should be true for any job that will interact '
    'with variables or a checkpoint that will be produced '
    'or consumed by TPU')

tf.flags.DEFINE_bool(
    'tflite_compatible', False,
    'Uses tflite converter-friendly ops at applicable places. This so far '
    '(08/2020) is a only best-effort option.')

tf.flags.DEFINE_bool(
    'pin_vars_to_cpu', False,
    'Pin variables to cpu:0.  This is useful for weight-sharing / multi-core '
    'inference on TPUs in which TPU core variables are managed via '
    'TPUPartitionedCallOp.')

tf.flags.DEFINE_bool('disable_py_utils_debug', False,
                     'If True disables all py_utils.Debug() logs.')

tf.flags.DEFINE_bool(
    'stateless_vars_init', False,
    'Use stateless TensorFlow random number generators (RNG) (e.g. '
    'tf.random.stateless_uniform) to initialize variables instead of the '
    'default ones (e.g. tf.random.uniform). This is useful to make variable '
    'initialization deterministic on different replicas such as on TPUs, '
    'since XLA does not fully respect the contract with respect to '
    'user-specified seeds, when using TensorFlow stateful RNGs.')

# NOTE: Using absl flags in libraries are frowned upon for several reasons:
#
# 1) They require app.run() or explicit flag parsing, preventing the use of
# these libraries in environments that don't look like normal binaries (colab
# notebooks).
#
# 2) They are process-level globals that cannot be scoped or configured except
# once during binary startup.
#
# Because py_utils is a library, no more flags should be used in this file; the
# existing flags are present for backwards compatibility.  Instead, consider
# using a stack-scoped configuration object such as the Cluster object. We guard
# against issue 1 above by using _FromGlobal below, which uses the default value
# of the FLAG even if flags are unparsed.

FLAGS = tf.flags.FLAGS


# pylint: disable=invalid-name
@tf.autograph.experimental.do_not_convert
def _FromGlobal(field_name, allow_override_from_cluster=False):
  """Get 'field_name' from a global configuration object.

  Currently the global configuration object used is FLAGS, but this may
  change to Cluster() or an equivalent stack-scoped config object.

  Args:
    field_name: The string field name to look up.
    allow_override_from_cluster: Allow the Cluster() to override FLAGS.

  Returns:
    The value associated with the global configuration string 'field_name'.
  """

  if allow_override_from_cluster:
    cluster = cluster_factory.Current()
    if field_name in cluster.params:
      params_value = cluster.params.Get(field_name)
      # Return the value in the cluster params if it is not None
      if params_value is not None:
        return params_value

  # Now check the FLAGS object for backwards compatibility.
  #
  # If not explicitly set, get the field from the FLAGS object.  If FLAGS
  # have not been parsed yet, the default value of the flag will be used.
  return FLAGS[field_name].value
# pylint: enable=invalid-name


def enable_asserts():  # pylint: disable=invalid-name
  res = _FromGlobal('enable_asserts', allow_override_from_cluster=True)
  assert res in [True, False]
  return res


def enable_check_numerics():  # pylint: disable=invalid-name
  res = _FromGlobal('enable_check_numerics', allow_override_from_cluster=True)
  assert res in [True, False]
  return res


def use_xla():  # pylint: disable=invalid-name
  res = _FromGlobal('xla_device', allow_override_from_cluster=True)
  if res:
    assert res in ('', 'cpu', 'gpu', 'tpu')
  return res


def use_tpu():  # pylint: disable=invalid-name
  res = _FromGlobal('xla_device', allow_override_from_cluster=True) == 'tpu'
  if res:
    assert not enable_asserts()  # asserts not supported on tpu
  return res


def testonly_skip_norm_layers():  # pylint: disable=invalid-name
  return _FromGlobal('testonly_skip_norm_layers')


def tpu_compat():  # pylint: disable=invalid-name
  return use_tpu() or _FromGlobal('tpu_compatible')


def use_stateless_vars_init():  # pylint: disable=invalid-name
  return _FromGlobal('stateless_vars_init')
