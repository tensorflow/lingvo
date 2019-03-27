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
"""Various methods for bfloat16 training & inference.

Bfloat16VariableSaveable: Saveable that restores variable into bfloat16 type.

Usage:

  Given a checkpoint_path with a variable of type tf.float32, this particular
  saveable allows restore them as tf.bfloat16. This is specifically useful for
  inference.

  Say: checkpoint_path contains a variable "var" with dtype tf.float32::

      variable_name = "var"
      original_dtype =  tf.float32

      bfloat16_var = tf.Variable(
          0.0, name=variable_name, dtype=tf.bfloat16, use_resource=True)

      saveable = bfloat16_variables.Bfloat16VariableSaveable(
          bfloat16_var, original_dtype, slice_spec, variable_name)
      saver = tf.train.Saver(
          {variable_name: saveable}, restore_sequentially=True)
      saver.restore(sess, checkpoint_path)

      # bfloat16_var is now loaded from the checkpoint.
"""
import tensorflow as tf
from tensorflow.python.training import saver


class Bfloat16VariableSaveable(saver.BaseSaverBuilder.SaveableObject):
  """Saveable that loads Variables as bfloat16."""

  def __init__(self, var, orig_dtype, slice_spec, name):
    # TODO(rohananil): Investigate if we can avoid using a callable, instead
    # change the saveable api to make use of dtype passed in.
    def _make_callable_var():
      return var

    spec = saver.BaseSaverBuilder.SaveSpec(
        _make_callable_var,
        slice_spec,
        name,
        dtype=orig_dtype,
        device=var.device)
    super(Bfloat16VariableSaveable, self).__init__(var, [spec], name)

  def restore(self, restored_tensors, restored_shapes):
    restored_tensor = restored_tensors[0]
    if restored_shapes is not None:
      restored_tensor = tf.reshape(restored_tensor, restored_shapes[0])
    return tf.assign(
        self.op,
        tf.cast(restored_tensor, tf.bfloat16),
        validate_shape=restored_shapes is None and
        self.op.get_shape().is_fully_defined())


def get_saver_spec_for_variables_with_bf16_overrides(variables_to_restore):
  """Returns a dictionary containing overrides to load variables as bf16.

  Args:
    variables_to_restore: A mapping from variable to name (on checkpoint) to the
      Variable object.

  Returns:
    A saver dictionary which can be used to load from checkpoints.
  """
  saver_dict = {}
  for var_name, v in variables_to_restore.items():
    if v.dtype == tf.bfloat16:
      # TODO(rohananil): Add support for PartitionedVariables if there is
      # demand.
      savable = Bfloat16VariableSaveable(v, tf.float32, '', var_name)
      saver_dict[var_name] = savable
    else:
      saver_dict[var_name] = v
  return saver_dict
