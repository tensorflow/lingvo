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

  Say: checkpoint_path contains a variable "var" with dtype tf.float32.

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


def quantization_noise_from_step_num():
  """A quantization noise equal to (phi * (step_num + 1)) mod 1.0.

  quantization_noise is a float32 scalar in [0, 1). quantization_noise should
  take different values across different steps, approximating a uniform
  distribution over [0, 1).

  In the case of replicated TPU training, quantization_noise should be identical
  across replicas in order to keep the parameters identical across replicas.
  The natural choice for quantization_noise would be tf.random_uniform(),
  but this is not possible for TPU, since there is currently no way to seed
  the different cores to produce identical values across replicas. Instead we
  use this quantization_noise_from_step_num() to generate noise.

  Returns:
    a float32 scalar
  """
  step = tf.to_int32(tf.train.get_or_create_global_step()) + 1
  phi = ((5**0.5) - 1) / 2
  # Naive computation tf.mod(phi * step, 1.0) in float32 would be disastrous
  # due to loss of precision when the step number gets large.
  # Computation in doubles does not work on TPU, so we use this complicated
  # alternative computation which does not suffer from these roundoff errors.
  ret = 0.0
  for i in xrange(30):
    ret += (((phi * (2**i)) % 1.0)  # double-precision computation in python
            * tf.to_float(tf.mod(step // (2**i), 2)))
  return tf.mod(ret, 1.0)


def randomized_roundoff_to_bfloat16(x, quantization_noise, cand1, cand2):
  """Round-off x to cand1 or to cand2 in an unbiased way.

  Cand1 and cand2 are the same shape as x.
  For every element of x, the corresponding elements of cand1 and cand2 should
  be the two closest bfloat16 values to x.  Order does not matter.
  cand1 and cand2 must differ from each other.

  Args:
    x: A float32 Tensor.
    quantization_noise: A Tensor broadcastable to the shape of x containing
      random uniform values in [0.0, 1.0].
    cand1: A bfloat16 Tensor the same shape as x.
    cand2: A bfloat16 Tensor the same shape as x.

  Returns:
    A bfloat16 Tensor.
  """
  cand1_f = tf.to_float(cand1)
  cand2_f = tf.to_float(cand2)
  step_size = cand2_f - cand1_f
  fpart = (x - cand1_f) / step_size
  ret = tf.where(tf.greater(fpart, quantization_noise), cand2, cand1)
  return ret


def to_bfloat16_unbiased(x, quantization_noise=None):
  """Convert a float32 to a bfloat16 using randomized roundoff.

  The current implementation uses quantization_noise_from_step_num to generate
  quantization_noise, which requires global_step, and is not deterministic.
  To use it for inference, it might be feasible to replace the noise generation
  function with a constant, e.g., 0.5.

  Args:
    x: A float32 Tensor.
    quantization_noise: A float, specifying the quantization noise.

  Returns:
    A bfloat16 Tensor, with the same shape as x.
  """
  if quantization_noise is None:
    quantization_noise = quantization_noise_from_step_num()
  x_sign = tf.sign(x)
  # Make sure x is positive.  If it is zero, the two candidates are identical.
  x = x * x_sign + 1e-30
  cand1 = tf.to_bfloat16(x)
  cand1_f = tf.to_float(cand1)
  # This relies on the fact that for a positive bfloat16 b,
  # b * 1.005 gives you the next higher bfloat16 and b*0.995 gives you the
  # next lower one. Both 1.005 and 0.995 are ballpark estimation.
  cand2 = tf.to_bfloat16(
      tf.where(tf.greater(x, cand1_f), cand1_f * 1.005, cand1_f * 0.995))
  ret = randomized_roundoff_to_bfloat16(x, quantization_noise, cand1, cand2)
  return ret * tf.to_bfloat16(x_sign)


class Bfloat16VariableSaveable(saver.BaseSaverBuilder.SaveableObject):
  """Saveable that loads Variables as bfloat16."""

  def __init__(self, var, orig_dtype, slice_spec, name):
    # TODO(rohananil): Investigate if we can avoid using a callable, instead
    # change the saveable api to make use of dtype passed in.
    def _make_callable_var():
      return var

    spec = saver.BaseSaverBuilder.SaveSpec(
        _make_callable_var, slice_spec, name, dtype=orig_dtype)
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
