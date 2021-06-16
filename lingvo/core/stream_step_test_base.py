# Lint as: python3
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
"""A base class for test StreamStep() of related layers.

As of Jun 2021 it's written for conformer (https://arxiv.org/abs/2005.08100)
related layers.
"""

import math
from absl.testing import parameterized
from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.core import test_utils
import numpy as np


class StreamStepTestBase(test_utils.TestCase, parameterized.TestCase):
  """Common base for testing StreamStep().

  Sub classes will need to define the following functions:

  input_rank()
  _GetParams()
  _FProp()
  _GetFPropOutput()
  """

  @property
  def input_rank(self):
    return 3

  def _GetInputs(self, batch_size, max_seqlen, input_dim, full_seq=False):
    # Prepares inputs.
    np.random.seed(None)
    if self.input_rank == 3:
      inputs = np.random.normal(
          0.5, 1, [batch_size, max_seqlen, input_dim]).astype(np.float32)
    else:
      assert self.input_rank == 4
      inputs = np.random.normal(
          0.5, 1, [batch_size, max_seqlen, 1, input_dim]).astype(np.float32)
    print(f'np.sum(inputs): {np.sum(inputs)}')
    inputs = tf.convert_to_tensor(inputs)

    if not full_seq:
      seqlen = np.random.randint(
          low=max_seqlen // 2,
          high=max_seqlen + 1,
          size=(batch_size,),
          dtype=np.int32)
    else:
      seqlen = np.full((batch_size,), max_seqlen, dtype=np.int32)
    print(f'seqlen: {seqlen}')

    seqlen = tf.convert_to_tensor(seqlen)
    paddings = py_utils.PaddingsFromLengths(seqlen, max_seqlen)
    return inputs, paddings

  def _PadInput(self, inputs, paddings, num_frames):
    if self.input_rank == 3:
      inputs = tf.pad(inputs, [[0, 0], [0, num_frames], [0, 0]])
    else:
      inputs = tf.pad(inputs, [[0, 0], [0, num_frames], [0, 0], [0, 0]])
    paddings = tf.pad(paddings, [[0, 0], [0, num_frames]], constant_values=1)
    return inputs, paddings

  def _NormalizeStreamStepOutput(self,
                                 outputs,
                                 paddings,
                                 right_context,
                                 max_seqlen,
                                 num_layers=1):
    # outputs has right_context * num_layers-frames delay from inputs.
    outputs = outputs[:, right_context * num_layers:]
    # later outputs corresponds to padded inputs to complete the last frame's
    # right context.
    outputs = outputs[:, :max_seqlen]
    out_rank = py_utils.GetRank(outputs)
    paddings = paddings[:, :max_seqlen]
    return outputs * py_utils.AppendDims(1. - paddings, out_rank - 2)

  def _GetParams(self, **kwargs):
    """Returns layer params."""
    raise NotImplementedError()

  def _FProp(self, layer, inputs, paddings):
    """Returns layer fprop results."""
    raise NotImplementedError()

  def _GetFPropOutput(self, fprop_out):
    """Returns key layer output (as opposed to other outputs, e.g. paddings)."""
    raise NotImplementedError()

  def _TestStreamStepHelper(self, **kwargs):
    """Main helper method."""
    batch_size, max_seqlen, input_dim = 2, 32, kwargs['input_dim']

    stride = kwargs.get('stride', 1)
    # max_seqlen is divisible by stride.
    assert max_seqlen % stride == 0

    right_context = kwargs.get('right_context', 0)

    # Prepares inputs.
    inputs, paddings = self._GetInputs(batch_size, max_seqlen, input_dim)

    # Gets params
    p = self._GetParams(**kwargs)

    # Builds graph.
    with self.session(use_gpu=False) as sess:
      l = p.Instantiate()
      init_op = tf.global_variables_initializer()

      fprop_out = self._FProp(l, inputs, paddings)
      base_outputs = self._GetFPropOutput(fprop_out)
      out_rank = py_utils.GetRank(base_outputs)
      base_outputs *= py_utils.AppendDims(1. - paddings, out_rank - 2)

      try:
        state = l.zero_state(batch_size)
      except TypeError:
        state = l.zero_state(l.theta, batch_size)
      outputs = []
      for i in range(max_seqlen // stride +
                     int(math.ceil(right_context / stride))):
        if i < max_seqlen // stride:
          step_inputs = inputs[:, stride * i:stride * (i + 1)]
          step_paddings = paddings[:, stride * i:stride * (i + 1)]
        else:
          step_inputs = tf.zeros_like(inputs[:, 0:stride])
          step_paddings = tf.ones_like(paddings[:, 0:stride])
        output, _, state = l.StreamStep(l.theta, step_inputs, step_paddings,
                                        state)
        outputs.append(output)

      outputs = tf.concat(outputs, axis=1)
      outputs = self._NormalizeStreamStepOutput(outputs, paddings,
                                                right_context, max_seqlen)

      sess.run(init_op)

      expected, actual = sess.run([base_outputs, outputs])
      print(f'expected: {repr(expected)}, {expected.shape}')
      print(f'actual: {repr(actual)}, {actual.shape}')
      print(f'np.sum(np.abs(expected)): {np.sum(np.abs(expected))}')
      print(f'np.sum(np.abs(actual)): {np.sum(np.abs(actual))}')
      tol = kwargs.get('tol', 1e-6)
      self.assertAllClose(expected, actual, atol=tol, rtol=tol)

  def _BuildStackingBaseGraph(self, layers, num_layers, inputs, paddings):
    outputs = inputs
    for l in layers:
      fprop_out = self._FProp(l, outputs, paddings)
      outputs = self._GetFPropOutput(fprop_out)
    # [b, t, -1]
    outputs *= tf.expand_dims(1. - paddings, -1)
    return outputs

  def _BuildStackingStreamGraph(self, layers, num_layers, inputs, paddings,
                                stride, right_context):
    batch_size, max_seqlen, dim = py_utils.GetShape(inputs)
    assert max_seqlen % stride == 0
    states = [l.zero_state(batch_size) for l in layers]

    outputs = []
    for i in range(
        int(math.ceil((max_seqlen + right_context * num_layers) / stride))):
      if i < max_seqlen // stride:
        step_inputs = inputs[:, stride * i:stride * (i + 1)]
        step_paddings = paddings[:, stride * i:stride * (i + 1)]
      else:
        step_inputs = tf.zeros([batch_size, stride, dim])
        step_paddings = tf.ones([batch_size, stride])

      output, out_paddings = step_inputs, step_paddings
      new_states = []
      for l, state0 in zip(layers, states):
        output, out_paddings, state1 = l.StreamStep(l.theta, output,
                                                    out_paddings, state0)
        new_states.append(state1)
      states = new_states
      outputs.append(output)

    outputs = tf.concat(outputs, axis=1)
    outputs = self._NormalizeStreamStepOutput(outputs, paddings, right_context,
                                              max_seqlen, num_layers)
    return outputs

  def _TestRightContextStackingLayersHelper(self, **kwargs):
    """Applicable only if the layer implements StreamStep() with right context."""
    batch_size, max_seqlen, input_dim = 2, 32, kwargs['input_dim']

    stride = kwargs['stride']
    num_layers = kwargs['num_layers']
    right_context = kwargs.get('right_context', 0)

    assert max_seqlen % stride == 0

    # Prepares inputs.
    inputs, paddings = self._GetInputs(batch_size, max_seqlen, input_dim)

    # Gets params.
    p = self._GetParams(**kwargs)
    ps = [p.Copy().Set(name=f'base{i}') for i in range(num_layers)]

    # Builds graphs.
    layers = [x.Instantiate() for x in ps]
    base_outputs = self._BuildStackingBaseGraph(layers, num_layers, inputs,
                                                paddings)

    outputs = self._BuildStackingStreamGraph(layers, num_layers, inputs,
                                             paddings, stride, right_context)

    init_op = tf.global_variables_initializer()
    with self.session(use_gpu=False) as sess:
      sess.run(init_op)

      expected, actual = sess.run([base_outputs, outputs])
      print(f'expected: {repr(expected)}, {expected.shape}')
      print(f'actual: {repr(actual)}, {actual.shape}')
      print(f'np.sum(np.abs(expected)): {np.sum(np.abs(expected))}')
      print(f'np.sum(np.abs(actual)): {np.sum(np.abs(actual))}')
      self.assertAllClose(expected, actual, atol=5e-5)
      self.assertEqual(
          tuple(expected.shape), (batch_size, max_seqlen, input_dim))
