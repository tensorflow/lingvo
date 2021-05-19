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
"""Common utility functions for generating summaries."""

import re
import time
import lingvo.compat as tf
from lingvo.core import base_input_generator
from lingvo.core import cluster_factory
from lingvo.core import plot
from lingvo.core import py_utils
import numpy as np


def _ShouldAddSummary():
  return cluster_factory.Current().add_summary


def scalar(*args, **kwargs):  # pylint: disable=invalid-name
  if _ShouldAddSummary():
    return tf.summary.scalar(*args, **kwargs)


def scalar_input_stats(*args, **kwargs):  # pylint: disable=invalid-name
  collections = kwargs.pop('collections', []) + [
      base_input_generator.INPUT_DATA_STATS_SUMMARIES_COLLECTION
  ]
  return tf.summary.scalar(*args, **kwargs, collections=collections)


def histogram(*args, **kwargs):  # pylint: disable=invalid-name
  if _ShouldAddSummary():
    return tf.summary.histogram(*args, **kwargs)


def image(*args, **kwargs):  # pylint: disable=invalid-name
  if _ShouldAddSummary():
    return tf.summary.image(*args, **kwargs)


def text(*args, **kwargs):  # pylint: disable=invalid-name
  if _ShouldAddSummary():
    return tf.summary.text(*args, **kwargs)


def scalar_v2(*args, **kwargs):  # pylint: disable=invalid-name
  if _ShouldAddSummary():
    if len(args) <= 2 and 'step' not in kwargs:
      kwargs['step'] = py_utils.GetGlobalStep()
    tf.compat.v2.summary.scalar(*args, **kwargs)


def histogram_v2(*args, **kwargs):  # pylint: disable=invalid-name
  if _ShouldAddSummary():
    if len(args) <= 2 and 'step' not in kwargs:
      kwargs['step'] = py_utils.GetGlobalStep()
    tf.compat.v2.summary.histogram(*args, **kwargs)


def image_v2(*args, **kwargs):  # pylint: disable=invalid-name
  if _ShouldAddSummary():
    if len(args) <= 2 and 'step' not in kwargs:
      kwargs['step'] = py_utils.GetGlobalStep()
    tf.compat.v2.summary.image(*args, **kwargs)


def text_v2(*args, **kwargs):  # pylint: disable=invalid-name
  if _ShouldAddSummary():
    if len(args) <= 2 and 'step' not in kwargs:
      kwargs['step'] = py_utils.GetGlobalStep()
    tf.compat.v2.summary.text(*args, **kwargs)


def SequenceLength(padding):
  """Computes the length of a sequence based on binary padding.

  Args:
    padding: A tensor of binary paddings shaped [batch, seqlen].

  Returns:
    seq_lens, A tensor of shape [batch] containing the non-padded length of each
      element of plot_tensor along the batch dimension.
  """
  seq_lens = tf.cast(tf.round(tf.reduce_sum(1 - padding, axis=1)), tf.int32)
  # Get rid of any extra dimensions.
  batch_size = tf.shape(padding)[0]
  seq_lens = tf.reshape(seq_lens, [batch_size], name='seq_lens')
  return seq_lens


def TrimPaddingAndPlotSequence(fig, axes, seq_matrix, seq_len, **kwargs):
  """Trims the time axis of seq_matrix with shape (dim, time) and plots it.

  For use as a plot function with MatplotlibFigureSummary.

  Args:
    fig:  A matplotlib figure handle.
    axes:  A matplotlib axes handle.
    seq_matrix:  A 2D ndarray shaped (num_rows, time).
    seq_len:  Integer length to use to trim the time axis of seq_matrix.
    **kwargs:  Additional keyword args to pass to plot.AddImage.
  """
  plot.AddImage(fig, axes, seq_matrix[:, :seq_len], **kwargs)


def TrimPaddingAndPlotAttention(fig,
                                axes,
                                atten_matrix,
                                src_len,
                                tgt_len,
                                transcript=None,
                                **kwargs):
  """Trims axes of atten_matrix with shape (tgt_time, src_time) and plots it.

  For use as a plot function with MatplotlibFigureSummary.

  Args:
    fig:  A matplotlib figure handle.
    axes:  A matplotlib axes handle.
    atten_matrix:  A 2D ndarray shaped (tgt_time, src_time).
    src_len:  Integer length to use to trim the src_time axis of atten_matrix.
    tgt_len:  Integer length to use to trim the tgt_time axis of atten_matrix.
    transcript: transcript for the target sequence.
    **kwargs:  Additional keyword args to pass to plot.AddImage.
  """
  plot.AddImage(
      fig, axes, atten_matrix[:tgt_len, :src_len], clim=(0, 1), **kwargs)
  if transcript is not None:
    if isinstance(transcript, np.ndarray):
      transcript = ' '.join(transcript[:src_len])
    axes.set_xlabel(plot.ToUnicode(transcript), size='x-small', wrap=True)


def AddAttentionSummary(name,
                        attention_tensors,
                        src_paddings,
                        tgt_paddings,
                        transcripts=None,
                        max_outputs=3):
  """Adds an image summary showing the attention probability matrix and state.

  Tensors are in sequence tensor format with the batch dimension in axis 1.

  Args:
    name: Summary name.
    attention_tensors: A list of 3D tensors shaped [target_len, batch_size,
      source_len] where attention[i, j, k] is the probability for the i-th
      output attending to the k-th input for element j in the batch.
    src_paddings: A tensor of binary paddings shaped [source_len, batch] for the
      source sequence. Or a list of tensors of the same length as
      attention_tensors with a separate paddings for each entry in
      attention_tensors.
    tgt_paddings: A tensor of binary paddings shaped [target_len, batch] for the
      target sequence. Or a list of tensors of the same length as
      attention_tensors with a separate paddings for each entry in
      attention_tensors.
    transcripts: Optional, transcripts shaped [batch, source_len] for the source
      sequence.
    max_outputs: Integer maximum number of elements of the batch to plot.
  """

  def Transpose(paddings):
    paddings = paddings if isinstance(paddings, list) else [paddings]
    return [tf.transpose(p) for p in paddings]

  AddAttentionSummaryBatchMajor(
      name, [tf.transpose(a, [1, 0, 2]) for a in attention_tensors],
      Transpose(src_paddings), Transpose(tgt_paddings), transcripts,
      max_outputs)


def AddAttentionSummaryBatchMajor(name,
                                  attention_tensors,
                                  src_paddings,
                                  tgt_paddings,
                                  transcripts=None,
                                  max_outputs=3):
  """Adds an image summary showing the attention probability matrix and state.

  As opposed to AddAttentionSummary() takes all tensors with batch dimension in
  axis 0.

  Args:
    name: Summary name.
    attention_tensors: A list of 3D tensors shaped [batch_size, target_len,
      source_len] where attention[b, i, j] is the probability for the i-th
      output attending to the j-th input for element b in the batch.
    src_paddings: A tensor of binary paddings shaped [batch, source_len] for the
      source sequence. Or a list of tensors of the same length as
      attention_tensors with a separate paddings for each entry in
      attention_tensors.
    tgt_paddings: A tensor of binary paddings shaped [batch, target_len] for the
      target sequence. Or a list of tensors of the same length as
      attention_tensors with a separate paddings for each entry in
      attention_tensors.
    transcripts: Optional, transcripts shaped [batch, source_len] for the source
      sequence.
    max_outputs: Integer maximum number of elements of the batch to plot.
  """
  def VerifyLen(paddings):
    length = len(paddings) if isinstance(paddings, list) else 1
    if length != 1 and length != len(attention_tensors):
      raise ValueError('Bad length of paddings list {}'.format(length))

  VerifyLen(src_paddings)
  VerifyLen(tgt_paddings)

  # Verify shapes.
  for i, attention_tensor in enumerate(attention_tensors):
    src, tgt = src_paddings, tgt_paddings
    src = src[0 if len(src) == 1 else i] if isinstance(src, list) else src
    tgt = tgt[0 if len(tgt) == 1 else i] if isinstance(tgt, list) else tgt
    tgt_shape = py_utils.GetShape(tgt)
    attention_tensors[i] = tf.identity(
        py_utils.with_dependencies([
            py_utils.assert_equal(
                py_utils.GetShape(attention_tensor),
                tgt_shape[:2] + [py_utils.GetShape(src)[1]] + tgt_shape[2:])
        ], attention_tensor),
        re.sub(':.*$', '', GetTensorName(attention_tensor, name, i)))

  if not _ShouldAddSummary():
    return

  def ToLengths(paddings):
    paddings = paddings if isinstance(paddings, list) else [paddings]
    return [SequenceLength(p) for p in paddings]

  def Get(lengths, i):
    return lengths[0 if len(lengths) == 1 else i]

  src_lens = ToLengths(src_paddings)
  tgt_lens = ToLengths(tgt_paddings)

  with plot.MatplotlibFigureSummary(
      name + '/Attention',
      max_outputs=max_outputs,
      gridspec_kwargs={'hspace': 0.3}) as fig:
    for n, atten in enumerate(attention_tensors):
      # Diagnostic metric that decreases as attention picks up.
      max_entropy = tf.math.log(tf.cast(Get(src_lens, n), tf.float32))
      max_entropy = tf.expand_dims(tf.expand_dims(max_entropy, -1), -1)
      atten_normalized_entropy = -atten * tf.math.log(atten +
                                                      1e-10) / max_entropy
      scalar(name + '/Attention/average_normalized_entropy/%d' % n,
             tf.reduce_mean(atten_normalized_entropy))
      args = [atten, Get(src_lens, n), Get(tgt_lens, n)]
      if transcripts is not None and n == 0:
        args.append(transcripts)
      fig.AddSubplot(
          args,
          TrimPaddingAndPlotAttention,
          title=GetTensorName(atten, name, n),
          xlabel='Input',
          ylabel='Output')


def AddNormSummary(name, vs_gs):
  """"Returns and creates summary for norms of vs and their gradients gs.

  Args:
    name: A name string for summary.
    vs_gs: A `.NestedMap` or a list of `.NestedMap` of (variable, gradient).

  Returns:
    norm of variables, and norm of gradients.
  """
  flatten = py_utils.Flatten(vs_gs)
  v_norm = tf.sqrt(py_utils.SumSquared([v for (v, _) in flatten]))
  scalar('var_norm/%s' % name, v_norm)
  g_norm = tf.sqrt(py_utils.SumSquared([g for (_, g) in flatten]))
  scalar('grad_norm/%s' % name, g_norm)
  return v_norm, g_norm


def CollectVarHistogram(vs_gs):
  """Adds histogram summaries for variables and gradients."""

  for name, (var, grad) in vs_gs.FlattenItems():
    name = py_utils.SanitizeScopeKey(name)
    with tf.device(var.device), tf.name_scope(name + '/summary'):
      if isinstance(grad, tf.IndexedSlices):
        var = tf.gather(var, grad.indices)
        grad = grad.values
      if var.dtype.is_complex:
        var = tf.abs(var)
        grad = tf.abs(grad)

    histogram('var_hist/' + name, var)
    histogram('grad_hist/' + name, grad)


def PrepareSequenceForPlot(tensor, padding, name):
  """Prepares a sequence feature for plotting.

  The sequence feature is transposed and channels are flattened.

  Args:
    tensor: A n-D Tensor of shape [batch, time, ...].
    padding: A Tensor of shape [batch, time].
    name: A string as the name of the reshaped Tensor, which will be used as the
      subcaption for plotting.

  Returns:
    A tuple of:
      reshaped_tensor: A 3-D Tensor of shape [batch, dim, time].
      sequence_length: A 1-D Tensor of shape [batch].
  """
  # Flatten any dimensions beyond the third into the third.
  batch_size, max_len = py_utils.GetShape(tensor, 2)
  plot_tensor = tf.reshape(tensor, [batch_size, max_len, -1])
  plot_tensor = tf.transpose(plot_tensor, [0, 2, 1], name=name)
  return (plot_tensor, SequenceLength(padding))


def PlotSequenceFeatures(plots, name, **kwargs):
  """Plots a stack of sequence features.

  Args:
    plots: A list of tuple (tensor, seq_len), as returned by
      PrepareSequenceForPlot().
    name: A string for the caption of the plot.
    **kwargs: Keyword arguments passed to AddSubplot().
  """
  if not _ShouldAddSummary():
    return

  with plot.MatplotlibFigureSummary(name, figsize=(8, len(plots) * 3.5)) as fig:
    for i, (tensor, seq_len) in enumerate(plots):
      fig.AddSubplot([tensor, seq_len],
                     TrimPaddingAndPlotSequence,
                     title=GetTensorName(tensor, name, i),
                     **kwargs)


class StatsCounter:
  """A single counter in TF."""

  def __init__(self, name):
    self._name = name
    self._var = py_utils.CreateVariable(
        name=name,
        params=py_utils.WeightParams([], py_utils.WeightInit.Constant(0),
                                     tf.int64),
        trainable=False)
    self._value = self._var.value() + 0  # Makes a copy.

  def Value(self):
    """Returns the current counter value."""
    return self._value

  def IncBy(self, delta):
    """Increment the counter by delta and return the new value."""
    # NOTE: We must ensure _value is computed (_var + 0) before
    # updating _var with delta.
    delta = tf.cast(delta, tf.int64)
    with tf.control_dependencies([self._value]):
      scalar(self._name, self._value)
      return tf.identity(tf.assign_add(self._var, delta))


class StepRateTracker:
  """A class that tracks step/example rate."""

  def __init__(self):
    self._first_step = -1
    self._time_steps = []  # History of (timestamp, global_step, total_examples)

  def ComputeStepRate(self, current_steps, total_examples):
    """Computes the overall step rate."""
    if self._time_steps:
      total_examples += self._time_steps[-1][-1]
    else:
      self._first_step = current_steps
    self._time_steps.append((time.time(), current_steps, total_examples))
    # Keeps a relative long history to compute a smooth steps/second.
    # Removes duplicate stats for step = 0 to get rid of the warm-up period.
    # Scale up the amount of history used. The first few steps are generally
    # much slower and can skew the statistic significantly otherwise.
    if current_steps - self._first_step < 1000:
      history = 100
    elif current_steps - self._first_step < 10000:
      history = 1000
    else:
      history = 10000
    while (self._time_steps[-1][1] - self._time_steps[0][1] > history or
           (len(self._time_steps) > 1 and
            self._time_steps[0][1] == self._time_steps[1][1])):
      del self._time_steps[0]
    (t0, s0, e0), (t1, s1, e1) = self._time_steps[0], self._time_steps[-1]
    rate = 0.0
    example_rate = 0.0
    if t1 > t0 + 1:
      elapsed_secs = t1 - t0
      rate = (s1 - s0) / elapsed_secs
      example_rate = (e1 - e0) / elapsed_secs
    tf.logging.info('Steps/second: %f, Examples/second: %f', rate, example_rate)
    return rate, example_rate, total_examples


def ModelAnalysis(model):
  """Returns a text showing variable sizes and their total size."""

  class Analyzer:
    """Helper class."""

    def __init__(self):
      self._seen_var = {}
      self.total = 0

    def __call__(self, v):
      assert isinstance(v, tf.Variable)
      # pylint: disable=protected-access
      if not v.shape.is_fully_defined():
        # Only Cudnn RNN params lack static shapes.
        if hasattr(v, 'approx_size'):
          size = v.approx_size
        else:
          return '%-20s %10s %s' % (v.shape, 'n/a', v._shared_name)
      else:
        size = v.shape.num_elements()
      if v._shared_name not in self._seen_var:
        self._seen_var[v._shared_name] = size
        self.total += size
      return '%-20s %10d %s' % (v.shape, size, v._shared_name)

  analyzer = Analyzer()
  output = '\n'
  output += model.vars.Transform(analyzer).DebugString()
  output += '\n'
  output += '=' * 100
  output += f'\ntotal #params: {analyzer.total:,}\n'
  return output, analyzer.total


def GetTensorName(tensor, name_eager=None, i_eager=None):
  """Returns tensor name.

  It is useful for compatibility with eager mode.
  Args:
    tensor: tensor
    name_eager: additional string to append in eager mode
    i_eager: additional index to append in eager mode

  Returns:
    tensor.name in session mode, or concatenation of name_eager, i_eager
      in eager mode
  """
  if not tf.executing_eagerly():
    tensor_name = tensor.name
  else:
    if name_eager and i_eager:
      tensor_name = f'[eager]_{name_eager}_{i_eager}'
    elif name_eager:
      tensor_name = f'[eager]_{name_eager}'
    elif i_eager:
      tensor_name = f'[eager]_{i_eager}'
    else:
      tensor_name = '[eager]'
  return tensor_name
