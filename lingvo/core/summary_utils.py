# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lingvo.compat as tf
from lingvo.core import cluster_factory
from lingvo.core import plot
from lingvo.core import py_utils
import numpy as np


def _ShouldAddSummary():
  return cluster_factory.Current().add_summary


def scalar(*args, **kwargs):  # pylint: disable=invalid-name
  if _ShouldAddSummary():
    tf.summary.scalar(*args, **kwargs)


def histogram(*args, **kwargs):  # pylint: disable=invalid-name
  if _ShouldAddSummary():
    tf.summary.histogram(*args, **kwargs)


def image(*args, **kwargs):  # pylint: disable=invalid-name
  if _ShouldAddSummary():
    tf.summary.image(*args, **kwargs)


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


def AddAttentionSummary(attention_tensors,
                        src_paddings,
                        tgt_paddings,
                        transcripts=None,
                        max_outputs=3):
  """Adds an image summary showing the attention probability matrix and state.

  Tensors are in sequence tensor format with the batch dimension in axis 1.

  Args:
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
      [tf.transpose(a, [1, 0, 2]) for a in attention_tensors],
      Transpose(src_paddings), Transpose(tgt_paddings), transcripts,
      max_outputs)


def AddAttentionSummaryBatchMajor(attention_tensors,
                                  src_paddings,
                                  tgt_paddings,
                                  transcripts=None,
                                  max_outputs=3):
  """Adds an image summary showing the attention probability matrix and state.

  As opposed to AddAttentionSummary() takes all tensors with batch dimension in
  axis 0.

  Args:
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

  name = attention_tensors[0].name + '/Attention'
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
      name, max_outputs=max_outputs, gridspec_kwargs={'hspace': 0.3}) as fig:
    for n, atten in enumerate(attention_tensors):
      # Diagnostic metric that decreases as attention picks up.
      max_entropy = tf.log(tf.cast(Get(src_lens, n), tf.float32))
      max_entropy = tf.expand_dims(tf.expand_dims(max_entropy, -1), -1)
      atten_normalized_entropy = -atten * tf.log(atten + 1e-10) / max_entropy
      scalar('Attention/average_normalized_entropy/%d' % n,
             tf.reduce_mean(atten_normalized_entropy))
      args = [atten, Get(src_lens, n), Get(tgt_lens, n)]
      if transcripts is not None and n == 0:
        args.append(transcripts)
      fig.AddSubplot(
          args,
          TrimPaddingAndPlotAttention,
          title=atten.name,
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
  flatten = py_utils.NestedMap(child=vs_gs).Flatten()
  v_norm = tf.sqrt(py_utils.SumSquared([v for (v, _) in flatten]))
  scalar('var_norm/%s' % name, v_norm)
  g_norm = tf.sqrt(py_utils.SumSquared([g for (_, g) in flatten]))
  scalar('grad_norm/%s' % name, g_norm)
  return v_norm, g_norm


def CollectVarHistogram(vs_gs):
  """Adds histogram summaries for variables and gradients."""

  for name, (var, grad) in vs_gs.FlattenItems():
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
    for tensor, seq_len in plots:
      fig.AddSubplot([tensor, seq_len],
                     TrimPaddingAndPlotSequence,
                     title=tensor.name,
                     **kwargs)
