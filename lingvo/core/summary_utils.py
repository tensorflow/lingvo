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

import numpy as np

import tensorflow as tf

from lingvo.core import plot
from lingvo.core import py_utils


def scalar(params, *args, **kwargs):  # pylint: disable=invalid-name
  if params.add_summary:
    tf.summary.scalar(*args, **kwargs)


def histogram(params, *args, **kwargs):  # pylint: disable=invalid-name
  if params.add_summary:
    tf.summary.histogram(*args, **kwargs)


def SequenceLength(padding):
  """Computes the length of a sequence based on binary padding.

  Args:
    padding: A tensor of binary paddings shaped [batch, seqlen].

  Returns:
    seq_lens: A tensor of shape [batch] containing the non-padded length of each
      element of plot_tensor along the batch dimension.
  """
  seq_lens = tf.cast(tf.reduce_sum(1 - padding, axis=1), tf.int32)
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


def AddAttentionSummary(params,
                        attention_tensors,
                        src_paddings,
                        tgt_paddings,
                        transcripts=None,
                        max_outputs=3):
  """Adds an image summary showing the attention probability matrix and state.

  Args:
    params: A param object.
    attention_tensors: A list of 3D tensors shaped [target_len, batch_size,
       source_len] where attention[i, j, k] is the probability for the i-th
       output attending to the k-th input for element j in the batch.
    src_paddings: A tensor of binary paddings shaped [source_len, batch] for the
      source sequence.
    tgt_paddings: A tensor of binary paddings shaped [target_len, batch] for the
      target sequence.
    transcripts: Optional, transcripts shaped [batch, target_len] for the source
      sequence.
    max_outputs: Integer maximum number of elements of the batch to plot.

  Returns:
    The added image summary.
  """
  name = attention_tensors[0].name + '/Attention'
  if not params.add_summary:
    return tf.summary.scalar('disabled_%s' % name, 0)
  fig = plot.MatplotlibFigureSummary(name, max_outputs=max_outputs)
  src_lens = SequenceLength(tf.transpose(src_paddings))
  tgt_lens = SequenceLength(tf.transpose(tgt_paddings))
  for n, atten in enumerate(attention_tensors):
    # Diagnostic metric that decreases as attention picks up.
    max_entropy = tf.log(tf.cast(src_lens, tf.float32))
    max_entropy = tf.expand_dims(tf.expand_dims(max_entropy, 0), -1)
    atten_normalized_entropy = -atten * tf.log(atten + 1e-10) / max_entropy
    scalar(params, 'Attention/average_normalized_entropy/%d' % n,
           tf.reduce_mean(atten_normalized_entropy))
    args = [tf.transpose(atten, [1, 0, 2]), src_lens, tgt_lens]
    if transcripts is not None and n == 0:
      args.append(transcripts)
    fig.AddSubplot(
        args,
        TrimPaddingAndPlotAttention,
        title=atten.name,
        xlabel='Input',
        ylabel='Output')
  return fig.Finalize()


def AddNormSummary(params, name, vs_gs):
  """"Returns and creates summary for norms of vs and their gradients gs.

  Args:
    params: A layer hyperparams.
    name: A name string for summary.
    vs_gs: A NestedMap or a list of NestedMap of (variable, gradient).

  Returns:
    norm of variables, and norm of gradients.
  """
  flatten = py_utils.NestedMap(child=vs_gs).Flatten()
  v_norm = tf.sqrt(py_utils.SumSquared([v for (v, _) in flatten]))
  scalar(params, 'var_norm_%s' % name, v_norm)
  g_norm = tf.sqrt(py_utils.SumSquared([g for (_, g) in flatten]))
  scalar(params, 'grad_norm_%s' % name, g_norm)
  return v_norm, g_norm


def CollectVarHistogram(params, vs_gs):

  def SummaryNamePrefix(n):
    return n.split(':')[0].replace('/', '.') + '/'

  for var, grad in vs_gs.Flatten():
    with tf.device(
        var.device), tf.name_scope(var.name.split(':')[0] + '/summary'):
      name_prefix = SummaryNamePrefix(var.name)
      if isinstance(grad, tf.IndexedSlices):
        var = tf.gather(var, grad.indices)
        grad = grad.values
      histogram(params, name_prefix + 'var_hist', var)
      histogram(params, name_prefix + 'grad_hist', grad)
