# -*- coding: utf-8 -*-
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
"""Utilities for converting CuDNN RNN params to Lingvo RNN weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops

NUM_PARAMS_PER_LSTM_LAYER = cudnn_rnn_ops.CudnnLSTM._NUM_PARAMS_PER_LAYER  # pylint:disable=protected-access
UNI_RNN = cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION
BI_RNN = cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION


class CuDNNLSTMInitializer(object):
  """Helper class for cudnn rnn weights initialization."""

  def __init__(self, num_input_nodes, num_cell_nodes, direction=UNI_RNN):
    self._input_nodes = num_input_nodes
    self._cell_nodes = num_cell_nodes
    self._direction = direction

  def OpaqueParamsShape(self, dtype):
    return cudnn_rnn_ops.cudnn_rnn_opaque_params_size(
        rnn_mode='lstm',
        num_layers=1,
        num_units=self._cell_nodes,
        input_size=self._input_nodes,
        input_mode='linear_input',
        direction=self._direction,
        dtype=dtype)

  @property
  def weight_shapes(self):
    """Return the shapes of weight tensors of each gate."""
    input_nodes = self._input_nodes
    cell_nodes = self._cell_nodes
    w_i, w_f, w_c, w_o = [(input_nodes, cell_nodes)] * 4
    r_i, r_f, r_c, r_o = [(cell_nodes, cell_nodes)] * 4
    if self._direction == BI_RNN:
      return [w_i, w_f, w_c, w_o, r_i, r_f, r_c, r_o] * 2
    else:
      return [w_i, w_f, w_c, w_o, r_i, r_f, r_c, r_o]

  @property
  def weight_sizes(self):
    """Return the sizes of weight tensor of each gate."""
    shapes = self.weight_shapes
    return [np.prod(shape) for shape in shapes]

  @property
  def weight_size(self):
    """Return the accumulated weight size."""
    return np.sum(self.weight_sizes)

  @property
  def bias_shapes(self):
    """Return the shapes of bias tensors of each gate."""
    if self._direction == BI_RNN:
      return [(self._cell_nodes)] * 16
    else:
      return [(self._cell_nodes)] * 8

  @property
  def bias_sizes(self):
    """Return the sizes of bias tensor of each gate."""
    return self.bias_shapes

  @property
  def bias_size(self):
    """Return the accumulated bias size."""
    return np.sum(self.bias_sizes)

  def InitOpaqueParams(self, dtype, base_initializer):
    """Uses base_initializer to init weights from opaque cudnn params.

    Args:
      dtype: data type.
      base_initializer: a callable that returns a tensor given shape, dtype and
          partition_info.
    Returns:
      A initialized opaque cudnn params. Its weights are initialized with the
      base_initializer, and biases are set to zero.
    """
    # The shape argument isn't used.
    weights = [
        base_initializer(sp, dtype, partition_info=None)
        for sp in self.weight_shapes
    ]
    biases = [tf.zeros(sp, dtype=dtype) for sp in self.bias_shapes]
    return cudnn_rnn_ops.cudnn_rnn_canonical_to_opaque_params(
        rnn_mode='lstm',
        num_layers=1,
        num_units=self._cell_nodes,
        input_size=self._input_nodes,
        weights=weights,
        biases=biases,
        input_mode='linear_input',
        direction=self._direction)


def _StitchWeights(w_i, w_f, w_c, w_o,
                   r_i, r_f, r_c, r_o,
                   input_dim, cell_dim):
  r"""Stitching LSTM per-gate weights to comform to LSTMCellSimple layout.

  LSTMCellSimple uses a single weight Tensor of shape [input_dim, 4 * cell_dim].
  This method puts the weight tensors together.

  Args:
    w_i:
    w_f:
    w_c:
    w_o:
      weights applied on cell input.
    r_i:
    r_f:
    r_c:
    r_o:
      weights applied on recurrent input.
    input_dim: an int, LSTM input dim.
    cell_dim: an int, LSTM cell dim.

  Returns:
    A weight Tensor.
  """
  # pylint: disable=invalid-name
  W_i = tf.concat([
      tf.reshape(w_i, [cell_dim, input_dim]),
      tf.reshape(r_i, [cell_dim, cell_dim])], axis=1)
  W_f = tf.concat([
      tf.reshape(w_f, [cell_dim, input_dim]),
      tf.reshape(r_f, [cell_dim, cell_dim])], axis=1)
  W_c = tf.concat([
      tf.reshape(w_c, [cell_dim, input_dim]),
      tf.reshape(r_c, [cell_dim, cell_dim])], axis=1)
  W_o = tf.concat([
      tf.reshape(w_o, [cell_dim, input_dim]),
      tf.reshape(r_o, [cell_dim, cell_dim])], axis=1)
  # pylint: enable=invalid-name
  # CuDNN weights are in ifco order, Lingvo LSTMCellSimple is cifo order.
  return tf.transpose(tf.concat([W_c, W_i, W_f, W_o], axis=0))


def _StitchBiases(b_wi, b_wf, b_wc, b_wo,
                  b_ri, b_rf, b_rc, b_ro):
  r"""Stitching LSTM per-gate biases to comform to LSTMCellSimple layout.

  LSTMCellSimple uses a single bias Tensor of shape [4 * cell_dim]. This method
  puts the bias tensors together.

  Args:
    b_wi:
    b_wf:
    b_wc:
    b_wo:
      biases applied on cell input.
    b_ri:
    b_rf:
    b_rc:
    b_ro:
      biases applied on recurrent input.

  Returns:
    A bias Tensor.
  """
  return (
      tf.concat([b_wc, b_wi, b_wf, b_wo], axis=0) +
      tf.concat([b_rc, b_ri, b_rf, b_ro], axis=0))


def _CuDNNParamsToCanonical(cudnn_params, input_dim, cell_dim, direction):
  r"""Convert a single piece CuDNN params to canonical params of LSTM gates.

  Args:
    cudnn_params: A Tensor containing all weights and biases of a
      CuDNN LSTM. The shape of cudnn_params given input_dim, cell_dim and
      direction can be obtained by py_utils.CuDNNInitializer.effective_shape.
    input_dim: an int, LSTM cell input dimension.
    cell_dim: an int, LSTM cell hidden dimension.
    direction: cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION or
      cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION.
  Returns:
    A list of weight Tensor and a list of bias Tensor, in the order they appear
    in input `cudnn_params`, described above.
  Raises:
    ValueError: for invalid `direction`.
  """
  if direction not in (UNI_RNN, BI_RNN):
    raise ValueError('\'direction\' must be %s or %s, receive %s.' %
                     (UNI_RNN, BI_RNN, direction))
  cudnn_initializer = CuDNNLSTMInitializer(input_dim, cell_dim, direction)
  weights, biases = tf.split(cudnn_params,
                             [cudnn_initializer.weight_size,
                              cudnn_initializer.bias_size],
                             axis=0)
  weights = tf.split(weights, cudnn_initializer.weight_sizes, axis=0)
  biases = tf.split(biases, cudnn_initializer.bias_sizes, axis=0)
  return weights, biases


def RecoverLSTMCellSimpleWeightsFromCuDNN(cudnn_params, input_dim, cell_dim,
                                          direction):
  r"""Recover LSTMCellSimple-compatible weights from (uni/bi)CuDNNLSTM weights.

  Args:
    cudnn_params: A Tensor containing all weights and biases of a
      CuDNN LSTM. The shape of cudnn_params given input_dim, cell_dim and
      direction can be obtained by py_utils.CuDNNInitializer.effective_shape.
    input_dim: an int, LSTM cell input dimension.
    cell_dim: an int, LSTM cell hidden dimension.
    direction: cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION or
      cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION.
  Returns:
    A list of weight Tensor and a list of bias Tensor.
  Raises:
    ValueError: for invalid `direction`.
  """
  if direction not in (cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION,
                       cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION):
    raise ValueError('\'direction\' must be %s or %s, receive %s.' %
                     (cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION,
                      cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION, direction))
  weights, biases = _CuDNNParamsToCanonical(cudnn_params, input_dim, cell_dim,
                                            direction)
  if direction == cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION:
    assert len(weights) == NUM_PARAMS_PER_LSTM_LAYER
    assert len(weights) == NUM_PARAMS_PER_LSTM_LAYER
    w_i, w_f, w_c, w_o, r_i, r_f, r_c, r_o = weights
    b_wi, b_wf, b_wc, b_wo, b_ri, b_rf, b_rc, b_ro = biases
    w = _StitchWeights(w_i, w_f, w_c, w_o,
                       r_i, r_f, r_c, r_o,
                       input_dim, cell_dim)
    b = _StitchBiases(b_wi, b_wf, b_wc, b_wo,
                      b_ri, b_rf, b_rc, b_ro)
    return (w, b)
  else:
    assert len(weights) == 2 * NUM_PARAMS_PER_LSTM_LAYER
    assert len(weights) == 2 * NUM_PARAMS_PER_LSTM_LAYER
    (fwd_w_i, fwd_w_f, fwd_w_c, fwd_w_o,
     fwd_r_i, fwd_r_f, fwd_r_c, fwd_r_o,
     bak_w_i, bak_w_f, bak_w_c, bak_w_o,
     bak_r_i, bak_r_f, bak_r_c, bak_r_o) = weights
    (fwd_b_wi, fwd_b_wf, fwd_b_wc, fwd_b_wo,
     fwd_b_ri, fwd_b_rf, fwd_b_rc, fwd_b_ro,
     bak_b_wi, bak_b_wf, bak_b_wc, bak_b_wo,
     bak_b_ri, bak_b_rf, bak_b_rc, bak_b_ro) = biases
    fwd_w = _StitchWeights(fwd_w_i, fwd_w_f, fwd_w_c, fwd_w_o,
                           fwd_r_i, fwd_r_f, fwd_r_c, fwd_r_o,
                           input_dim, cell_dim)
    bak_w = _StitchWeights(bak_w_i, bak_w_f, bak_w_c, bak_w_o,
                           bak_r_i, bak_r_f, bak_r_c, bak_r_o,
                           input_dim, cell_dim)
    fwd_b = _StitchBiases(fwd_b_wi, fwd_b_wf, fwd_b_wc, fwd_b_wo,
                          fwd_b_ri, fwd_b_rf, fwd_b_rc, fwd_b_ro)
    bak_b = _StitchBiases(bak_b_wi, bak_b_wf, bak_b_wc, bak_b_wo,
                          bak_b_ri, bak_b_rf, bak_b_rc, bak_b_ro)
    return (fwd_w, bak_w), (fwd_b, bak_b)


class CudNNParamsFormatConverterLSTM(
    cudnn_rnn_ops.CudnnParamsFormatConverterLSTM):
  r"""Lingvo CuDNN LSTM params converter.

  Used by Lingvo CuDNNLSTMSaveable to convert between Cudnn and Lingvo LSTM
  formats.
  """

  def _cudnn_to_tf_gate_params(self, *cu_gate_order):
    """Put CuDNN gate params to lingvo RNN cell order."""
    i_g, f_g, c_g, o_g = cu_gate_order
    return [c_g, i_g, f_g, o_g]

  def _tf_to_cudnn_gate_params(self, *tf_gate_order):
    """Put lingvo RNN cell gate params to CuDNN order."""
    c_g, i_g, f_g, o_g = tf_gate_order
    return [i_g, f_g, c_g, o_g]


class CuDNNLSTMSaveable(tf.contrib.cudnn_rnn.CudnnLSTMSaveable):
  r"""Lingvo CuDNN LSTM opaque params saveable.

  Save CuDNN opaque params in lingvo canonical format such that the
  checkpoints can be used by both CuDNN and platform-independent RNN cells.

  CuDNN LSTM equation:

      | i_t = σ(w_i * x_t + r_i * h_(t-1) + b_wi + b_ri)
      | f_t = σ(w_f * x_t + r_f * h_(t-1) + b_wf + b_rf)
      | o_t = σ(w_o * x_t + r_o * h_(t-1) + b_wo + b_ro)
      | c'_t = tanh(w_c * x_t + r_c * h_(t-1) + b_wc + b_rc)
      | c_t = f_t ◦ c_(t-1) + i_t ◦ c'_t
      | h_t = o_t ◦ tanh(c_t)

  When saving, the opaque param is first transformed into a list of tensors
  in CuDNN canonical format, then further processed to be in the format of
  LSTMCellSimple vars.

  When recovering from a CuDNN graph, the restored tensors go through the
  reverse of the aforementioned process.

  When recovering from graphs built with LSTMCellSimple, the tensors in the
  checkpoints are ready to use, with the right shapes and names.

  Specifically the tensors are saved in the following order:

  .. code-block:: none

      ------------------------------------------------------------
      | weights                    | biases                      |
      ------------------------------------------------------------
       \                             \
        -------------------------------
        | layer1     |layer2     |... |
        -------------------------------
        \             \
         ---------------
         |fwd   |bak   |
         ------------

  Conceptually, for each layer, cudnn lstm has the following params and layout:

  .. code-block:: none

      -------------------------------------------------
      | w_i | w_f | w_c | w_o | r_i | r_f | r_c | r_o |
      -------------------------------------------------
      ---------------------------------------------------------
      | b_wi | b_wf | b_wc | b_wo | b_ri | b_rf | b_rc | b_ro |
      ---------------------------------------------------------

  While Lingvo LSTM params and layout is the following:

  .. code-block:: none

      -----------------------------
      | w_c' | w_i' | w_f' | w_o' |
      | r_c' | r_i' | r_f' | r_o' |
      -----------------------------
      ---------------------------------------------------------
      | b_wc + b_rc | b_wi + b_ri | b_wf + b_rf | b_wo + b_ro |
      ---------------------------------------------------------

  The shapes of each element before transpose is reflected by
  `CuDNNLSTMInitializer.{weight_shapes, biase_shapes}`.
  """

  _format_converter_cls = CudNNParamsFormatConverterLSTM

  def __init__(self,
               opaque_params,
               cell_nodes,
               input_nodes,
               rnn_cell_name,
               scope=None,
               name='params_canonical'):
    """Constructor.

    Args:
      opaque_params: opaque CuDNN params, a single tensor w/ no static shape.
      cell_nodes: a int, num of nodes in a lstm cell.
      input_nodes: a int, the num of nodes in input.
      rnn_cell_name: the name of RNN cell in the CuDNNLSTM-ish layer. Configured
        via LSTMCellSimple.Params().name.
      scope: the variable scope of the layer variable. If not set, default to
        current variable scope.
      name: name of the saveable, should be unique in a graph.
    """
    self._rnn_cell_name = rnn_cell_name
    scope = scope or tf.get_variable_scope()
    super(CuDNNLSTMSaveable, self).__init__(
        opaque_params=opaque_params,
        num_layers=1,
        num_units=cell_nodes,
        input_size=input_nodes,
        direction=UNI_RNN,
        scope=scope,
        name=name)

  def _tf_canonical_names_single_layer(self, prefix, tf_wts_names, tf_bs_names):
    r"""Transform single layer Cudnn canonicals to tf canonicals.

    Args:
      prefix: the shared prefix of all tensor names.
      tf_wts_names: a list where names of transformed weights are stored.
      tf_bs_names: a list where names of transformed biases are stored.
    """
    tf_wts_names.append(prefix + '/wm/var')
    tf_bs_names.append(prefix + '/b/var')

  def _tf_canonical_name_prefix(self, layer, is_fwd=True):
    """The prefix of names under which lingvo canonical params are saved."""
    del is_fwd
    # Lingvo only uses single layer.
    assert layer == 0
    return self._rnn_cell_name


class BidiCuDNNLSTMSaveable(CuDNNLSTMSaveable):
  """Lingvo CuDNN LSTM opaque params saveable."""

  def __init__(self,
               opaque_params,
               cell_nodes,
               input_nodes,
               fw_cell_name,
               bw_cell_name,
               scope=None,
               name='params_canonical'):
    """Constructor.

    Args:
      opaque_params: opaque CuDNN params, a single tensor w/ no static shape.
      cell_nodes: a int, num of nodes in a lstm cell.
      input_nodes: a int, the num of nodes in input.
      fw_cell_name:
      bw_cell_name: the name of RNN cell in the BidiCuDNNLSTM-ish layer.
        Configured via LSTMCellSimple.Params().name.
      scope: the variable scope of the layer variable. If not set, default to
        current variable scope.
      name: name of the saveable, should be unique in a graph.
    """
    self._fw_cell_name = fw_cell_name
    self._bw_cell_name = bw_cell_name
    scope = scope or tf.get_variable_scope()
    super(CuDNNLSTMSaveable, self).__init__(
        opaque_params=opaque_params,
        num_layers=1,
        num_units=cell_nodes,
        input_size=input_nodes,
        direction=BI_RNN,
        scope=scope,
        name=name)

  def _tf_canonical_name_prefix(self, layer, is_fwd=True):
    """The prefix of names under which lingvo canonical params are saved."""
    # Lingvo only uses single layer.
    assert layer == 0
    return self._fw_cell_name if is_fwd else self._bw_cell_name
