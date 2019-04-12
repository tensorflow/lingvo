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
"""Image classification models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow as tf

from lingvo.core import base_layer
from lingvo.core import base_model
from lingvo.core import layers
from lingvo.core import lr_schedule
from lingvo.core import metrics
from lingvo.core import plot
from lingvo.core import py_utils


class BaseClassifier(base_model.BaseTask):
  """Base class for image classifier."""

  @classmethod
  def Params(cls):
    p = super(BaseClassifier, cls).Params()
    p.Define('softmax', layers.SimpleFullSoftmax.Params(), 'Softmax layer.')
    return p

  def _AddSummary(self, batch, prediction):
    """Adds image summaries for the batch."""
    if self.params.is_eval:
      # Image summaries only works in evaler/decoder.
      fig = plot.MatplotlibFigureSummary(
          'examples', figsize=(1, 1), max_outputs=10)

      def Draw(fig, axes, img, label, pred):
        plot.AddImage(
            fig=fig,
            axes=axes,
            data=img[:, :, 0] / 256.,
            show_colorbar=False,
            suppress_xticks=True,
            suppress_yticks=True)
        axes.text(
            x=0.5,
            y=0,
            s=u'%d vs. %d' % (label, pred),
            transform=axes.transAxes,
            horizontalalignment='center')

      fig.AddSubplot([batch.raw, batch.label, prediction], Draw)
      fig.Finalize()

  def _Accuracy(self, k, logits, labels, weights):
    """Compute top-k accuracy.

    Args:
      k: An int scalar. Top-k.
      logits: A [N, C] float tensor.
      labels: A [N] int vector.
      weights: A [N] float vector.

    Returns:
      A float scalar. The accuracy at precision k.
    """
    logits = py_utils.HasRank(logits, 2)
    n, c = tf.unstack(tf.shape(logits), 2)
    labels = py_utils.HasShape(labels, [n])
    weights = py_utils.HasShape(weights, [n])
    correct = tf.nn.in_top_k(logits, labels, k)
    return tf.reduce_sum(
        tf.cast(correct, weights.dtype) * weights) / tf.maximum(
            1e-8, tf.reduce_sum(weights))


class ModelV1(BaseClassifier):
  """CNNs with maxpooling followed by a softmax."""

  @classmethod
  def Params(cls):
    p = super(ModelV1, cls).Params()
    p.Define(
        'filter_shapes', [(0, 0, 0, 0)],
        'Conv filter shapes. Must be a list of sequences of 4. '
        'Elements are in order of height, width, in_channel, out_channel')
    p.Define(
        'window_shapes', [(0, 0)],
        'Max pooling window shapes. Must be a list of sequences of 2. '
        'Elements are in order of height, width.')
    p.Define('batch_norm', False, 'Apply BN or not after the conv.')
    p.Define('dropout_prob', 0.0,
             'Probability of the dropout applied after pooling.')

    tp = p.train
    tp.learning_rate = 1e-4  # Adam base LR.
    tp.lr_schedule = (
        lr_schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule
        .Params().Set(
            warmup=100, decay_start=100000, decay_end=1000000, min=0.1))
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(ModelV1, self).__init__(params)
    p = self.params
    assert p.name

    with tf.variable_scope(p.name):
      assert len(p.filter_shapes) == len(p.window_shapes)

      # A few conv + max pooling layers.
      shape = tf.TensorShape([None] + list(p.input.data_shape))
      conv_params = []
      pooling_params = []
      for i, (kernel, window) in enumerate(
          zip(p.filter_shapes, p.window_shapes)):
        conv_params.append(layers.ConvLayer.Params().Set(
            name='conv%d' % i,
            filter_shape=kernel,
            filter_stride=(1, 1),
            batch_norm=p.batch_norm))
        pooling_params.append(layers.PoolingLayer.Params().Set(
            name='pool%d' % i, window_shape=window, window_stride=window))
      self.CreateChildren('conv', conv_params)
      self.CreateChildren('pool', pooling_params)

      # Logs expected activation shapes.
      for i in range(len(self.conv)):
        tf.logging.info('shape %d %s', i, shape)
        shape = self.conv[i].OutShape(shape)
        tf.logging.info('shape %d %s', i, shape)
        shape = self.pool[i].OutShape(shape)
      tf.logging.info('shape %s', shape)

      # FC layer to project down to p.softmax.input_dim.
      self.CreateChild(
          'fc',
          layers.FCLayer.Params().Set(
              name='fc',
              input_dim=np.prod(shape[1:]),
              output_dim=p.softmax.input_dim))
      self.CreateChild('softmax', p.softmax)

  def FPropTower(self, theta, input_batch):
    p = self.params
    batch = tf.shape(input_batch.data)[0]
    height, width, depth = p.input.data_shape
    act = tf.reshape(input_batch.data, [batch, height, width, depth])
    for i in range(len(self.conv)):
      # Conv, BN (optional)
      act, _ = self.conv[i].FProp(theta.conv[i], act)
      # MaxPool
      act, _ = self.pool[i].FProp(theta.pool[i], act)
      # Dropout (optional)
      if p.dropout_prob > 0.0 and not p.is_eval:
        act = tf.nn.dropout(
            act, keep_prob=1.0 - p.dropout_prob, seed=p.random_seed)
    # FC
    act = self.fc.FProp(theta.fc, tf.reshape(act, [batch, -1]))

    # Softmax
    labels = tf.to_int64(input_batch.label)
    xent = self.softmax.FProp(
        theta=theta.softmax,
        inputs=act,
        class_weights=input_batch.weight,
        class_ids=labels)

    self._AddSummary(input_batch, xent.per_example_argmax)

    rets = {
        'loss': (xent.avg_xent, batch),
        'log_pplx': (xent.avg_xent, batch),
        'num_preds': (batch, 1),
    }
    if p.is_eval:
      acc1 = self._Accuracy(1, xent.logits, labels, input_batch.weight)
      acc5 = self._Accuracy(5, xent.logits, labels, input_batch.weight)
      rets.update(accuracy=(acc1, batch), acc5=(acc5, batch))
    return rets, {}

  def Decode(self, input_batch):
    with tf.name_scope('decode'):
      return self.FPropDefaultTheta(input_batch)[0]

  def CreateDecoderMetrics(self):
    return {
        'num_samples_in_batch': metrics.AverageMetric(),
    }

  def PostProcessDecodeOut(self, dec_out_dict, dec_metrics_dict):
    dec_metrics_dict['num_samples_in_batch'].Update(
        dec_out_dict['num_samples_in_batch'][0])


class ModelV2(BaseClassifier):
  """CNNs followed by a softmax."""

  @classmethod
  def Params(cls):
    p = super(ModelV2, cls).Params()
    p.Define('extract', None, 'Param for the layer to extract image features.')
    p.Define('label_smoothing', 0., 'Smooth the labels towards 1/num_classes.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(ModelV2, self).__init__(params)
    p = self.params
    assert p.name

    with tf.variable_scope(p.name):
      self.CreateChild('extract', p.extract)
      self.CreateChild('softmax', p.softmax)

  def ComputePredictions(self, theta, input_batch):
    # Forward through layers.
    act = self.extract.FProp(theta.extract, input_batch.data)
    # Avg pool
    act = tf.reduce_mean(act, axis=[1, 2])
    logits = self.softmax.Logits(theta.softmax, act)
    return py_utils.NestedMap(act=act, logits=logits)

  def ComputeLoss(self, theta, input_batch, predictions):
    p = self.params
    batch = tf.shape(input_batch.data)[0]
    act = predictions.act
    with tf.colocate_with(act):
      tf.logging.info("{}'s device: {}".format(act, act.device))
      # Softmax
      labels = tf.to_int64(input_batch.label)
      onehot_labels = tf.one_hot(labels, p.softmax.num_classes)
      if p.label_smoothing > 0:
        smooth_positives = 1.0 - p.label_smoothing
        smooth_negatives = p.label_smoothing / p.softmax.num_classes
        onehot_labels = onehot_labels * smooth_positives + smooth_negatives

      xent = self.softmax.FProp(
          theta=theta.softmax,
          inputs=act,
          class_weights=input_batch.weight,
          class_probabilities=onehot_labels)

    self._AddSummary(input_batch, xent.per_example_argmax)

    rets = {
        'loss': (xent.avg_xent, batch),
        'log_pplx': (xent.avg_xent, batch),
        'num_preds': (batch, 1),
    }
    if self._compute_accuracy():
      acc1 = self._Accuracy(1, xent.logits, labels, input_batch.weight)
      acc5 = self._Accuracy(5, xent.logits, labels, input_batch.weight)
      rets.update(accuracy=(acc1, batch), acc5=(acc5, batch))
    return rets, {}

  def _compute_accuracy(self):
    return self.params.is_eval

  def Inference(self):
    """Constructs inference subgraphs.

    Returns:
      A dictionary of the form {'subgraph_name': (fetches, feeds)}. Each of
      fetches and feeds is itself a dictionary which maps a string name (which
      describes the tensor) to a corresponding tensor in the inference graph
      which should be fed/fetched from.
    """
    subgraphs = {}
    with tf.name_scope('inference'):
      subgraphs['default'] = self._InferenceSubgraph_Default()
    return subgraphs

  def _InferenceSubgraph_Default(self):
    """Constructs graph for single-image inference.

    Returns:
      (fetches, feeds) where both fetches and feeds are dictionaries. Each
      dictionary consists of keys corresponding to tensor names, and values
      corresponding to a tensor in the graph which should be input/read from.
    """
    p = self.params
    with tf.name_scope('default'):
      normalized_image = tf.placeholder(
          dtype=p.dtype, shape=p.input.data_shape, name='normalized_image')
      inputs = py_utils.NestedMap(data=normalized_image[tf.newaxis, ...])
      logits = tf.reshape(
          self.ComputePredictions(self.theta, inputs).logits,
          [p.softmax.num_classes],
          name='logits')
      feeds = {
          'normalized_image': normalized_image,
      }
      fetches = {
          'logits': logits,
          'probs': tf.nn.softmax(logits, name='probs'),
          'prediction': tf.argmax(logits, name='prediction'),
      }
      return fetches, feeds
