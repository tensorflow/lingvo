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
"""Tests for plot."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from lingvo.core import plot
from lingvo.core import test_utils


class PlotTest(test_utils.TestCase):

  def testToUnicode(self):
    str_str = 'pójdź kińże tę chmurność w głąb flaszy'
    uni_str = u'pójdź kińże tę chmurność w głąb flaszy'

    self.assertEqual(plot.ToUnicode(str_str), uni_str)
    self.assertEqual(plot.ToUnicode(str_str), plot.ToUnicode(uni_str))


class MatplotlibFigureSummaryTest(test_utils.TestCase):

  FIGSIZE = (8, 4)
  EXPECTED_DPI = 100
  DEFAULT_DATA = tf.ones((5, 4))

  def setUp(self):
    with self.session() as s:
      fig = plot.MatplotlibFigureSummary('DEFAULT', self.FIGSIZE, max_outputs=1)
      batched_data = tf.expand_dims(self.DEFAULT_DATA, 0)  # Batch size 1.
      fig.AddSubplot([batched_data])
      im = fig.Finalize()
      summary_str = s.run(im)
    summary = tf.summary.Summary.FromString(summary_str)
    self.default_encoded_image = summary.value[0].image.encoded_image_string

  def testBasic(self):
    with self.session() as s:
      fig = plot.MatplotlibFigureSummary(
          'matplotlib_figure', self.FIGSIZE, max_outputs=1)
      batched_data = tf.expand_dims(self.DEFAULT_DATA, 0)  # Batch size 1.
      fig.AddSubplot([batched_data])
      im = fig.Finalize()
      summary_str = s.run(im)
    summary = tf.summary.Summary.FromString(summary_str)
    self.assertEqual(len(summary.value), 1)
    value = summary.value[0]
    self.assertEqual(value.tag, 'matplotlib_figure/image')
    self.assertEqual(value.image.width, self.EXPECTED_DPI * self.FIGSIZE[0])
    self.assertEqual(value.image.height, self.EXPECTED_DPI * self.FIGSIZE[1])
    self.assertEqual(value.image.colorspace, 3)
    self.assertEqual(value.image.encoded_image_string,
                     self.default_encoded_image)

  def testUnicodeText(self):
    with self.session() as s:
      fig = plot.MatplotlibFigureSummary(
          'matplotlib_uni', self.FIGSIZE, max_outputs=1)
      batched_data = tf.expand_dims(self.DEFAULT_DATA, 0)  # Batch size 1.
      fig.AddSubplot([batched_data], xlabel=u'bździągwa', ylabel='żółć')
      im = fig.Finalize()
      summary_str = s.run(im)
    summary = tf.summary.Summary.FromString(summary_str)
    self.assertEqual(len(summary.value), 1)
    value = summary.value[0]
    self.assertEqual(value.tag, 'matplotlib_uni/image')
    self.assertEqual(value.image.width, self.EXPECTED_DPI * self.FIGSIZE[0])
    self.assertEqual(value.image.height, self.EXPECTED_DPI * self.FIGSIZE[1])
    self.assertEqual(value.image.colorspace, 3)

  def testMultipleSubplots(self):
    batch_size = 4
    tensors = [tf.ones((batch_size, 3, 5)), tf.ones((batch_size, 2, 2))]
    with self.session() as s:
      max_outputs = 3
      fig = plot.MatplotlibFigureSummary('fig', self.FIGSIZE, max_outputs)
      for t in tensors:
        fig.AddSubplot([t])
      im = fig.Finalize()
      summary_str = s.run(im)
    summary = tf.summary.Summary.FromString(summary_str)
    self.assertEqual(len(summary.value), max_outputs)
    for n, value in enumerate(summary.value):
      self.assertEqual(value.tag, 'fig/image/%d' % n)
      self.assertEqual(value.image.width, self.EXPECTED_DPI * self.FIGSIZE[0])
      self.assertEqual(value.image.height, self.EXPECTED_DPI * self.FIGSIZE[1])
      self.assertEqual(value.image.colorspace, 3)
      self.assertNotEqual(value.image.encoded_image_string,
                          self.default_encoded_image)

  def testCustomPlotFunc(self):
    batch_size = 3
    data = tf.ones((batch_size, 3, 5))
    trim = tf.constant([[3, 2], [2, 5], [3, 3]])
    titles = tf.constant(['batch1', 'batch2', 'batch3'])

    def TrimAndAddImage(fig, axes, data, trim, title, **kwargs):
      plot.AddImage(fig, axes, data[:trim[0], :trim[1]], title=title, **kwargs)

    with self.session() as s:
      fig = plot.MatplotlibFigureSummary(
          'fig_custom_plotfunc',
          max_outputs=batch_size,
          plot_func=TrimAndAddImage)
      fig.AddSubplot([data, trim, titles])
      im = fig.Finalize()
      summary_str = s.run(im)
    summary = tf.summary.Summary.FromString(summary_str)
    self.assertEqual(len(summary.value), batch_size)
    for n, value in enumerate(summary.value):
      self.assertEqual(value.tag, 'fig_custom_plotfunc/image/%d' % n)
      self.assertGreater(value.image.width, 1)
      self.assertGreater(value.image.height, 1)
      self.assertEqual(value.image.colorspace, 3)
      self.assertNotEqual(value.image.encoded_image_string,
                          self.default_encoded_image)

  def testDoesNotDieOnMatplotlibError(self):
    invalid_dim_data = tf.ones((5,))
    with self.session() as s:
      fig = plot.MatplotlibFigureSummary('summary', self.FIGSIZE, max_outputs=1)
      fig.AddSubplot([invalid_dim_data])
      im = fig.Finalize()
      summary_str = s.run(im)
    summary = tf.summary.Summary.FromString(summary_str)
    self.assertEqual(len(summary.value), 1)
    value = summary.value[0]
    # Generates dummy 1-pixel image.
    self.assertEqual(value.image.width, 1)
    self.assertEqual(value.image.height, 1)

  def testLargerBatch(self):
    batch_size = 4
    tensors = [tf.ones((batch_size, 3, 5)), tf.ones((batch_size, 2, 2))]
    with self.session() as s:
      fig = plot.MatplotlibFigureSummary(
          'larger_batch', self.FIGSIZE, max_outputs=batch_size)
      for t in tensors:
        fig.AddSubplot([t])
      im = fig.Finalize()
      summary_str = s.run(im)
    summary = tf.summary.Summary.FromString(summary_str)
    self.assertEqual(len(summary.value), batch_size)
    for n, value in enumerate(summary.value):
      self.assertEqual(value.tag, u'larger_batch/image/%d' % n)
      self.assertEqual(value.image.width, self.EXPECTED_DPI * self.FIGSIZE[0])
      self.assertEqual(value.image.height, self.EXPECTED_DPI * self.FIGSIZE[1])
      self.assertEqual(value.image.colorspace, 3)
      self.assertNotEqual(value.image.encoded_image_string,
                          self.default_encoded_image)

  def testCanChangeFigsize(self):
    figsize = (self.FIGSIZE[0], 2 * self.FIGSIZE[1])
    with self.session() as s:
      fig = plot.MatplotlibFigureSummary('summary', figsize, max_outputs=1)
      batched_data = tf.expand_dims(self.DEFAULT_DATA, 0)  # Batch size 1.
      fig.AddSubplot([batched_data])
      im = fig.Finalize()
      summary_str = s.run(im)

      summary_str = s.run(im)
    summary = tf.summary.Summary.FromString(summary_str)
    self.assertEqual(len(summary.value), 1)
    value = summary.value[0]
    self.assertEqual(value.image.width, self.EXPECTED_DPI * figsize[0])
    self.assertEqual(value.image.height, self.EXPECTED_DPI * figsize[1])
    self.assertNotEqual(value.image.encoded_image_string,
                        self.default_encoded_image)

  def testEnforcesConsistentBatchSize(self):
    batch_size = 4
    tensors = [tf.ones((batch_size, 3, 5)), tf.ones((batch_size - 2, 2, 2))]
    with self.session() as s:
      fig = plot.MatplotlibFigureSummary('summary', self.FIGSIZE, max_outputs=1)
      for t in tensors:
        fig.AddSubplot([t])
      im = fig.Finalize()
      with self.assertRaises(tf.errors.InvalidArgumentError):
        s.run(im)

  def testOnlyPlotsFirstMaxOutputImages(self):
    batch_size = 4
    tensors = [tf.ones((batch_size, 3, 5)), tf.ones((batch_size, 2, 2))]
    with self.session() as s:
      fig = plot.MatplotlibFigureSummary('summary', self.FIGSIZE, max_outputs=2)
      for t in tensors:
        fig.AddSubplot([t])
      im = fig.Finalize()
      summary_str = s.run(im)
    summary = tf.summary.Summary.FromString(summary_str)
    self.assertEqual(len(summary.value), 2)

  def testLimitsOutputImagesIfBatchIsSmall(self):
    batch_size = 1
    tensors = [tf.zeros((batch_size, 3, 5)), tf.ones((batch_size, 2, 2))]
    with self.session() as s:
      fig = plot.MatplotlibFigureSummary('summary', self.FIGSIZE, max_outputs=3)
      for t in tensors:
        fig.AddSubplot([t])
      im = fig.Finalize()
      summary_str = s.run(im)
    summary = tf.summary.Summary.FromString(summary_str)
    self.assertEqual(len(summary.value), 1)

  def testMatrix(self):
    summary = plot.Matrix('summary', (4, 4), np.random.rand(10, 10))
    self.assertEqual(len(summary.value), 1)
    value = summary.value[0]
    self.assertGreater(value.image.width, 0)
    self.assertGreater(value.image.height, 0)

  def testScatter(self):
    summary = plot.Scatter(
        'summary', (4, 4), xs=np.random.rand(10), ys=np.random.rand(10))
    self.assertEqual(len(summary.value), 1)
    value = summary.value[0]
    self.assertGreater(value.image.width, 0)
    self.assertGreater(value.image.height, 0)

  def testScatter3D(self):
    # Passing `zs` means the plot tries to use '3d' projection, which is not
    # installed by default, so raises a ValueError.
    with self.assertRaisesRegexp(ValueError, 'Unknown projection'):
      _ = plot.Scatter(
          'summary', (4, 4),
          xs=np.random.rand(10),
          ys=np.random.rand(10),
          zs=np.random.rand(10))


if __name__ == '__main__':
  tf.test.main()
