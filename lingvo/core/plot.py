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
"""Utilities for generating image summaries using matplotlib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import traceback

from matplotlib.backends import backend_agg
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import six
from six.moves import cStringIO
from six.moves import range
import tensorflow as tf


def ToUnicode(text):
  if not isinstance(text, six.text_type):
    text = text.decode('utf-8')
  return text


def AddPlot(unused_fig,
            axes,
            data,
            title=u'',
            xlabel=u'',
            ylabel=u'',
            fontsize='small',
            xlim=None,
            ylim=None,
            suppress_xticks=False,
            suppress_yticks=False):
  """Convenience function to add a plot."""
  axes.plot(data)
  axes.set_title(ToUnicode(title), size=fontsize)
  axes.set_xlabel(ToUnicode(xlabel), size=fontsize)
  axes.set_ylabel(ToUnicode(ylabel), size=fontsize)
  if xlim:
    axes.set_xlim(xlim)
  if ylim:
    axes.set_ylim(ylim)
  if suppress_xticks:
    axes.set_xticks([])
  if suppress_yticks:
    axes.set_yticks([])


def AddImage(fig,
             axes,
             data,
             cmap='bone_r',
             clim=None,
             show_colorbar=True,
             title=u'',
             xlabel=u'',
             ylabel=u'',
             fontsize='small',
             origin='lower',
             suppress_xticks=False,
             suppress_yticks=False,
             aspect='auto'):
  """Convenience function to plot data as an image on the given axes."""
  image = axes.imshow(
      data, cmap=cmap, origin=origin, aspect=aspect, interpolation='nearest')
  if show_colorbar:
    fig.colorbar(image)
  if clim is not None:
    image.set_clim(clim)
  axes.set_title(ToUnicode(title), size=fontsize)
  axes.set_xlabel(ToUnicode(xlabel), size=fontsize)
  axes.set_ylabel(ToUnicode(ylabel), size=fontsize)
  if suppress_xticks:
    axes.set_xticks([])
  if suppress_yticks:
    axes.set_yticks([])


def AddScatterPlot(unused_fig,
                   axes,
                   xs,
                   ys,
                   title=u'',
                   xlabel=u'',
                   ylabel=u'',
                   fontsize='small',
                   xlim=None,
                   ylim=None,
                   suppress_xticks=False,
                   suppress_yticks=False,
                   **kwargs):
  """Convenience function to add a scatter plot."""
  # For 3D axes, check to see whether zlim is specified and apply it.
  if 'zlim' in kwargs:
    zlim = kwargs.pop('zlim')
    if zlim:
      axes.set_zlim(zlim)

  axes.scatter(xs, ys, **kwargs)
  axes.set_title(ToUnicode(title), size=fontsize)
  axes.set_xlabel(ToUnicode(xlabel), size=fontsize)
  axes.set_ylabel(ToUnicode(ylabel), size=fontsize)
  if xlim:
    axes.set_xlim(xlim)
  if ylim:
    axes.set_ylim(ylim)
  if suppress_xticks:
    axes.set_xticks([])
  if suppress_yticks:
    axes.set_yticks([])


_SubplotMetadata = collections.namedtuple('_SubplotMetadata',
                                          ['tensor_list', 'plot_func'])


class MatplotlibFigureSummary(object):
  """Helper to minimize boilerplate in creating a summary with several subplots.

  Typical usage::

      >>> fig_helper = plot.MatplotlibFigureSummary(
      ...    'summary_name', shared_subplot_kwargs={'xlabel': 'Time'})
      >>> fig_helper.AddSubplot([tensor1], title='tensor1')
      >>> fig_helper.AddSubplot([tensor2], title='tensor2', ylabel='Frequency')
      >>> image_summary = fig_helper.Finalize()
  """

  def __init__(self,
               name,
               figsize=(8, 10),
               max_outputs=3,
               subplot_grid_shape=None,
               gridspec_kwargs=None,
               plot_func=AddImage,
               shared_subplot_kwargs=None):
    """Creates a new MatplotlibFigureSummary object.

    Args:
      name: A string name for the generated summary.
      figsize: A 2D tuple containing the overall figure (width, height)
        dimensions in inches.
      max_outputs: The maximum number of images to generate.
      subplot_grid_shape: A 2D tuple containing the height and width dimensions
        of the subplot grid.  height * width must be >= the number of subplots.
        Defaults to (num_subplots, 1), i.e. a vertical stack of plots.
      gridspec_kwargs: A dict of extra keyword args to use when initializing the
        figure's gridspec, as supported by matplotlib.gridspec.GridSpec.
      plot_func: A function shared across all subplots used to populate a single
        subplot.  See the docstring for AddSubplot for details.
      shared_subplot_kwargs: A dict of extra keyword args to pass to the plot
        function for all subplots.  This is useful for specifying properties
        such as 'clim' which should be consistent across all subplots.
    """
    self._name = name
    self._figsize = figsize
    self._max_outputs = max_outputs
    self._subplot_grid_shape = subplot_grid_shape
    self._gridspec_kwargs = gridspec_kwargs if gridspec_kwargs else {}
    self._plot_func = plot_func
    self._shared_subplot_kwargs = (
        shared_subplot_kwargs if shared_subplot_kwargs else {})
    self._subplots = []

  def AddSubplot(self, tensor_list, plot_func=None, **kwargs):
    r"""Adds a subplot from tensors using plot_fun to populate the subplot axes.

    Args:
      tensor_list: A list of tensors to be realized as numpy arrays and passed
        as arguments to plot_func.  The first dimension of each tensor in the
        list corresponds to batch, and must be the same size for each tensor.
      plot_func: A function with signature f(fig, axes, data1, data2, ...,
        datan, \*\*kwargs) that will be called with the realized data from
        tensor_list to plot data on axes in fig.  This function is called
        independently on each element of the batch.  Overrides plot_func passed
        in to the constructor.
      **kwargs: A dict of additional non-tensor keyword args to pass to
        plot_func when generating the plot, overridding any
        shared_subplot_kwargs.  Useful for e.g. specifying a subplot's title.
    """
    merged_kwargs = dict(self._shared_subplot_kwargs, **kwargs)
    if plot_func is None:
      plot_func = self._plot_func
    plot_func = functools.partial(plot_func, **merged_kwargs)
    self._subplots.append(_SubplotMetadata(tensor_list, plot_func))

  def Finalize(self):
    """Finishes creation of the overall figure, returning the image summary."""
    subplot_grid_shape = self._subplot_grid_shape
    if subplot_grid_shape is None:
      subplot_grid_shape = (len(self._subplots), 1)

    # AddMatplotlibFigureSummary (due to restrictions of py_func) only supports
    # flattened list of tensors so we must do some bookkeeping to maintain a
    # mapping from _SubplotMetadata object to flattened_tensors.
    subplot_slices = []
    flattened_tensors = []
    for subplot in self._subplots:
      start = len(flattened_tensors)
      subplot_slices.append((start, start + len(subplot.tensor_list)))
      flattened_tensors.extend(subplot.tensor_list)

    def PlotFunc(fig, *numpy_data_list):
      gs = gridspec.GridSpec(*subplot_grid_shape, **self._gridspec_kwargs)
      for n, subplot in enumerate(self._subplots):
        axes = fig.add_subplot(gs[n])
        start, end = subplot_slices[n]
        subplot_data = numpy_data_list[start:end]
        subplot.plot_func(fig, axes, *subplot_data)

    func = functools.partial(_RenderMatplotlibFigures, self._figsize,
                             self._max_outputs, PlotFunc)
    batch_sizes = [tf.shape(t)[0] for t in flattened_tensors]
    num_tensors = len(flattened_tensors)
    with tf.control_dependencies([
        tf.assert_equal(
            batch_sizes, [batch_sizes[0]] * num_tensors, summarize=num_tensors)
    ]):
      rendered = tf.py_func(
          func, flattened_tensors, tf.uint8, name='RenderMatplotlibFigures')
    return tf.summary.image(self._name, rendered, max_outputs=self._max_outputs)


def _RenderOneMatplotlibFigure(fig, plot_func, *numpy_data_list):
  fig.clear()
  plot_func(fig, *numpy_data_list)
  fig.canvas.draw()
  ncols, nrows = fig.canvas.get_width_height()
  image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
  return image.reshape(nrows, ncols, 3)


def _RenderMatplotlibFigures(figsize, max_outputs, plot_func, *numpy_data_list):
  r"""Renders a figure containing several subplots using matplotlib.

  This is an internal implementation detail of MatplotlibFigureSummary.Finalize
  and should not be called directly.

  The unconventional function signature is used to work around the behavior of
  `tf.py_func` which always passes in different tensors as positional arguments.

  Args:
    figsize: A 2D tuple containing the overall figure (width, height) dimensions
      in inches.
    max_outputs: The maximum number of images to generate.
    plot_func: A function with signature f(fig, data1, data2, ..., datan) that
      will be called with \*numpy_data_list to plot data in fig.
    *numpy_data_list: A list of numpy matrices to plot specified as separate
      arguments.

  Returns:
    A numpy 4D array of type np.uint8 which can be used to generate a
    `tf.image_summary` when converted to a tf tensor.
  """
  batch_size = numpy_data_list[0].shape[0]
  max_outputs = min(max_outputs, batch_size)
  images = []

  # Use plt.Figure instead of plt.figure to avoid a memory leak (matplotlib
  # keeps global references to every figure created with plt.figure). When not
  # using plt.figure we have to create a canvas manually.
  fig = plt.Figure(figsize=figsize, dpi=100, facecolor='white')
  backend_agg.FigureCanvasAgg(fig)

  for b in range(max_outputs):
    data = [numpy_data[b] for numpy_data in numpy_data_list]
    try:
      images.append(_RenderOneMatplotlibFigure(fig, plot_func, *data))
    except Exception as e:  # pylint: disable=broad-except
      tf.logging.warning('Error rendering example %d using matplotlib: %s\n%s',
                         b, e, traceback.format_exc())
    if len(images) == max_outputs:
      break
  plt.close(fig)

  # Pad with dummy black images in case there were too many rendering errors.
  while len(images) < max_outputs:
    image_shape = (1, 1, 1)
    if images:
      image_shape = images[0].shape
    images.append(np.ones(image_shape, dtype=np.uint8))

  return np.array(images)


def _FigureToSummary(name, fig):
  """Create tf.Summary proto from matplotlib.figure.Figure ."""
  canvas = backend_agg.FigureCanvasAgg(fig)
  fig.canvas.draw()
  ncols, nrows = fig.canvas.get_width_height()
  png_file = cStringIO()
  canvas.print_figure(png_file)
  png_str = png_file.getvalue()
  return tf.Summary(value=[
      tf.Summary.Value(
          tag='%s/image' % name,
          image=tf.Summary.Image(
              height=nrows,
              width=ncols,
              colorspace=3,
              encoded_image_string=png_str))
  ])


def Image(name, figsize, image, setter=None, **kwargs):
  """Plot an image in numpy and generates tf.Summary proto for it.

  Args:
    name: Image summary name.
    figsize: A 2D tuple containing the overall figure (width, height) dimensions
      in inches.
    image: A 2D/3D numpy array in the format accepted by pyplot.imshow.
    setter: A callable taking (fig, axes). Useful to fine-tune layout of the
      figure, xlabel, xticks, etc.
    **kwargs: Additional arguments to AddImage.

  Returns:
    A `tf.Summary` proto contains one image visualizing 'image.
  """
  assert image.ndim in (2, 3), '%s' % image.shape
  fig = plt.Figure(figsize=figsize, dpi=100, facecolor='white')
  axes = fig.add_subplot(1, 1, 1)
  AddImage(fig, axes, image, origin='upper', show_colorbar=False, **kwargs)
  if setter:
    setter(fig, axes)
  return _FigureToSummary(name, fig)


def Scatter(name, figsize, xs, ys, setter=None, **kwargs):
  """Plot a scatter plot in numpy and generates tf.Summary proto for it.

  Args:
    name: Scatter plot summary name.
    figsize: A 2D tuple containing the overall figure (width, height) dimensions
      in inches.
    xs: A set of x points to plot.
    ys: A set of y points to plot.
    setter: A callable taking (fig, axes). Useful to fine-tune layout of the
      figure, xlabel, xticks, etc.
    **kwargs: Additional arguments to AddScatterPlot.

  Returns:
    A `tf.Summary` proto contains one image visualizing 'image.
  """
  fig = plt.Figure(figsize=figsize, dpi=100, facecolor='white')

  # If z data is provided, use 3d projection.
  #
  # This requires the mplot3d toolkit (e.g., from mpl_toolkits import mplot3d)
  # to be registered in the program.
  if 'zs' in kwargs:
    axes = fig.add_subplot(111, projection='3d')
  else:
    axes = fig.add_subplot(1, 1, 1)
  AddScatterPlot(fig, axes, xs, ys, **kwargs)
  if setter:
    setter(fig, axes)
  return _FigureToSummary(name, fig)


Matrix = Image  # pylint: disable=invalid-name


def Curve(name, figsize, xs, ys, setter=None, **kwargs):
  """Plot curve(s) to a `tf.Summary` proto.

  Args:
    name: Image summary name.
    figsize: A 2D tuple containing the overall figure (width, height) dimensions
      in inches.
    xs: x values for matplotlib.pyplot.plot.
    ys: y values for matplotlib.pyplot.plot.
    setter: A callable taking (fig, axes). Useful to fine-control layout of the
      figure, xlabel, xticks, etc.
    **kwargs: Extra args for matplotlib.pyplot.plot.

  Returns:
    A `tf.Summary` proto contains the line plot.
  """
  fig = plt.Figure(figsize=figsize, dpi=100, facecolor='white')
  axes = fig.add_subplot(1, 1, 1)
  axes.plot(xs, ys, '.-', **kwargs)
  if setter:
    setter(fig, axes)
  return _FigureToSummary(name, fig)
