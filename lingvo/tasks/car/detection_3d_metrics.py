# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Metrics for 3D detection problems."""

from lingvo import compat as tf
from lingvo.core import metrics
from lingvo.core import plot
from lingvo.core import py_utils
from lingvo.tasks.car import summary
from lingvo.tasks.car import transform_util
import matplotlib.colors as matplotlib_colors
import matplotlib.patches as matplotlib_patches
import matplotlib.patheffects as path_effects
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from tensorboard.plugins.mesh import summary as mesh_summary


class TopDownVisualizationMetric(metrics.BaseMetric):
  """Top-down detection visualization, expecting 3D laser points and 2D bboxes.

  Updates to this metric is expected to be `.NestedMap` containing:
    - visualization_labels: [N, B1] int tensor containing visualization labels.
    - predicted_bboxes: [N, B1, 5] float tensor containing predicted 2D bboxes
      each with (x, y, dx, dy, phi).
    - visualization_weights: [N, B1] float tensor containing weights for each
      prediction. predictions with 0 weight will not be drawn.
    - points_xyz: [N, P, 3] float tensor containing (x, y, z) coordinates.
    - points_padding: [N, P] tensor containing 1 if the point is a padded point.
    - gt_bboxes_2d: [N, B2, 5] float tensor containing ground-truth 2D bboxes.
    - gt_bboxes_2d_weights: [N, B2] float tensor containing weights for each
      ground-truth. predictions with 0 weight will not be drawn. The
      ground-truth mask can be used here.
    - labels: [N, B2] int tensor containing ground-truth labels.
    - difficulties: [N, B2]: int tensor containing the difficulty levels of
      each groundtruth box.

  Default parameters visualize the area around the car, with the car centered in
  the image, over a 32m x 48m range.

  Ground-truth boxes will be drawn with color=cyan (see DrawBBoxesOnImages for
  details). Predicted boxes will be drawn with a color from the PIL color list,
  with a different color per class.
  """

  def __init__(self,
               top_down_transform,
               class_id_to_name=None,
               image_height=1536,
               image_width=1024,
               figsize=None,
               ground_removal_threshold=-1.35,
               sampler_num_samples=8):
    """Initialize TopDownVisualizationMetric.

    Args:
      top_down_transform: transform_util.Transform object that specifies
        how to transform a coordinate in the world coordinate to the top
        down projection.  See documentation for
        transform_util.MakeCarToImageTransform for more details on
        configuration.
      class_id_to_name: Dictionary mapping from class id to name.
      image_height: int image height.
      image_width: int image width.
      figsize: (w, h) float tuple. This is the size of the rendered figure in
        inches. A dpi=100 is used in plot.Image; note that the axes and title
        will take up space in the final rendering. If None, this will default to
        (image_width / 100 * 1.5, image_height / 100 * 1.5).
      ground_removal_threshold: Floating point value used to color ground points
        differently.  Defaults to -1.35 which happens to work well for KITTI.
      sampler_num_samples: Number of batches to keep for visualizing.
    """
    self._class_id_to_name = class_id_to_name or {}
    self._image_width = image_width
    self._image_height = image_height

    figsize = figsize or (image_width / 100. * 1.5, image_height / 100. * 1.5)
    self._figsize = figsize

    self._ground_removal_threshold = ground_removal_threshold
    self._sampler = py_utils.UniformSampler(num_samples=sampler_num_samples)
    self._top_down_transform = top_down_transform
    self._summary = None

  def Update(self, decoded_outputs):
    """Add top down visualization to summaries.

    Args:
      decoded_outputs: A `.NestedMap` containing the fields
        visualization_labels, predicted_bboxes, visualization_weights,
        points_xyz, points_padding, gt_bboxes_2d, gt_bboxes_2d_weights, and
        labels.
    """
    self._sampler.Add(decoded_outputs)
    # Invalidate cache.
    self._summary = None

  def _XYWHToExtrema(self, bboxes):
    """Convert from x, y, dx, dy to extrema ymin, xmin, ymax, xmax."""
    mtrix = np.array([
        # x    y    dx   dy
        [0.0, 1.0, 0.0, -.5],  # ymin
        [1.0, 0.0, -.5, 0.0],  # xmin
        [0.0, 1.0, 0.0, 0.5],  # ymax
        [1.0, 0.0, 0.5, 0.0],  # xmax
    ]).T
    bboxes = bboxes.copy()
    bboxes[..., :4] = np.matmul(bboxes[..., :4], mtrix)
    return bboxes

  def _DrawLasers(self, images, points_xyz, points_padding, transform):
    """Draw laser points."""
    for batch_idx in range(images.shape[0]):
      for points_idx in range(points_xyz.shape[1]):
        if points_padding[batch_idx, points_idx] == 0:
          x, y, z = points_xyz[batch_idx, points_idx, :3]
          tx, ty, _ = transform_util.TransformPoint(transform, x, y, z)
          if tx < 0 or ty < 0 or tx >= images.shape[2] or ty >= images.shape[1]:
            continue

          # Drop ground points from visualization.
          if z < self._ground_removal_threshold:
            # Brown out the color for ground points.
            color = (64, 48, 48)
          else:
            color = (255, 255, 255)

          images[batch_idx, int(ty), int(tx), :] = color

  def Summary(self, name):
    self._EvaluateIfNecessary(name)
    return self._summary

  def _EvaluateIfNecessary(self, name):
    """Create a top down image summary, if not already created."""
    if self._summary is not None:
      return

    tf.logging.info('Generating top down summary.')
    ret = tf.Summary()

    transform = self._top_down_transform

    for batch_idx, batch_sample in enumerate(self._sampler.samples):
      batch_size = batch_sample.labels.shape[0]
      visualization_labels = batch_sample.visualization_labels
      predicted_bboxes = batch_sample.predicted_bboxes
      visualization_weights = batch_sample.visualization_weights
      points_xyz = batch_sample.points_xyz
      points_padding = batch_sample.points_padding
      gt_bboxes_2d = batch_sample.gt_bboxes_2d
      gt_bboxes_2d_weights = batch_sample.gt_bboxes_2d_weights
      labels = batch_sample.labels
      difficulties = batch_sample.difficulties
      source_ids = batch_sample.source_ids

      # Create base images for entire batch that we will update.
      images = np.zeros([batch_size, self._image_height, self._image_width, 3],
                        dtype=np.uint8)

      # Draw lasers first, so that bboxes can be on top.
      self._DrawLasers(images, points_xyz, points_padding, transform)

      # Draw ground-truth bboxes.
      gt_bboxes_2d = np.where(
          np.expand_dims(gt_bboxes_2d_weights > 0, -1), gt_bboxes_2d,
          np.zeros_like(gt_bboxes_2d))
      transformed_gt_bboxes_2d = summary.TransformBBoxesToTopDown(
          gt_bboxes_2d, transform)

      summary.DrawBBoxesOnImages(
          images,
          transformed_gt_bboxes_2d,
          gt_bboxes_2d_weights,
          labels,
          self._class_id_to_name,
          groundtruth=True)

      # Draw predicted bboxes.
      predicted_bboxes = np.where(
          np.expand_dims(visualization_weights > 0, -1), predicted_bboxes,
          np.zeros_like(predicted_bboxes))
      transformed_predicted_bboxes = summary.TransformBBoxesToTopDown(
          predicted_bboxes, transform)

      summary.DrawBBoxesOnImages(
          images,
          transformed_predicted_bboxes,
          visualization_weights,
          visualization_labels,
          self._class_id_to_name,
          groundtruth=False)

      # Draw the difficulties on the image.
      self.DrawDifficulty(images, transformed_gt_bboxes_2d,
                          gt_bboxes_2d_weights, difficulties)

      for idx in range(batch_size):
        source_id = source_ids[idx]

        def AnnotateImage(fig, axes, source_id=source_id):
          """Add source_id to image."""
          del fig
          # Draw in top middle of image.
          text = axes.text(
              500,
              15,
              source_id,
              fontsize=16,
              color='blue',
              fontweight='bold',
              horizontalalignment='center')
          text.set_path_effects([
              path_effects.Stroke(linewidth=3, foreground='lightblue'),
              path_effects.Normal()
          ])

        image_summary = plot.Image(
            name='{}/{}/{}'.format(name, batch_idx, idx),
            aspect='equal',
            figsize=self._figsize,
            image=images[idx, ...],
            setter=AnnotateImage)
        ret.value.extend(image_summary.value)

    tf.logging.info('Done generating top down summary.')
    self._summary = ret

  def DrawDifficulty(self, images, gt_bboxes, gt_box_weights, difficulties):
    """Draw the difficulty values on each ground truth box."""
    batch_size = np.shape(images)[0]
    try:
      font = ImageFont.truetype('arial.ttf', size=20)
    except IOError:
      font = ImageFont.load_default()

    for batch_id in range(batch_size):
      image = images[batch_id, :, :, :]
      original_image = image
      image = Image.fromarray(np.uint8(original_image)).convert('RGB')
      draw = ImageDraw.Draw(image)
      difficulty_vector = difficulties[batch_id]
      box_data = gt_bboxes[batch_id]

      for box_id in range(box_data.shape[0]):
        box_weight = gt_box_weights[batch_id, box_id]
        if box_weight == 0:
          continue
        center_x = box_data[box_id, 0]
        center_y = box_data[box_id, 1]
        difficulty_value = str(difficulty_vector[box_id])

        # Draw a rectangle background slightly larger than the text.
        text_width, text_height = font.getsize(difficulty_value)
        draw.rectangle(
            [(center_x - text_width / 1.8, center_y - text_height / 1.8),
             (center_x + text_width / 1.8, center_y + text_height / 1.8)],
            fill='darkcyan')

        # Center the text in the rectangle
        draw.text((center_x - text_width / 2, center_y - text_height / 2),
                  str(difficulty_value),
                  fill='lightcyan',
                  font=font)
      np.copyto(original_image, np.array(image))


class WorldViewer(metrics.BaseMetric):
  """World Viewer for 3d point cloud scenes."""
  # Defines the maximum hue range for point cloud colorization by distance.
  _MAX_HUE = 0.65

  # Distance from car after which we consider all points equally far.
  _MAX_DISTANCE_METERS = 40.

  def __init__(self, sampler_num_samples=8):
    """Init."""
    self._sampler = py_utils.UniformSampler(num_samples=sampler_num_samples)
    self._summary = None

  def Update(self, decoded_outputs):
    """Add point cloud mesh data to be summarized.

    Args:
      decoded_outputs: A `.NestedMap` containing the fields
        visualization_labels, predicted_bboxes, visualization_weights,
        points_xyz, points_padding, gt_bboxes_2d, gt_bboxes_2d_weights, and
        labels.
    """
    self._sampler.Add(decoded_outputs)
    # Invalidate cache.
    self._summary = None

  def Summary(self, name):
    self._EvaluateIfNecessary(name)
    return self._summary

  def _EvaluateIfNecessary(self, name):
    """Create a mesh summary, if not already created."""
    if self._summary is not None:
      return

    summ = None
    tf.logging.info('Generating mesh summary.')
    for i, batch_sample in enumerate(self._sampler.samples):
      points_xyz = batch_sample.points_xyz[i:i + 1]
      points_padding = batch_sample.points_padding[i:i + 1]
      points_mask = (1. - points_padding).astype(bool)
      # Apply mask and expand to include a batch dimension.
      points_xyz = points_xyz[points_mask][np.newaxis, ...]

      # Compute colors based off distance from car.
      distance = np.sqrt(points_xyz[0, :, 0]**2 + points_xyz[0, :, 1]**2 +
                         points_xyz[0, :, 2]**2)
      # Normalize by some max distance beyond which we don't distinguish
      # distance.
      max_distance = np.ones_like(distance) * WorldViewer._MAX_DISTANCE_METERS
      distance = np.minimum(max_distance, distance)
      scale = (max_distance - distance) / max_distance

      # Convert to RGB.
      hue = np.minimum(WorldViewer._MAX_HUE, scale)[..., np.newaxis]
      # Invert hue so red is closer.
      hue = WorldViewer._MAX_HUE - hue
      s, v = np.ones_like(hue), np.ones_like(hue)
      hsv = np.hstack([hue, s, v])
      rgb = matplotlib_colors.hsv_to_rgb(hsv)
      colors = np.minimum(255., rgb * 255.).astype(np.uint8)
      colors = colors[np.newaxis, ...]
      summ = mesh_summary.pb(
          '{}/point_cloud/{}'.format(name, i),
          vertices=points_xyz,
          colors=colors,
          faces=None)
      # At the moment, only one scene summary is supported; writing
      # more makes the TensorBoard mesh visualizer hang.
      break

    if summ:
      self._summary = summ


class CameraVisualization(metrics.BaseMetric):
  """Camera detection visualization.

  Visualizes a camera image and predicted bounding boxes on top
  of the image.

  Updates to this metric is expected to be `.NestedMap` containing:

    camera_images: [N, W, H, 3] float tensor containing camera image data.

    bbox_corners: [N, B1, 8, 2] float tensor containing bounding box corners.
    For each batch (N), for each box B, there are 8 corners, each with
    an X and Y value.

    bbox_scores: [N, B1] float tensor containing predicted box scores.

  """

  def __init__(self,
               figsize=(15, 15),
               bbox_score_threshold=0.01,
               sampler_num_samples=8,
               draw_3d_boxes=True):
    """Initialize CameraVisualization.

    Args:
      figsize: (w, h) float tuple. This is the size of the rendered figure in
        inches. A dpi=100 is used in plot.Image; note that the axes and title
        will take up space in the final rendering. If None, this will default to
        (image_width / 100 * 1.5, image_height / 100 * 1.5).
      bbox_score_threshold: The threshold over which bboxes will be drawn on the
        image.
      sampler_num_samples: Number of batches to keep for visualizing.
      draw_3d_boxes: Whether to draw 2d or 3d bounding boxes.  3d bounding
        boxes depict the 8 corners of the bounding box, whereas the 2d
        bounding boxes depict the extrema x and y dimensions of the boxes
        on the image plane.
    """
    self._figsize = figsize
    self._bbox_score_threshold = bbox_score_threshold,
    self._sampler = py_utils.UniformSampler(num_samples=sampler_num_samples)
    self._draw_3d_boxes = draw_3d_boxes
    self._summary = None

  def Update(self, decoded_outputs):
    self._sampler.Add(decoded_outputs)
    # Invalidate cache.
    self._summary = None

  def Summary(self, name):
    self._EvaluateIfNecessary(name)
    return self._summary

  def _EvaluateIfNecessary(self, name):
    """Create a camera image summary if not already created."""
    if self._summary is not None:
      return

    ret = tf.Summary()

    for sample_idx, sample in enumerate(self._sampler.samples):
      batch_size = sample.camera_images.shape[0]

      for batch_idx in range(batch_size):
        image = sample.camera_images[batch_idx]

        # [num bboxes, 8, 2].
        bbox_corners = sample.bbox_corners[batch_idx]

        # [num_bboxes]
        bbox_scores = sample.bbox_scores[batch_idx]

        def Draw3DBoxes(fig,
                        axes,
                        bbox_corners=bbox_corners,
                        bbox_scores=bbox_scores):
          """Draw 3d bounding boxes."""
          del fig
          for bbox_id in range(bbox_corners.shape[0]):
            # Skip visualizing low-scoring boxes.
            bbox_score = bbox_scores[bbox_id]
            if bbox_score < self._bbox_score_threshold:
              continue
            bbox_data = bbox_corners[bbox_id]

            # Draw the score of each box.
            #
            # Turn score into an integer for better display.
            center_x = np.mean(bbox_data[:, 0])
            center_y = np.mean(bbox_data[:, 1])
            bbox_score = int(bbox_score * 100)
            text = axes.text(
                center_x,
                center_y,
                bbox_score,
                fontsize=12,
                color='red',
                fontweight='bold')
            text.set_bbox(dict(facecolor='yellow', alpha=0.4))

            # The BBoxToCorners function produces the points
            # in a deterministic order, which we use to draw
            # the faces of the polygon.
            #
            # The first 4 points are the "top" of the bounding box.
            # The second 4 points are the "bottom" of the bounding box.
            #
            # We then draw the last 4 connecting points by choosing
            # two of the connecting faces in the right order.
            face_points = []
            face_points += [[
                bbox_data[0, :], bbox_data[1, :], bbox_data[2, :],
                bbox_data[3, :]
            ]]
            face_points += [[
                bbox_data[4, :], bbox_data[5, :], bbox_data[6, :],
                bbox_data[7, :]
            ]]
            face_points += [[
                bbox_data[1, :], bbox_data[2, :], bbox_data[6, :],
                bbox_data[5, :]
            ]]
            face_points += [[
                bbox_data[0, :], bbox_data[3, :], bbox_data[7, :],
                bbox_data[4, :]
            ]]
            for face in face_points:
              # Each face is a list of 4 x,y points
              face_xy = np.array(face)
              axes.add_patch(
                  matplotlib_patches.Polygon(
                      face_xy, closed=True, edgecolor='red', facecolor='none'))

        def Draw2DBoxes(fig,
                        axes,
                        bbox_corners=bbox_corners,
                        bbox_scores=bbox_scores):
          """Draw 2d boxes on the figure."""
          del fig
          # Extract the 2D extrema of each bbox and the max score
          for bbox_id in range(bbox_corners.shape[0]):
            # Skip visualizing low-scoring boxes.
            bbox_score = bbox_scores[bbox_id]
            if bbox_score < self._bbox_score_threshold:
              continue
            bbox_data = bbox_corners[bbox_id]

            ymin = np.min(bbox_data[:, 1])
            xmin = np.min(bbox_data[:, 0])
            ymax = np.max(bbox_data[:, 1])
            xmax = np.max(bbox_data[:, 0])
            height = ymax - ymin
            width = xmax - xmin
            # Turn score into an integer for better display.
            bbox_score = int(bbox_score * 100)
            text = axes.text(
                xmin,
                ymin,
                bbox_score,
                fontsize=12,
                color='red',
                fontweight='bold')
            text.set_bbox(dict(facecolor='yellow', alpha=0.4))
            axes.add_patch(
                matplotlib_patches.Rectangle((xmin, ymin),
                                             width,
                                             height,
                                             edgecolor='red',
                                             facecolor='none'))

        # For each image, draw the boxes on that image.
        draw_fn = Draw3DBoxes if self._draw_3d_boxes else Draw2DBoxes
        image_summary = plot.Image(
            name='{}/{}/{}'.format(name, sample_idx, batch_idx),
            aspect='equal',
            figsize=self._figsize,
            image=image,
            setter=draw_fn)
        ret.value.extend(image_summary.value)
    self._summary = ret
