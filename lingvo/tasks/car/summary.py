# Lint as: python2, python3
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
"""Functions to help with plotting summaries.

These functions try to take care in ensuring that transformations on points and
headings are consistent.  For example, when points are transformed from the
standard x/y plane (+x going right, +y going up) to an alternate formulation
such as 'flipped axes', the heading must also be updated to account for the
change in axis directions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from lingvo import compat as tf
from lingvo.core import plot
from lingvo.tasks.car import transform_util
import matplotlib.patheffects as path_effects
import numpy as np

import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from six.moves import range


# Source code for keys can be found at:
# https://github.com/python-pillow/Pillow/blob/080bfd3ee1412b401d520fe26c51e2f5515e3a65/src/PIL/ImageColor.py#L163
def _PILColorList():
  pil_color_list = sorted(ImageColor.colormap.keys())
  pil_color_list.remove('black')
  return pil_color_list


PIL_COLOR_LIST = _PILColorList()


def ExtractRunIds(run_segments):
  """Extract the RunIds from the run_segments feature field.

  Args:
    run_segments: a string Tensor of shape [batch, 1] containing a text proto.

      See `SummaryTest.testExtractRunIds` for an example.

  Returns:
    A string Tensor of shape [batch], containing the extracted run id.
  """
  run_segments = tf.convert_to_tensor(run_segments)[:, 0]
  return tf.strings.regex_replace(run_segments,
                                  r'[^:]+: "(.+)"\n[^:]+: (\d+)(.|\n)*',
                                  r'\1_\2')


def CameraImageSummary(frontal_images, run_segment_strings, figsize=(6, 4)):
  """Write frontal_images as tf.Summaries.

  Args:
    frontal_images: Float tensor of frontal camera images: Shape: [batch,
      height, width, depth]. Expected aspect ratio of 3:2 for visualization.
    run_segment_strings: Tensor of strings: Shape: [batch, 1].  The associated
      RunSegment proto for the batch.
    figsize: Tuple indicating size of camera image. Default is (6, 4)
    indicating a 3:2 aspect ratio for visualization.
  """
  # Parse the run segment strings to extract the run segment info.
  run_segment_ids = ExtractRunIds(run_segment_strings)

  def DrawCameraImage(fig, axes, frontal_image, run_segment_id):
    """Draw camera image for image summary."""
    plot.AddImage(
        fig=fig,
        axes=axes,
        data=frontal_image / 256.,
        show_colorbar=False,
        suppress_xticks=True,
        suppress_yticks=True)
    txt = axes.text(
        x=0.5,
        y=0.01,
        s=run_segment_id,
        color='blue',
        fontsize=14,
        transform=axes.transAxes,
        horizontalalignment='center')
    txt.set_path_effects([
        path_effects.Stroke(linewidth=3, foreground='lightblue'),
        path_effects.Normal()
    ])

  with plot.MatplotlibFigureSummary(
      'examples', figsize=figsize, max_outputs=10) as fig:
    # Plot raw frontal image samples for each example.
    fig.AddSubplot([frontal_images, run_segment_ids], DrawCameraImage)


def _CarToImageTransform():
  return transform_util.MakeCarToImageTransform(
      pixels_per_meter=10., image_ref_x=250, image_ref_y=750, flip_axes=True)


def DrawTopDown(lasers):
  """Draw the laser points in the top down car view."""
  # For every laser point, convert the point to a top down image.

  # Lasers is a [r, b, s*6] tensor where b is the batch size.
  # Transpose back to batch major
  lasers = np.transpose(lasers, [1, 0, 2])
  # Reshape data to get all points in the spin.
  lasers = np.reshape(lasers, [np.shape(lasers)[0], -1])

  car_to_image_transform = _CarToImageTransform()

  # Create an empty image of the appropriate size.
  batch_size = min(8, np.shape(lasers)[0])
  images = np.zeros(shape=(batch_size, 1000, 500, 3), dtype=np.uint8)

  # TODO(vrv): Slice the lasers into [b, x, y, z] matrix
  # and then do a batch_matmul on all points at once.
  max_npoints = np.shape(lasers)[1] // 6
  for b in range(batch_size):
    for i in range(max_npoints):
      index = i * 6
      x = lasers[b][index]
      y = lasers[b][index + 1]
      z = lasers[b][index + 2]

      # TODO(vrv): Use lasers_padding to filter out invalid points.
      # For now, just assume that all zeros means that the point
      # shouldn't be drawn.
      if x == 0 and y == 0 and z == 0:
        continue

      tx, ty, _ = transform_util.TransformPoint(car_to_image_transform, x, y, z)
      # Point outside image.
      if tx < 0 or ty < 0 or tx >= images.shape[2] or ty >= images.shape[1]:
        continue

      # Fill in that point in the image
      images[b, int(ty), int(tx), :] = (255, 255, 255)

  return images


def MakeRectangle(l, w, theta, offset=(0, 0)):
  """Make rotated rectangle."""
  c, s = math.cos(theta), math.sin(theta)
  coordinates = [(l / 2.0, w / 2.0), (l / 2.0, -w / 2.0), (-l / 2.0, -w / 2.0),
                 (-l / 2.0, w / 2.0)]
  return [(c * x - s * y + offset[0], s * x + c * y + offset[1])
          for (x, y) in coordinates]


def DrawHeadingTriangle(draw, x, y, heading, color, scale=25):
  """Draw a triangle indicating `heading` at `x`, `y` with `color`."""
  ch = scale * math.cos(heading)
  sh = scale * math.sin(heading)
  lead_point = (x - ch, y - sh)
  anchor_1 = (x - sh / 3, y + ch / 3)
  anchor_2 = (x + sh / 3, y - ch / 3)
  draw.line([anchor_1, lead_point, anchor_2], fill=color, width=4)


def DrawCircle(draw, x, y, fill, outline, circle_size=5):
  """Draw a circle at `x`, `y`."""
  draw.ellipse(
      (x - circle_size, y - circle_size, x + circle_size, y + circle_size),
      fill=fill,
      outline=outline)


# TODO(vrv): Support multiple display strings.
def DrawBoundingBoxOnImage(image,
                           box,
                           display_str,
                           color='red',
                           thickness=4,
                           text_loc='BOTTOM'):
  """Draw bounding box on the input image."""
  original_image = image
  image = Image.fromarray(np.uint8(original_image)).convert('RGB')
  draw = ImageDraw.Draw(image)

  center_x, center_y, width, height, heading = box
  box2d = transform_util.Box2D(center_x, center_y, width, height, heading)
  corners = list(box2d.corners.reshape(-1, 2))
  points = [tuple(c) for c in corners]
  points += [points[0]]
  draw.line(points, fill=color, width=thickness)

  # Draw heading.
  max_dim = max(width, height) / 2.
  end_heading_point = (center_x + max_dim * math.cos(heading),
                       center_y + max_dim * math.sin(heading))
  start_heading_point = ((end_heading_point[0] - center_x) / 2 + center_x,
                         (end_heading_point[1] - center_y) / 2 + center_y)

  heading_points = [start_heading_point, end_heading_point]
  draw.line(heading_points, fill=color, width=thickness)

  # Compute extremes so we can anchor the labels to them.
  xs = [x[0] for x in points]
  ys = [x[1] for x in points]
  left = np.min(xs)
  bottom = np.min(ys)
  top = np.max(ys)

  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  text_width, text_height = font.getsize(display_str)
  margin = np.ceil(0.05 * text_height)
  if text_loc == 'TOP':
    text_bottom = top
  else:
    text_bottom = bottom + text_height

  draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                  (left + text_width, text_bottom)],
                 fill=color)
  draw.text((left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
  np.copyto(original_image, np.array(image))


def VisualizeBoxes(image,
                   boxes,
                   classes,
                   scores,
                   class_id_to_name,
                   min_score_thresh=.25,
                   line_thickness=4,
                   groundtruth_box_visualization_color='black',
                   skip_scores=False,
                   skip_labels=False,
                   text_loc='TOP'):
  """Visualize boxes on top down image."""
  box_to_display_str_map = collections.defaultdict(str)
  box_to_color_map = collections.defaultdict(str)
  num_boxes = boxes.shape[0]
  for i in range(num_boxes):
    if scores is not None and scores[i] < min_score_thresh:
      continue
    box = tuple(boxes[i].tolist())
    display_str = ''
    if not skip_labels:
      if classes[i] in class_id_to_name:
        class_name = class_id_to_name[classes[i]]
        display_str = str(class_name)
      else:
        display_str = 'N/A'
    if not skip_scores:
      if not display_str:
        display_str = '{}%'.format(int(100 * scores[i]))
      else:
        display_str = '{}: {}%'.format(display_str, int(100 * scores[i]))
    box_to_display_str_map[box] = display_str
    if scores is None:
      box_to_color_map[box] = groundtruth_box_visualization_color
    else:
      box_to_color_map[box] = PIL_COLOR_LIST[classes[i] % len(PIL_COLOR_LIST)]

  # Draw all boxes onto image.
  for box, color in box_to_color_map.items():
    DrawBoundingBoxOnImage(
        image,
        box,
        color=color,
        thickness=line_thickness,
        display_str=box_to_display_str_map[box],
        text_loc=text_loc)
  return image


def TransformBBoxesToTopDown(bboxes, car_to_image_transform=None):
  """Convert bounding boxes from car coordinates to top down pixel coordinates.

  Args:
    bboxes: A (batch, nbboxes, 4 or 5) np.float32 tensor containing bounding box
      xywhh in car coordinates (smooth adjusted by car pose).
    car_to_image_transform: An optional Transform object. If None, this will be
      created using _CarToImageTransform.

  Returns:
    np.array of shape (batch, nbboxes) containing the bounding boxes in top down
    image space.
  """
  if car_to_image_transform is None:
    car_to_image_transform = _CarToImageTransform()

  batch_size = np.shape(bboxes)[0]
  nbboxes = np.shape(bboxes)[1]
  transformed_boxes = np.zeros_like(bboxes)

  for batch_id in range(batch_size):
    for box_id in range(nbboxes):
      # TODO(vrv): When we predict heading, we should assert
      # that the length of bbox_data is 5.
      bbox_data = np.squeeze(bboxes[batch_id, box_id, :])

      x, y, width, height = (bbox_data[0], bbox_data[1], bbox_data[2],
                             bbox_data[3])

      if len(bbox_data) == 5:
        heading = bbox_data[4]
      else:
        heading = 0.0

      # Skip boxes that cannot be visualized.
      if width <= 0 or height <= 0:
        continue
      if any([np.isnan(c) or np.isinf(c) for c in bbox_data]):
        continue

      # Bounding boxes are in car coordinates (smooth adjusted by car pose).
      # Transform from car coordinates to new coordinates.
      bbox_car = transform_util.Box2D(x, y, width, height, heading)
      bbox_transformed = bbox_car.Apply(car_to_image_transform)
      bbox_values = bbox_transformed.AsNumpy()
      if len(bbox_data) == 4:
        bbox_values = bbox_values[:-1]
      transformed_boxes[batch_id, box_id, :] = bbox_values

  return transformed_boxes


def DrawBBoxesOnImages(images, bboxes, box_weights, labels, class_id_to_name,
                       groundtruth):
  """Draw ground truth boxes on top down image.

  Args:
    images: A 4D uint8 array (batch, height, width, depth) of images to draw on
      top of.
    bboxes: A (batch, nbboxes, 4 or 5) np.float32 tensor containing bounding box
      xywhh to draw specified in top down pixel values.
    box_weights: A (batch, nbboxes) float matrix indicating the predicted score
      of the box.  If the score is 0.0, no box is drawn.
    labels: A (batch, nbboxes) integer matrix indicating the true or predicted
      label indices.
    class_id_to_name: Dictionary mapping from class id to name.
    groundtruth: Boolean indicating whether bounding boxes are ground truth.

  Returns:
    'images' with the bboxes drawn on top.
  """
  # Assert 4d shape.
  assert len(np.shape(images)) == 4

  if np.shape(images)[3] == 1:
    # Convert from grayscale to RGB.
    images = np.tile(images, (1, 1, 1, 3))

  # Assert channel dimension is 3 dimensional.
  assert np.shape(images)[3] == 3

  batch_size = np.shape(images)[0]
  nbboxes = np.shape(bboxes)[1]

  for batch_id in range(batch_size):
    image = images[batch_id, :, :, :]
    # Draw a box for each box and label if weights is 1.

    transformed_boxes = []
    box_scores = []
    label_ids = []

    for box_id in range(nbboxes):
      box_weight = box_weights[batch_id, box_id]
      # If there is no box to draw, continue.
      if box_weight == 0.0:
        continue

      # TODO(vrv): When we predict heading, we should assert
      # that the length of bbox_data is 5.
      bbox_data = np.squeeze(bboxes[batch_id, box_id, :])
      if len(bbox_data) == 5:
        x, y, width, length, heading = bbox_data
      else:
        x, y, width, length = bbox_data
        heading = 0.0

      # Check whether we can draw the box.
      bbox = transform_util.Box2D(x, y, width, length, heading)
      ymin, xmin, ymax, xmax = bbox.Extrema()

      if ymin == 0 and xmin == 0 and ymax == 0 and xmax == 0:
        continue

      # TODO(vrv): Support drawing boxes on the edge of the
      # image.
      if (xmin < 0 or ymin < 0 or xmax >= image.shape[1] or
          ymax >= image.shape[0]):
        continue

      # We can draw a box on the image, so fill in the score, the boxes,
      # and the label.
      transformed_boxes.append([x, y, width, length, heading])
      box_scores.append(box_weight)
      label_ids.append(labels[batch_id, box_id])

    if transformed_boxes:
      transformed_boxes = np.stack(transformed_boxes, axis=0)
      scores = None if groundtruth else np.array(box_scores)
      text_loc = 'TOP' if groundtruth else 'BOTTOM'
      VisualizeBoxes(
          image=image,
          boxes=transformed_boxes,
          classes=label_ids,
          scores=scores,
          class_id_to_name=class_id_to_name,
          groundtruth_box_visualization_color='cyan',
          skip_scores=groundtruth,
          skip_labels=False,
          text_loc=text_loc)
  return images


def DrawTrajectory(image, bboxes, masks, labels, is_groundtruth):
  """Draw the trajectory of bounding boxes on 'image'.

  Args:
    image: The uint8 image array to draw on.  Assumes [1000, 500, 3] input with
      RGB value ranges.
    bboxes: A [num_steps, num_objects, 5] float array containing the bounding
      box information over a sequence of steps.  bboxes are expected to be in
      car coordinates.
    masks: A [num_steps, num_objects] integer array indicating whether the
      corresponding bbox entry in bboxes is present (1 = present).
    labels: A [num_steps, num_objects] integer label indicating which class is
      being predicted.  Used for colorizing based on labels.
    is_groundtruth: True if the scene is the groundtruth vs. the predicted.

  Returns:
    The updated image array.
  """
  image = Image.fromarray(np.uint8(image)).convert('RGB')
  draw = ImageDraw.Draw(image)

  try:
    font = ImageFont.truetype('arial.ttf', 20)
  except IOError:
    font = ImageFont.load_default()

  pixels_per_meter = 10.
  image_ref_x = 250.
  image_ref_y = 750.
  car_to_image_transform = transform_util.MakeCarToImageTransform(
      pixels_per_meter=pixels_per_meter,
      image_ref_x=image_ref_x,
      image_ref_y=image_ref_y,
      flip_axes=True)

  # Iterate over each object and produce the series of visualized trajectories
  # over time.
  for object_idx in range(bboxes.shape[1]):
    # Annotate the box with the class type
    label = labels[0, object_idx]

    # Choose a label_consistent color.
    color = PIL_COLOR_LIST[label % len(PIL_COLOR_LIST)]
    # Make predictions white.
    if not is_groundtruth:
      color = 'white'

    # Convert string color name to RGB so we can manipulate it.
    color_rgb = ImageColor.getrgb(color)

    # For each sample, extract the data, transform to image coordinates, and
    # store in centroids.
    centroids = []
    for time in range(bboxes.shape[0]):
      if masks[time, object_idx] == 0:
        continue

      center_x, center_y, width, height, heading = bboxes[time, object_idx, :]

      # Compute the new heading.
      heading = transform_util.TransformHeading(car_to_image_transform, heading)

      # Transform from car to image coords.
      x, y, _ = transform_util.TransformPoint(car_to_image_transform, center_x,
                                              center_y, 0.0)

      # Hack to scale from meters to pixels.
      width *= pixels_per_meter
      height *= pixels_per_meter

      # Collect the centroids of all of the points.
      centroids.append((x, y, heading))

      # Draw the groundtruth bounding box at the first timestep.
      if is_groundtruth and time == 0:
        # Draw a rectangle
        rect = MakeRectangle(height, width, heading, offset=(x, y))
        rect += [rect[0]]
        draw.line(rect, fill=color_rgb, width=4)

        delta = 20

        # Annotate the box with the object index
        draw.text((x + delta, y + delta),
                  str(object_idx),
                  fill='white',
                  font=font)

        # Draw a callout
        draw.line([(x, y), (x + delta, y + delta)], fill='white', width=1)

    # Extract the point pairs from centroids and draw a line through them.
    point_pairs = []
    for (x, y, heading) in centroids:
      point_pairs.append((x, y))
    if point_pairs:
      draw.line(point_pairs, width=4, fill=color_rgb)

    # Draw the centroids.
    triangle_color_rgb = color_rgb
    for i, (x, y, heading) in enumerate(centroids):
      if i == 0:
        # Draw the heading for the first timestep.
        scale = 25 if is_groundtruth else 15
        DrawHeadingTriangle(draw, x, y, heading, triangle_color_rgb, scale)
      else:
        # Draw a circle for the centroids of other timesteps.
        outline_color = color_rgb
        circle_size = 5 if is_groundtruth else 3
        DrawCircle(
            draw,
            x,
            y,
            fill=triangle_color_rgb,
            outline=outline_color,
            circle_size=circle_size)

      # Desaturate the color with every timestep.
      increment = 45  # Allow this to be modified?
      triangle_color_rgb = (triangle_color_rgb[0] - increment,
                            triangle_color_rgb[1] - increment,
                            triangle_color_rgb[2] - increment)

  return np.array(image)


def GetTrajectoryComparison(gt_bboxes, gt_masks, gt_labels, pred_bboxes,
                            pred_masks, pred_labels):
  """Draw a trajectory comparison of groundtruth and predicted.

  Args:
    gt_bboxes: A [batch_size, num_steps, num_objects, 5] float array containing
      the bounding box information over a sequence of steps.
    gt_masks: A [batch_size, num_steps, num_objects] integer array indicating
      whether the corresponding bbox entry in bboxes is present (1 = present).
    gt_labels: A [batch_size, num_steps, num_objects] integer label indicating
      which class is being predicted.  Used for colorizing based on labels.
    pred_bboxes: A [batch_size, num_steps, num_objects, 5] float array
      containing the bounding box information over a sequence of steps.
    pred_masks: A [batch_size, num_steps, num_objects] integer array indicating
      whether the corresponding bbox entry in bboxes is present (1 = present).
    pred_labels: A [batch_size, num_steps, num_objects] integer label indicating
      which class is being predicted.  Used for colorizing based on labels.

  Returns:
    images: A np.uint8 images array that can be displayed of size
      [batch_size, 1000, 500, 3].
  """
  batch_size = gt_bboxes.shape[0]
  images = np.zeros([batch_size, 1000, 500, 3], dtype=np.uint8)
  for b in range(batch_size):
    images[b] = DrawTrajectory(
        images[b], gt_bboxes[b], gt_masks[b], gt_labels[b], is_groundtruth=True)
    images[b] = DrawTrajectory(
        images[b],
        pred_bboxes[b],
        pred_masks[b],
        pred_labels[b],
        is_groundtruth=False)
  return images
